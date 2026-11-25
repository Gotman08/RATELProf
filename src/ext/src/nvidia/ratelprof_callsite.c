/**
 * @file ratelprof_callsite.c
 * @brief Best-effort resolution of the user call site of a CUDA API call.
 *
 * With the interposition approach of the AMD backend, the call site is simply
 * __builtin_return_address(0) of the wrapper. With CUPTI callbacks the callback
 * runs deep inside the CUDA libraries, so the call stack is unwound and the
 * first frame that does not belong to the CUDA stack (libcupti, libcuda,
 * libcudart, RATELProf itself, or the CUDA runtime statically linked in the
 * executable) is returned.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <execinfo.h>
#include <link.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <ratelprof.h>
#include "ratelprof_ext.h"
#include "ratelprof_ext/ratelprof_cupti.h"

#define RP_MAX_FRAMES 48
#define RP_MAX_RANGES 1024

typedef struct { uintptr_t lo, hi; uint8_t ignore; } rp_range_t;
typedef struct { size_t n; rp_range_t r[RP_MAX_RANGES]; } rp_ranges_t;

static _Atomic(rp_ranges_t*) ranges = NULL;
static pthread_mutex_t refresh_lock = PTHREAD_MUTEX_INITIALIZER;

/* Static CUDA runtime symbols of the main executable (sorted). */
typedef struct { uintptr_t lo, hi; } rp_sym_t;
static rp_sym_t* cudart_syms = NULL;
static size_t    cudart_nsyms = 0;
static uintptr_t exe_lo = 0, exe_hi = 0;

static const char* ignored_dso[] = {
    "libcupti", "libcuda.so", "libcudart", "libnvidia-", "libratelprof", "libplugin_rprofrep",
    "libnvToolsExt", "libnvperf", "libcudadebugger", NULL
};

static int is_ignored_dso(const char* path)
{
    if (!path) return 0;
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    for (int i = 0; ignored_dso[i]; i++)
        if (strstr(base, ignored_dso[i]) == base) return 1;
    return 0;
}

static int is_cudart_symbol(const char* name)
{
    if (!name) return 0;
    if (strncmp(name, "cuda", 4) == 0) return 1;            /* cudaMalloc, cudaLaunchKernel... */
    if (strncmp(name, "__cuda", 6) == 0) return 1;          /* __cudaRegisterFatBinary...      */
    if (strncmp(name, "__cudart", 8) == 0) return 1;
    if (strncmp(name, "cudart", 6) == 0) return 1;
    if (strncmp(name, "__device_stub_", 14) == 0) return 1; /* nvcc generated launch stubs     */
    if (strncmp(name, "_ZN6cudart", 10) == 0) return 1;     /* cudart:: internals              */
    if (strncmp(name, "nvtx", 4) == 0) return 1;            /* NVTX v3 inline implementation   */
    if (strncmp(name, "_ZL", 3) == 0 && strstr(name, "cudart")) return 1;
    if (strncmp(name, "_Z", 2) == 0) {                      /* cudaLaunchKernel<T>(...) etc.   */
        const char* p = name + 2;
        while (*p >= '0' && *p <= '9') p++;
        if (strncmp(p, "cuda", 4) == 0 || strncmp(p, "__device_stub", 13) == 0) return 1;
    }
    return 0;
}

static int rp_phdr_cb(struct dl_phdr_info* info, size_t size, void* data)
{
    (void)size;
    rp_ranges_t* t = (rp_ranges_t*)data;
    const char* name = (info->dlpi_name && info->dlpi_name[0]) ? info->dlpi_name : NULL;
    uint8_t ignore = (uint8_t)is_ignored_dso(name);
    for (int i = 0; i < info->dlpi_phnum && t->n < RP_MAX_RANGES; i++) {
        const ElfW(Phdr)* ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_LOAD || !(ph->p_flags & PF_X)) continue;
        uintptr_t lo = (uintptr_t)(info->dlpi_addr + ph->p_vaddr);
        t->r[t->n].lo = lo;
        t->r[t->n].hi = lo + ph->p_memsz;
        t->r[t->n].ignore = ignore;
        t->n++;
        if (!name && exe_lo == 0) { exe_lo = lo; exe_hi = lo + ph->p_memsz; }
    }
    return 0;
}

static void rp_refresh_ranges(void)
{
    pthread_mutex_lock(&refresh_lock);
    rp_ranges_t* t = (rp_ranges_t*)calloc(1, sizeof(rp_ranges_t));
    if (t) {
        dl_iterate_phdr(rp_phdr_cb, t);
        /* The previous table is leaked on purpose: other threads may read it. */
        atomic_store_explicit(&ranges, t, memory_order_release);
    }
    pthread_mutex_unlock(&refresh_lock);
}

static int rp_lookup(uintptr_t a)
{
    rp_ranges_t* t = atomic_load_explicit(&ranges, memory_order_acquire);
    if (!t) return -1;
    for (size_t i = 0; i < t->n; i++)
        if (a >= t->r[i].lo && a < t->r[i].hi) return t->r[i].ignore;
    return -1;
}

static int cmp_sym(const void* a, const void* b)
{
    const rp_sym_t* x = (const rp_sym_t*)a;
    const rp_sym_t* y = (const rp_sym_t*)b;
    return (x->lo > y->lo) - (x->lo < y->lo);
}

/* Read the .symtab of the main executable to know where the statically linked
 * CUDA runtime lives (its symbols are not visible to dladdr). */
static void rp_load_exe_cudart_symbols(void)
{
    int fd = open("/proc/self/exe", O_RDONLY);
    if (fd < 0) return;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(ElfW(Ehdr))) { close(fd); return; }
    uint8_t* m = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return;

    const ElfW(Ehdr)* eh = (const ElfW(Ehdr)*)m;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0 || eh->e_shoff == 0) { munmap(m, st.st_size); return; }
    const ElfW(Shdr)* sh = (const ElfW(Shdr)*)(m + eh->e_shoff);

    /* Load bias of a PIE executable: difference between the runtime address of
     * the first executable PT_LOAD and its link-time address. */
    uintptr_t bias = 0;
    if (eh->e_type == ET_DYN) {
        const ElfW(Phdr)* ph = (const ElfW(Phdr)*)(m + eh->e_phoff);
        for (int i = 0; i < eh->e_phnum; i++) {
            if (ph[i].p_type == PT_LOAD && (ph[i].p_flags & PF_X)) { bias = exe_lo - ph[i].p_vaddr; break; }
        }
    }

    size_t cap = 256, n = 0;
    rp_sym_t* syms = (rp_sym_t*)malloc(cap * sizeof(rp_sym_t));
    for (int i = 0; syms && i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_SYMTAB) continue;
        const ElfW(Sym)* sym = (const ElfW(Sym)*)(m + sh[i].sh_offset);
        size_t count = sh[i].sh_size / sizeof(ElfW(Sym));
        const char* strtab = (const char*)(m + sh[sh[i].sh_link].sh_offset);
        for (size_t j = 0; j < count; j++) {
            if (ELF64_ST_TYPE(sym[j].st_info) != STT_FUNC || sym[j].st_value == 0) continue;
            if (!is_cudart_symbol(strtab + sym[j].st_name)) continue;
            if (n == cap) { cap *= 2; rp_sym_t* t = (rp_sym_t*)realloc(syms, cap * sizeof(rp_sym_t)); if (!t) break; syms = t; }
            syms[n].lo = bias + sym[j].st_value;
            syms[n].hi = syms[n].lo + (sym[j].st_size ? sym[j].st_size : 1);
            n++;
        }
    }
    munmap(m, st.st_size);
    if (syms && n) {
        qsort(syms, n, sizeof(rp_sym_t), cmp_sym);
        cudart_syms = syms;
        cudart_nsyms = n;
    } else {
        free(syms);
    }
}

static int rp_in_exe_cudart(uintptr_t a)
{
    if (!cudart_syms || a < exe_lo || a >= exe_hi) return 0;
    size_t lo = 0, hi = cudart_nsyms;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (a < cudart_syms[mid].lo) hi = mid;
        else if (a >= cudart_syms[mid].hi) lo = mid + 1;
        else return 1;
    }
    return 0;
}

void ratelprof_nv_callsite_init(void)
{
    void* warmup[4];
    backtrace(warmup, 4); /* loads the unwinder outside of any CUDA callback */
    rp_refresh_ranges();
    rp_load_exe_cudart_symbols();
}

void* ratelprof_nv_find_callsite(void)
{
    void* frames[RP_MAX_FRAMES];
    int n = backtrace(frames, RP_MAX_FRAMES);
    int refreshed = 0;
    for (int i = 1; i < n; i++) {
        uintptr_t a = (uintptr_t)frames[i];
        int ign = rp_lookup(a);
        if (ign < 0 && !refreshed) { rp_refresh_ranges(); refreshed = 1; ign = rp_lookup(a); }
        if (ign == 1) continue;
        if (rp_in_exe_cudart(a)) continue;
        if (ign < 0 || (a < exe_lo || a >= exe_hi)) {
            /* Shared library with a visible CUDA runtime symbol (cudart linked statically in a .so). */
            Dl_info info;
            if (dladdr((void*)a, &info) && is_cudart_symbol(info.dli_sname)) continue;
        }
        /* Return address -> address of the call instruction. */
        return (void*)(a - 1);
    }
    return NULL;
}
