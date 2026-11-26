/**
 * @file ratelprof_static_hooks.c
 * @brief Interception beyond LD_PRELOAD. See ratelprof_static_hooks.h.
 *
 * Locking: glibc's dlsym/dlopen take the loader lock, and library constructors
 * run with that lock held and may call dlsym (hence our dlsym wrapper). So no
 * dlsym/dlopen is ever called while `lock` is held: wrapper addresses and the
 * demangler are resolved once, at activation.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <dlfcn.h>
#include <link.h>
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ratelprof.h"
#include "ratelprof/ratelprof_hook.h"
#include "ratelprof/ratelprof_static_hooks.h"
#include "utils/logger.h"

extern void fallback(void);

#ifdef RATELPROF_VENDOR_NVIDIA
/* Defined with the wrappers (generated): weak so that the core library links alone. */
extern const ratelprof_method_hook_t ratelprof_omp_tgt_rtl_methods[] __attribute__((weak));
#endif

/* ------------------------------------------------------------------------- */
/* Real dlsym                                                                 */
/* ------------------------------------------------------------------------- */

typedef void* (*rp_dlsym_t)(void*, const char*);

void* ratelprof_real_dlsym(void* handle, const char* symbol)
{
    static rp_dlsym_t fn = NULL;
    if (!fn) {
        /* Base version of dlsym: x86_64 GLIBC_2.2.5, aarch64/ppc64le GLIBC_2.17;
         * glibc >= 2.34 also exports GLIBC_2.34. */
        static const char* versions[] = { "GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5", "GLIBC_2.0", NULL };
        for (int i = 0; versions[i] && !fn; i++) fn = (rp_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", versions[i]);
    }
    return fn ? fn(handle, symbol) : NULL;
}

/* ------------------------------------------------------------------------- */
/* State                                                                      */
/* ------------------------------------------------------------------------- */

typedef struct {
    const char* name;      /* domain function name (NULL: empty slot) */
    int         domain;
    int         id;
    void*       wrapper;   /* RATELProf wrapper exported under that name */
} rp_fun_t;

typedef struct {
    uintptr_t base;
    char      name[512];
    bool      alive;
} rp_obj_t;

typedef struct {
    uintptr_t addr;        /* patched function (0: removed) */
    void*     tramp;
    void*     impl_set;    /* value written to api_fn, if any */
    int       obj, domain, id;
} rp_hooked_t;

typedef struct { void* real; void* stub; } rp_stub_t;

typedef char* (*rp_demangle_t)(const char*, char*, size_t*, int*);

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static bool            active = false;   /* accessed with __atomic builtins */
#define IS_ACTIVE() __atomic_load_n(&active, __ATOMIC_ACQUIRE)
static bool            dom_on[RATELPROF_NB_DOMAIN];
static void**          dom_default[RATELPROF_NB_DOMAIN];  /* impl reachable through dlsym at start */
/* First exported definition seen in a loaded object (libraries loaded with
 * RTLD_LOCAL are invisible to dlsym(RTLD_NEXT)), with its object index. */
static void**          dom_cand[RATELPROF_NB_DOMAIN];
static int*            dom_cand_obj[RATELPROF_NB_DOMAIN];

static rp_fun_t* funs = NULL;
static size_t    funs_cap = 0;

#define RP_MAX_OBJS 4096
static rp_obj_t    objs[RP_MAX_OBJS];
static int         nobjs = 0;
static unsigned long long seen_adds = ~0ull, seen_subs = ~0ull;

static rp_hooked_t* hooked = NULL;
static size_t       nhooked = 0, cap_hooked = 0;

static rp_stub_t*   stubs = NULL;
static size_t       nstubs = 0, cap_stubs = 0;

static uintptr_t     self_lo = 0, self_hi = 0, self_base = 0;
static rp_demangle_t rp_demangle_fn = NULL;
static bool          verbose = false;

static inline bool in_self(const void* p) { return (uintptr_t)p >= self_lo && (uintptr_t)p < self_hi; }

static uint64_t fnv1a(const char* s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return h;
}

static const rp_fun_t* lookup(const char* name)
{
    if (!funs_cap) return NULL;
    size_t i = (size_t)fnv1a(name) & (funs_cap - 1);
    for (;;) {
        const rp_fun_t* f = &funs[i];
        if (!f->name) return NULL;
        if (strcmp(f->name, name) == 0) return f;
        i = (i + 1) & (funs_cap - 1);
    }
}

void ratelprof_static_hooks_domain_enabled(int domain, bool enabled)
{
    if (domain >= 0 && domain < RATELPROF_NB_DOMAIN) dom_on[domain] = enabled;
}

/* ------------------------------------------------------------------------- */
/* API table updates                                                          */
/* ------------------------------------------------------------------------- */

/* Make @p impl the implementation of (domain, id) when the table has none, or
 * when the current one is the function that has just been patched. */
static void* set_impl_if(int domain, int id, void* patched, void* impl)
{
    ratelprof_api_table_t* t = domains_data[domain].api_table_addr;
    if (!t || !t->api_fn) return NULL;
    void* cur = t->api_fn[id];
    if (cur != (void*)fallback && cur != patched) return NULL;
    if (t->api_ptr[id] == cur) t->api_ptr[id] = impl;
    t->api_fn[id] = impl;
    return impl;
}

/* ------------------------------------------------------------------------- */
/* ELF symbol tables                                                          */
/* ------------------------------------------------------------------------- */

static bool already_hooked(uintptr_t addr)
{
    for (size_t i = 0; i < nhooked; i++) if (hooked[i].addr == addr) return true;
    return false;
}

static void add_hooked(uintptr_t addr, void* tramp, void* impl_set, int obj, int domain, int id)
{
    if (nhooked == cap_hooked) {
        size_t n = cap_hooked ? cap_hooked * 2 : 256;
        rp_hooked_t* h = (rp_hooked_t*)realloc(hooked, n * sizeof(*h));
        if (!h) return;
        hooked = h; cap_hooked = n;
    }
    hooked[nhooked++] = (rp_hooked_t){ addr, tramp, impl_set, obj, domain, id };
}

static void hook_c_function(int oi, const char* name, uintptr_t addr, size_t size,
                            const rp_fun_t* f, bool exported, bool is_exe)
{
    if (already_hooked(addr) || in_self((void*)addr) || !f->wrapper || (void*)addr == f->wrapper) return;
    void* deflt = dom_default[f->domain] ? dom_default[f->domain][f->id] : NULL;

    if (exported && !is_exe && dom_cand[f->domain] && !dom_cand[f->domain][f->id]) {
        dom_cand[f->domain][f->id] = (void*)addr;
        dom_cand_obj[f->domain][f->id] = oi;
    }

    bool want;
#ifdef RATELPROF_VENDOR_NVIDIA
    if (f->domain == RATELPROF_DOMAIN_OMP_TGT_RTL) want = true;   /* plugins: always */
    else
#endif
    want = (void*)addr != deflt && (is_exe || !exported);       /* static copies */
    if (!want) return;

    const char* why = NULL;
    void* tramp = ratelprof_hook_function((void*)addr, size, f->wrapper, &why);
    if (!tramp) {
        LOG(verbose ? LOG_LEVEL_INFO : LOG_LEVEL_WARN, "Cannot intercept '%s' in %s: %s.\n", name, objs[oi].name, why ? why : "?");
        return;
    }
    void* impl_set = set_impl_if(f->domain, f->id, (void*)addr, tramp);
    add_hooked(addr, tramp, impl_set, oi, f->domain, f->id);
    LOG_IF(verbose, LOG_LEVEL_INFO, "Intercepting '%s' in %s (%s).\n", name, objs[oi].name,
           is_exe ? "executable" : (exported ? "exported" : "static copy"));
}

#ifdef RATELPROF_VENDOR_NVIDIA
static const char RP_PLUGIN_PREFIX[] = "_ZN4llvm3omp6target6plugin15GenericPluginTy";
static const char RP_PLUGIN_SCOPE[]  = "llvm::omp::target::plugin::GenericPluginTy::";

static void hook_plugin_method(int oi, const char* mangled, uintptr_t addr, size_t size)
{
    if (!dom_on[RATELPROF_DOMAIN_OMP_TGT_RTL] || !rp_demangle_fn || !ratelprof_omp_tgt_rtl_methods ||
        already_hooked(addr)) return;
    int st = 0;
    char* dem = rp_demangle_fn(mangled, NULL, NULL, &st);
    if (!dem || st != 0) { free(dem); return; }
    const char* m = strstr(dem, RP_PLUGIN_SCOPE);
    char* lp = m ? strchr(m, '(') : NULL;
    char* rp = lp ? strrchr(lp, ')') : NULL;
    if (!m || !lp || !rp || rp[1] != '\0') { free(dem); return; }   /* const methods excluded */
    m += sizeof(RP_PLUGIN_SCOPE) - 1;
    size_t mlen = (size_t)(lp - m);
    *rp = '\0';
    const char* params = lp + 1;

    for (const ratelprof_method_hook_t* h = ratelprof_omp_tgt_rtl_methods; h->name; h++) {
        if (strlen(h->name) != mlen || strncmp(h->name, m, mlen) != 0) continue;
        if (strcmp(h->params, params) != 0) {
            LOG(verbose ? LOG_LEVEL_INFO : LOG_LEVEL_WARN, "Plugin method GenericPluginTy::%s(%s) does not match the "
                "omp-target prototype (%s): not traced.\n", h->name, params, h->params);
            break;
        }
        const char* why = NULL;
        void* tramp = ratelprof_hook_function((void*)addr, size, h->entry, &why);
        if (!tramp) {
            LOG(verbose ? LOG_LEVEL_INFO : LOG_LEVEL_WARN, "Cannot intercept GenericPluginTy::%s in %s: %s.\n", h->name, objs[oi].name, why ? why : "?");
            break;
        }
        void* impl_set = set_impl_if(RATELPROF_DOMAIN_OMP_TGT_RTL, h->id, NULL, h->impl);
        if (!impl_set) {   /* a C entry point already provides it: still route the method */
            ratelprof_api_table_t* t = domains_data[RATELPROF_DOMAIN_OMP_TGT_RTL].api_table_addr;
            if (t->api_ptr[h->id] == t->api_fn[h->id]) t->api_ptr[h->id] = h->impl;
            t->api_fn[h->id] = h->impl;
            impl_set = h->impl;
        }
        add_hooked(addr, tramp, impl_set, oi, RATELPROF_DOMAIN_OMP_TGT_RTL, h->id);
        LOG_IF(verbose, LOG_LEVEL_INFO, "Intercepting GenericPluginTy::%s in %s.\n", h->name, objs[oi].name);
        break;
    }
    free(dem);
}
#endif

/* Scan the symbol table of object @p oi and patch the functions to intercept. */
static void scan_object(int oi, const char* path, uintptr_t base, bool is_exe)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || (size_t)sb.st_size < sizeof(Elf64_Ehdr)) { close(fd); return; }
    size_t fsize = (size_t)sb.st_size;
    const uint8_t* img = (const uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (img == MAP_FAILED) return;

    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)img;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0 || eh->e_ident[EI_CLASS] != ELFCLASS64 ||
        eh->e_shoff == 0 || eh->e_shentsize != sizeof(Elf64_Shdr) ||
        eh->e_shoff + (size_t)eh->e_shnum * sizeof(Elf64_Shdr) > fsize) {
        munmap((void*)img, fsize);
        return;
    }
    const Elf64_Shdr* sh = (const Elf64_Shdr*)(img + eh->e_shoff);
    const Elf64_Shdr* symtab = NULL;
    for (int i = 0; i < eh->e_shnum; i++) if (sh[i].sh_type == SHT_SYMTAB) { symtab = &sh[i]; break; }
    if (!symtab)
        for (int i = 0; i < eh->e_shnum; i++) if (sh[i].sh_type == SHT_DYNSYM) { symtab = &sh[i]; break; }
    if (!symtab || symtab->sh_link >= eh->e_shnum ||
        symtab->sh_offset + symtab->sh_size > fsize) { munmap((void*)img, fsize); return; }
    const Elf64_Shdr* strsh = &sh[symtab->sh_link];
    if (strsh->sh_offset + strsh->sh_size > fsize) { munmap((void*)img, fsize); return; }

    const Elf64_Sym* syms = (const Elf64_Sym*)(img + symtab->sh_offset);
    size_t nsyms = symtab->sh_size / sizeof(Elf64_Sym);
    const char* strtab = (const char*)(img + strsh->sh_offset);
    uintptr_t bias = (eh->e_type == ET_EXEC) ? 0 : base;

    char buf[512];
    size_t nfunc = 0, nmatch = 0;
    for (size_t s = 0; s < nsyms; s++) {
        const Elf64_Sym* y = &syms[s];
        if (ELF64_ST_TYPE(y->st_info) != STT_FUNC || y->st_shndx == SHN_UNDEF || y->st_value == 0) continue;
        if (y->st_name >= strsh->sh_size) continue;
        const char* name = strtab + y->st_name;
        uintptr_t addr = bias + y->st_value;
#ifdef RATELPROF_VENDOR_NVIDIA
        if (name[0] == '_' && strncmp(name, RP_PLUGIN_PREFIX, sizeof(RP_PLUGIN_PREFIX) - 1) == 0) {
            hook_plugin_method(oi, name, addr, (size_t)y->st_size);
            continue;
        }
#endif
        const char* at = strchr(name, '@');
        if (at) {
            size_t n = (size_t)(at - name);
            if (n >= sizeof(buf)) continue;
            memcpy(buf, name, n); buf[n] = '\0';
            name = buf;
        }
        nfunc++;
        const rp_fun_t* f = lookup(name);
        if (!f) continue;
        nmatch++;
        unsigned char bind = ELF64_ST_BIND(y->st_info);
        unsigned char vis  = ELF64_ST_VISIBILITY(y->st_other);
        bool exported = bind != STB_LOCAL && (vis == STV_DEFAULT || vis == STV_PROTECTED);
        hook_c_function(oi, name, addr, (size_t)y->st_size, f, exported, is_exe);
    }
    LOG_IF(verbose, LOG_LEVEL_INFO, "in-memory interception: %s: %zu symbols, %zu functions, %zu traced names (%s).\n",
           path, nsyms, nfunc, nmatch, symtab->sh_type == SHT_SYMTAB ? "symtab" : "dynsym");
    munmap((void*)img, fsize);
}

/* ------------------------------------------------------------------------- */
/* Loaded objects                                                             */
/* ------------------------------------------------------------------------- */

typedef struct { uintptr_t base; char name[512]; bool is_exe; } rp_cand_t;
typedef struct { rp_cand_t* c; int n, cap; } rp_cands_t;

static int collect_cb(struct dl_phdr_info* info, size_t size, void* data)
{
    (void)size;
    rp_cands_t* cs = (rp_cands_t*)data;
    if (cs->n == cs->cap) return 0;
    rp_cand_t* c = &cs->c[cs->n];
    c->base = (uintptr_t)info->dlpi_addr;
    c->is_exe = (cs->n == 0);
    const char* nm = info->dlpi_name;
    if (c->is_exe || !nm || !nm[0]) {
        if (cs->n != 0) return 0;     /* vdso */
        snprintf(c->name, sizeof(c->name), "%s", "/proc/self/exe");
    } else {
        snprintf(c->name, sizeof(c->name), "%s", nm);
    }
    cs->n++;
    return 0;
}

static bool skip_object(const char* name, uintptr_t base)
{
    if (base == self_base && base != 0) return true;
    static const char* skip[] = { "linux-vdso", "ld-linux", "/libc.so", "/libm.so", "/libpthread",
                                  "/libdl.so", "/librt.so", "libcuda.so", "libcupti", "libnvidia-",
                                  "/libratelprof", "/libplugin_rprofrep",
                                  "/libstdc++", "/libgcc_s", NULL };
    for (int i = 0; skip[i]; i++) if (strstr(name, skip[i])) return true;
    return false;
}

/* One full pass: forget unloaded objects, scan new ones. Called with `lock` held. */
static void scan_pass(void)
{
    static rp_cand_t cand[RP_MAX_OBJS];
    rp_cands_t cs = { cand, 0, RP_MAX_OBJS };
    dl_iterate_phdr(collect_cb, &cs);

    for (int i = 0; i < nobjs; i++) objs[i].alive = false;
    for (int k = 0; k < cs.n; k++) {
        int found = -1;
        for (int i = 0; i < nobjs; i++)
            if (objs[i].base == cand[k].base && strcmp(objs[i].name, cand[k].name) == 0) { found = i; break; }
        if (found >= 0) { objs[found].alive = true; continue; }
        if (nobjs == RP_MAX_OBJS) continue;
        int oi = nobjs++;
        objs[oi].base = cand[k].base;
        snprintf(objs[oi].name, sizeof(objs[oi].name), "%s", cand[k].name);
        objs[oi].alive = true;
        if (!skip_object(cand[k].name, cand[k].base)) {
            LOG_IF(verbose, LOG_LEVEL_INFO, "in-memory interception: scanning %s (base %p).\n", cand[k].name, (void*)cand[k].base);
            scan_object(oi, cand[k].name, cand[k].base, cand[k].is_exe);
        }
    }
    /* Unloaded objects: forget their definitions... */
    for (int d = 0; d < RATELPROF_NB_DOMAIN; d++) {
        ratelprof_api_table_t* t = domains_data[d].api_table_addr;
        if (!dom_cand[d] || !t) continue;
        for (size_t id = 0; id < t->size; id++) {
            if (!dom_cand[d][id] || objs[dom_cand_obj[d][id]].alive) continue;
            if (t->api_fn[id] == dom_cand[d][id]) {
                if (t->api_ptr[id] == t->api_fn[id]) t->api_ptr[id] = (void*)fallback;
                t->api_fn[id] = (void*)fallback;
            }
            dom_cand[d][id] = NULL;
        }
    }
    /* ... and their trampolines must not stay as implementations. */
    for (size_t h = 0; h < nhooked; h++) {
        rp_hooked_t* x = &hooked[h];
        if (!x->addr || objs[x->obj].alive) continue;
        ratelprof_api_table_t* t = domains_data[x->domain].api_table_addr;
        if (x->impl_set && t && t->api_fn[x->id] == x->impl_set) {
            if (t->api_ptr[x->id] == x->impl_set) t->api_ptr[x->id] = (void*)fallback;
            t->api_fn[x->id] = (void*)fallback;
        }
        x->addr = 0;
    }
    int w = 0;
    for (int i = 0; i < nobjs; i++) {
        if (!objs[i].alive) continue;
        if (w != i) {
            objs[w] = objs[i];
            for (size_t h = 0; h < nhooked; h++) if (hooked[h].addr && hooked[h].obj == i) hooked[h].obj = w;
            for (int d = 0; d < RATELPROF_NB_DOMAIN; d++) {
                ratelprof_api_table_t* t = domains_data[d].api_table_addr;
                if (!dom_cand[d] || !t) continue;
                for (size_t id = 0; id < t->size; id++)
                    if (dom_cand[d][id] && dom_cand_obj[d][id] == i) dom_cand_obj[d][id] = w;
            }
        }
        w++;
    }
    nobjs = w;
}

static int counters_cb(struct dl_phdr_info* info, size_t size, void* data)
{
    unsigned long long* c = (unsigned long long*)data;
    if (size >= offsetof(struct dl_phdr_info, dlpi_subs) + sizeof(info->dlpi_subs)) {
        c[0] = info->dlpi_adds;
        c[1] = info->dlpi_subs;
    }
    return 1;   /* first object is enough */
}

static void rescan_locked(void)
{
    unsigned long long c[2] = { 0, 0 };
    dl_iterate_phdr(counters_cb, c);
    if (c[0] == seen_adds && c[1] == seen_subs) return;
    seen_adds = c[0];
    seen_subs = c[1];
    scan_pass();
}

void ratelprof_static_hooks_rescan(void)
{
    if (!IS_ACTIVE()) return;
    if (pthread_mutex_trylock(&lock) != 0) return;   /* another thread is scanning */
    rescan_locked();
    pthread_mutex_unlock(&lock);
}

/* ------------------------------------------------------------------------- */
/* Activation                                                                 */
/* ------------------------------------------------------------------------- */

static int self_cb(struct dl_phdr_info* info, size_t size, void* data)
{
    (void)size; (void)data;
    uintptr_t me = (uintptr_t)&self_cb;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr)* p = &info->dlpi_phdr[i];
        if (p->p_type != PT_LOAD) continue;
        uintptr_t lo = info->dlpi_addr + p->p_vaddr, hi = lo + p->p_memsz;
        if (me >= lo && me < hi) {
            uintptr_t mlo = UINTPTR_MAX, mhi = 0;
            for (int j = 0; j < info->dlpi_phnum; j++) {
                const ElfW(Phdr)* q = &info->dlpi_phdr[j];
                if (q->p_type != PT_LOAD) continue;
                uintptr_t a = info->dlpi_addr + q->p_vaddr, b = a + q->p_memsz;
                if (a < mlo) mlo = a;
                if (b > mhi) mhi = b;
            }
            self_lo = mlo; self_hi = mhi; self_base = info->dlpi_addr;
            return 1;
        }
    }
    return 0;
}

void ratelprof_static_hooks_activate(void)
{
    const char* env = getenv("RATELPROF_STATIC_HOOKS");
    if (env && atoi(env) == 0) return;
    verbose = getenv("RATELPROF_STATIC_HOOKS_VERBOSE") && atoi(getenv("RATELPROF_STATIC_HOOKS_VERBOSE")) != 0;
    if (!ratelprof_hook_supported()) {
        LOG(LOG_LEVEL_WARN, "In-memory interception is not supported on this architecture: "
            "statically linked functions are not traced.\n");
        return;
    }
    dl_iterate_phdr(self_cb, NULL);

    /* Everything requiring the loader lock is done here, before locking. */
    Dl_info di;
    void* self_handle = NULL;
    if (dladdr((void*)&ratelprof_static_hooks_activate, &di) && di.dli_fname)
        self_handle = dlopen(di.dli_fname, RTLD_NOLOAD | RTLD_LAZY);
    rp_demangle_fn = (rp_demangle_t)ratelprof_real_dlsym(RTLD_DEFAULT, "__cxa_demangle");

    size_t count = 0;
    for (int d = 0; d < RATELPROF_NB_DOMAIN; d++)
        if (dom_on[d] && domains_data[d].api_table_addr && domains_data[d].api_table_addr->is_populate)
            count += domains_data[d].nb_function;
    LOG_IF(verbose, LOG_LEVEL_INFO, "in-memory interception: %zu traced functions, self=%p-%p, demangler=%s.\n",
           count, (void*)self_lo, (void*)self_hi, rp_demangle_fn ? "yes" : "no");
    if (count == 0) { if (self_handle) dlclose(self_handle); return; }

    size_t cap = 16;
    while (cap < 2 * count) cap <<= 1;
    rp_fun_t* tab = (rp_fun_t*)calloc(cap, sizeof(rp_fun_t));
    if (!tab) { if (self_handle) dlclose(self_handle); return; }
    for (int d = 0; d < RATELPROF_NB_DOMAIN; d++) {
        ratelprof_api_table_t* t = domains_data[d].api_table_addr;
        if (!dom_on[d] || !t || !t->is_populate) continue;
        dom_default[d] = (void**)calloc(t->size, sizeof(void*));
        dom_cand[d]     = (void**)calloc(t->size, sizeof(void*));
        dom_cand_obj[d] = (int*)calloc(t->size, sizeof(int));
        if (!dom_cand[d] || !dom_cand_obj[d]) { free(dom_cand[d]); free(dom_cand_obj[d]); dom_cand[d] = NULL; dom_cand_obj[d] = NULL; }
        for (size_t id = 0; id < t->size; id++) {
            const char* name = domains_data[d].get_funame_by_id((int)id);
            if (!name) continue;
            if (dom_default[d]) dom_default[d][id] = (t->api_fn[id] == (void*)fallback) ? NULL : t->api_fn[id];
            size_t i = (size_t)fnv1a(name) & (cap - 1);
            while (tab[i].name) i = (i + 1) & (cap - 1);
            tab[i].name = name;
            tab[i].domain = d;
            tab[i].id = (int)id;
            tab[i].wrapper = self_handle ? ratelprof_real_dlsym(self_handle, name) : NULL;
            if (tab[i].wrapper && !in_self(tab[i].wrapper)) tab[i].wrapper = NULL;
        }
    }
    if (self_handle) dlclose(self_handle);
    if (verbose) {
        size_t nw = 0;
        for (size_t i = 0; i < cap; i++) if (tab[i].name && tab[i].wrapper) nw++;
        LOG(LOG_LEVEL_INFO, "in-memory interception: %zu wrappers found (self handle %s).\n", nw, self_handle ? "ok" : "missing");
    }

    pthread_mutex_lock(&lock);
    funs = tab;
    funs_cap = cap;
    __atomic_store_n(&active, true, __ATOMIC_RELEASE);
    rescan_locked();
    size_t n = nhooked;
    pthread_mutex_unlock(&lock);
    if (n) LOG(LOG_LEVEL_INFO, "%zu function(s) intercepted in memory (statically linked or pre-resolved).\n", n);
}

void ratelprof_static_hooks_deactivate(void)
{
    __atomic_store_n(&active, false, __ATOMIC_RELEASE);
}

/* ------------------------------------------------------------------------- */
/* dlsym                                                                      */
/* ------------------------------------------------------------------------- */

void* ratelprof_static_hooks_dlsym(void* handle, const char* symbol, void* real, void* caller)
{
    if (!IS_ACTIVE() || !real || !symbol || handle == RTLD_NEXT || handle == RTLD_DEFAULT) return real;
    if (in_self(caller) || in_self(real)) return real;
    const rp_fun_t* f = lookup(symbol);
    if (!f || !f->wrapper || !dom_on[f->domain]) return real;

    void* out = real;
    pthread_mutex_lock(&lock);
    rescan_locked();                          /* the object was probably just loaded */
    if (!already_hooked((uintptr_t)real)) {
        for (size_t i = 0; i < nstubs; i++) if (stubs[i].real == real) { out = stubs[i].stub; goto done; }
        void* stub = ratelprof_hook_make_stub(real, f->wrapper);
        if (stub) {
            if (nstubs == cap_stubs) {
                size_t n = cap_stubs ? cap_stubs * 2 : 128;
                rp_stub_t* s = (rp_stub_t*)realloc(stubs, n * sizeof(*s));
                if (s) { stubs = s; cap_stubs = n; }
            }
            if (nstubs < cap_stubs) stubs[nstubs++] = (rp_stub_t){ real, stub };
            out = stub;
            /* First implementation seen for a function unresolved at start-up. */
            set_impl_if(f->domain, f->id, NULL, real);
        }
    }
done:
    pthread_mutex_unlock(&lock);
    return out;
}

/* ------------------------------------------------------------------------- */
/* Late resolution                                                            */
/* ------------------------------------------------------------------------- */

void* ratelprof_api_resolve_late(ratelprof_api_table_t* t, ratelprof_api_id_t id, const char* name)
{
    /* The library may have been loaded after start-up: look again (no lock held). */
    void* f = ratelprof_real_dlsym(RTLD_NEXT, name);
    if (f && in_self(f)) f = NULL;
    pthread_mutex_lock(&lock);
    if (t->api_fn[id] == (void*)fallback) {
        if (f) {
            if (t->api_ptr[id] == (void*)fallback) t->api_ptr[id] = f;
            t->api_fn[id] = f;
        } else if (IS_ACTIVE()) {
            rescan_locked();                  /* static copy or RTLD_LOCAL library */
            int d = (int)t->domain;
            if (t->api_fn[id] == (void*)fallback && d >= 0 && d < RATELPROF_NB_DOMAIN &&
                dom_cand[d] && dom_cand[d][id]) {
                if (t->api_ptr[id] == (void*)fallback) t->api_ptr[id] = dom_cand[d][id];
                t->api_fn[id] = dom_cand[d][id];
            }
        }
    }
    void* out = t->api_fn[id];
    pthread_mutex_unlock(&lock);
    return out;
}
