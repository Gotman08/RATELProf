/**
 * @file ratelprof_hook.c
 * @brief Inline function hooking (aarch64, x86_64). See ratelprof_hook.h.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>

#include "ratelprof/ratelprof_hook.h"

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

__thread void* ratelprof_hook_slot __attribute__((tls_model("initial-exec"))) = NULL;

static pthread_mutex_t hook_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t          hook_count = 0;

size_t ratelprof_hook_count(void) { return hook_count; }

#if defined(__aarch64__) || defined(__x86_64__)
bool ratelprof_hook_supported(void) { return true; }
#else
bool ratelprof_hook_supported(void) { return false; }
#endif

/* ------------------------------------------------------------------------- */
/* Executable memory close to a target                                        */
/* ------------------------------------------------------------------------- */

#if defined(__aarch64__)
#define RP_HOOK_RANGE  ((intptr_t)120 << 20)   /* B reaches +-128 MiB */
#else
#define RP_HOOK_RANGE  ((intptr_t)1900 << 20)  /* rel32 reaches +-2 GiB */
#endif
#define RP_BLOCK_SIZE  ((size_t)64 << 10)
#define RP_MAX_BLOCKS  1024
#define RP_MAX_MAPS    8192

typedef struct {
    uint8_t* base;
    size_t   used;
    bool     rwx;       /* mapped RWX: can be extended later */
} rp_block_t;

static rp_block_t blocks[RP_MAX_BLOCKS];
static int        nblocks = 0;

typedef struct { uintptr_t start, end; bool heap; } rp_range_t;

static int read_maps(rp_range_t* out, int max)
{
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512];
    int n = 0;
    while (n < max && fgets(line, sizeof(line), f)) {
        unsigned long s, e;
        if (sscanf(line, "%lx-%lx", &s, &e) == 2) {
            out[n].start = s; out[n].end = e;
            out[n].heap = strstr(line, "[heap]") != NULL;
            n++;
        }
    }
    fclose(f);
    return n;
}

static uintptr_t mmap_min_addr(void)
{
    unsigned long v = 65536;
    FILE* f = fopen("/proc/sys/vm/mmap_min_addr", "r");
    if (f) { if (fscanf(f, "%lu", &v) != 1) v = 65536; fclose(f); }
    return v < 65536 ? 65536 : (uintptr_t)v;
}

static inline intptr_t dist(uintptr_t a, uintptr_t b) { return a > b ? (intptr_t)(a - b) : (intptr_t)(b - a); }

/* Map RP_BLOCK_SIZE bytes within @p range of @p near (anywhere if 0). */
static rp_block_t* new_block(uintptr_t near, intptr_t range)
{
    if (nblocks == RP_MAX_BLOCKS) return NULL;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void* p = MAP_FAILED;
    bool rwx = true;

    if (near) {
        static rp_range_t maps[RP_MAX_MAPS + 1];
        /* maps[0]: pseudo-mapping ending at mmap_min_addr, so that the gap
         * below the first mapping (non-PIE executables) is considered. */
        maps[0].start = 0; maps[0].end = mmap_min_addr(); maps[0].heap = false;
        int n = 1 + read_maps(maps + 1, RP_MAX_MAPS);
        uintptr_t best = 0;
        intptr_t best_d = range;
        /* Gaps between consecutive mappings (sorted by the kernel). */
        for (int i = 0; i + 1 < n; i++) {
            uintptr_t gs = (maps[i].end + page - 1) & ~(uintptr_t)(page - 1);
            uintptr_t ge = maps[i + 1].start;
            if (ge <= gs || ge - gs < RP_BLOCK_SIZE + 2 * page) continue;
            gs += page; ge -= page;                       /* keep guard pages */
            if (maps[i].heap) {                           /* leave room for brk() */
                uintptr_t room = (uintptr_t)64 << 20;
                if (ge - gs <= room + RP_BLOCK_SIZE) continue;
                gs += room;
            }
            uintptr_t cand = near < gs ? gs : (near + RP_BLOCK_SIZE > ge ? ge - RP_BLOCK_SIZE : near);
            cand &= ~(uintptr_t)(page - 1);
            if (cand < gs || cand + RP_BLOCK_SIZE > ge) continue;
            intptr_t d = dist(cand, near);
            intptr_t d2 = dist(cand + RP_BLOCK_SIZE, near);
            if (d2 > d) d = d2;
            if (d < best_d) { best_d = d; best = cand; }
        }
        if (!best) return NULL;
        p = mmap((void*)best, RP_BLOCK_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) {
            rwx = false;
            p = mmap((void*)best, RP_BLOCK_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        }
        if (p != MAP_FAILED && p != (void*)best) { munmap(p, RP_BLOCK_SIZE); return NULL; }
    } else {
        p = mmap(NULL, RP_BLOCK_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            rwx = false;
            p = mmap(NULL, RP_BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
    }
    if (p == MAP_FAILED) return NULL;
    rp_block_t* b = &blocks[nblocks++];
    b->base = (uint8_t*)p;
    b->used = 0;
    b->rwx  = rwx;
    return b;
}

/* Allocate @p size bytes of code memory within @p range of @p near (0:
 * anywhere). @p fresh: at the start of a new (page-aligned) block.
 * Blocks that could not be mapped RWX are used once (sealed afterwards). */
static uint8_t* alloc_code(uintptr_t near, size_t size, intptr_t range, bool fresh, rp_block_t** owner)
{
    size = (size + 15) & ~(size_t)15;
    for (int i = 0; i < nblocks && !fresh; i++) {
        rp_block_t* b = &blocks[i];
        if (!b->rwx || b->used + size > RP_BLOCK_SIZE) continue;
        if (near) {
            uintptr_t a = (uintptr_t)(b->base + b->used);
            if (dist(a, near) >= range || dist(a + size, near) >= range) continue;
        }
        *owner = b;
        uint8_t* p = b->base + b->used;
        b->used += size;
        return p;
    }
    rp_block_t* b = new_block(near, range);
    if (!b) return NULL;
    *owner = b;
    b->used = size;
    return b->base;
}

static void seal_code(rp_block_t* b, uint8_t* p, size_t size)
{
    if (!b->rwx) mprotect(b->base, RP_BLOCK_SIZE, PROT_READ | PROT_EXEC);
    __builtin___clear_cache((char*)p, (char*)p + size);
}

/* Make [addr, addr+len) writable, copy @p bytes, restore R-X. */
static bool write_code(void* addr, const void* bytes, size_t len)
{
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t s = (uintptr_t)addr & ~(page - 1);
    uintptr_t e = ((uintptr_t)addr + len + page - 1) & ~(page - 1);
    bool exec_kept = true;
    if (mprotect((void*)s, e - s, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        exec_kept = false;      /* W^X policy: brief R-W window */
        if (mprotect((void*)s, e - s, PROT_READ | PROT_WRITE) != 0) return false;
    }
#if defined(__aarch64__)
    if (len == 4) __atomic_store_n((uint32_t*)addr, *(const uint32_t*)bytes, __ATOMIC_RELEASE);
    else memcpy(addr, bytes, len);
#else
    memcpy(addr, bytes, len);
#endif
    mprotect((void*)s, e - s, PROT_READ | PROT_EXEC);
    (void)exec_kept;
    __builtin___clear_cache((char*)addr, (char*)addr + len);
    return true;
}

/* Offset of the hook slot from the thread pointer (static TLS: same for all threads). */
static intptr_t slot_tls_offset(void)
{
#if defined(__aarch64__)
    return (intptr_t)((uintptr_t)&ratelprof_hook_slot - (uintptr_t)__builtin_thread_pointer());
#elif defined(__x86_64__)
    uintptr_t fs0;
    __asm__ volatile("mov %%fs:0, %0" : "=r"(fs0));
    return (intptr_t)((uintptr_t)&ratelprof_hook_slot - fs0);
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------------- */
/* aarch64                                                                    */
/* ------------------------------------------------------------------------- */
#if defined(__aarch64__)

#define A64_NOP      0xd503201fu
#define A64_BTI_C    0xd503245fu
#define A64_PACIASP  0xd503233fu
#define A64_AUTIASP  0xd50323bfu
#define A64_PACIBSP  0xd503237fu
#define A64_AUTIBSP  0xd50323ffu

static inline int64_t sext(uint64_t v, int bits) { return (int64_t)(v << (64 - bits)) >> (64 - bits); }

/* Small assembler: code words first, 8-byte literals after. */
typedef struct {
    uint32_t code[64];
    int      ncode;
    uint64_t lit[16];
    int      nlit;
    int      lit_ref[16];   /* index of the LDR (literal) referencing lit[i] */
} a64_asm_t;

static void a64_emit(a64_asm_t* a, uint32_t w) { a->code[a->ncode++] = w; }

/* ldr x<rt>, =value */
static void a64_ldr_lit(a64_asm_t* a, int rt, uint64_t value)
{
    a->lit[a->nlit] = value;
    a->lit_ref[a->nlit] = a->ncode;
    a->nlit++;
    a64_emit(a, 0x58000000u | (uint32_t)rt);       /* imm19 fixed at link time */
}

/* Lay out code + literals at @p dst (whose runtime address is @p pc). */
static size_t a64_link(a64_asm_t* a, uint8_t* dst)
{
    size_t code_bytes = (size_t)a->ncode * 4;
    size_t lit_off = (code_bytes + 7) & ~(size_t)7;
    if (lit_off != code_bytes) a64_emit(a, A64_NOP), code_bytes += 4;
    for (int i = 0; i < a->nlit; i++) {
        int at = a->lit_ref[i];
        int64_t delta = (int64_t)(lit_off + (size_t)i * 8) - (int64_t)at * 4;
        a->code[at] |= (uint32_t)(((delta >> 2) & 0x7ffff) << 5);
    }
    memcpy(dst, a->code, code_bytes);
    memcpy(dst + lit_off, a->lit, (size_t)a->nlit * 8);
    return lit_off + (size_t)a->nlit * 8;
}

static void a64_set_slot_and_jump(a64_asm_t* a, uint64_t impl, uint64_t target, intptr_t tls_off)
{
    a64_emit(a, 0xd53bd050u);               /* mrs  x16, tpidr_el0      */
    a64_ldr_lit(a, 17, (uint64_t)tls_off);  /* ldr  x17, =tls_off       */
    a64_emit(a, 0x8b110210u);               /* add  x16, x16, x17       */
    a64_ldr_lit(a, 17, impl);               /* ldr  x17, =impl          */
    a64_emit(a, 0xf9000211u);               /* str  x17, [x16]          */
    a64_ldr_lit(a, 16, target);             /* ldr  x16, =target        */
    a64_emit(a, 0xd61f0200u);               /* br   x16                 */
}

/* Relocate the instruction @p insn that was at @p pc. Returns false if unsupported. */
static bool a64_relocate(a64_asm_t* a, uint32_t insn, uint64_t pc)
{
    if ((insn & 0x7c000000u) == 0x14000000u) {                     /* B / BL */
        uint64_t t = pc + (uint64_t)(sext(insn & 0x3ffffffu, 26) << 2);
        a64_ldr_lit(a, 17, t);
        a64_emit(a, (insn & 0x80000000u) ? 0xd63f0220u : 0xd61f0220u);   /* blr/br x17 */
        return true;
    }
    if ((insn & 0xff000010u) == 0x54000000u ||                       /* B.cond  */
        (insn & 0x7e000000u) == 0x34000000u ||                       /* CBZ/CBNZ */
        (insn & 0x7e000000u) == 0x36000000u) {                       /* TBZ/TBNZ */
        uint64_t t;
        uint32_t cond;
        if ((insn & 0x7e000000u) == 0x36000000u) {
            t = pc + (uint64_t)(sext((insn >> 5) & 0x3fffu, 14) << 2);
            cond = (insn & ~(0x3fffu << 5)) | (2u << 5);             /* +8 */
        } else {
            t = pc + (uint64_t)(sext((insn >> 5) & 0x7ffffu, 19) << 2);
            cond = (insn & ~(0x7ffffu << 5)) | (2u << 5);            /* +8 */
        }
        a64_emit(a, cond);                    /* taken -> L_taken           */
        a64_emit(a, 0x14000003u);             /* b L_cont (skip 2 insns)    */
        a64_ldr_lit(a, 17, t);                /* L_taken: ldr x17, =target  */
        a64_emit(a, 0xd61f0220u);             /* br x17                     */
        return true;                          /* L_cont: falls through      */
    }
    if ((insn & 0x1f000000u) == 0x10000000u) {                       /* ADR / ADRP */
        int rd = insn & 0x1f;
        uint64_t imm = ((insn >> 3) & 0x1ffffcu) | ((insn >> 29) & 3u);
        uint64_t t = (insn & 0x80000000u)
                   ? (pc & ~0xfffull) + (uint64_t)(sext(imm, 21) << 12)
                   : pc + (uint64_t)sext(imm, 21);
        a64_ldr_lit(a, rd, t);
        return true;
    }
    if ((insn & 0x3b000000u) == 0x18000000u) {                       /* LDR (literal) */
        int rt = insn & 0x1f;
        uint32_t opc = insn >> 30;
        uint64_t addr = pc + (uint64_t)(sext((insn >> 5) & 0x7ffffu, 19) << 2);
        if (insn & 0x04000000u) {                                     /* SIMD&FP */
            static const uint32_t ld[3] = { 0xbd400000u, 0xfd400000u, 0x3dc00000u };
            if (opc > 2) return false;
            a64_ldr_lit(a, 17, addr);
            a64_emit(a, ld[opc] | (17u << 5) | (uint32_t)rt);
        } else {
            if (opc == 3) { a64_emit(a, A64_NOP); return true; }      /* PRFM */
            static const uint32_t ld[3] = { 0xb9400000u, 0xf9400000u, 0xb9800000u };
            a64_ldr_lit(a, rt, addr);
            a64_emit(a, ld[opc] | ((uint32_t)rt << 5) | (uint32_t)rt);
        }
        return true;
    }
    a64_emit(a, insn);                                                /* position independent */
    return true;
}

/* True if a direct branch of the function [fn, fn+size) lands strictly
 * inside (lo, hi): those instructions would be overwritten by a long patch. */
static bool a64_branch_into(const uint32_t* fn, size_t size, uint64_t lo, uint64_t hi)
{
    size_t n = size / 4;
    if (n > 262144) n = 262144;
    for (size_t k = 0; k < n; k++) {
        uint32_t w = fn[k];
        uint64_t pc = (uint64_t)(fn + k), tgt;
        if ((w & 0x7c000000u) == 0x14000000u)
            tgt = pc + (uint64_t)(sext(w & 0x3ffffffu, 26) << 2);
        else if ((w & 0xff000010u) == 0x54000000u || (w & 0x7e000000u) == 0x34000000u)
            tgt = pc + (uint64_t)(sext((w >> 5) & 0x7ffffu, 19) << 2);
        else if ((w & 0x7e000000u) == 0x36000000u)
            tgt = pc + (uint64_t)(sext((w >> 5) & 0x3fffu, 14) << 2);
        else
            continue;
        if (tgt > lo && tgt < hi) return true;
    }
    return false;
}

void* ratelprof_hook_make_stub(void* impl, void* wrapper)
{
    a64_asm_t a = {0};
    a64_set_slot_and_jump(&a, (uint64_t)impl, (uint64_t)wrapper, slot_tls_offset());
    rp_block_t* b = NULL;
    pthread_mutex_lock(&hook_lock);
    uint8_t* p = alloc_code(0, 128, RP_HOOK_RANGE, false, &b);
    if (p) { a64_link(&a, p); seal_code(b, p, 128); }
    pthread_mutex_unlock(&hook_lock);
    return p;
}

static void* hook_impl(void* target, size_t size, void* wrapper, const char** why)
{
    uint32_t* t = (uint32_t*)target;
    if (((uintptr_t)t & 3) != 0) { *why = "misaligned function"; return NULL; }
    uint32_t first = t[0];
    int patch = 0;                     /* index of the instruction replaced by B */
    uint32_t auth = 0;                 /* AUTI*SP to undo a pointer-auth prologue */
    if (first == A64_BTI_C || first == 0xd503241fu || first == 0xd503249fu || first == 0xd50324dfu) {
        patch = 1;                     /* keep the BTI landing pad */
    } else if (first == A64_PACIASP || first == A64_PACIBSP) {
        patch = 1;                     /* keep it (it is a landing pad), undo it in the island */
        auth = (first == A64_PACIASP) ? A64_AUTIASP : A64_AUTIBSP;
    }
    if (size && (size_t)(patch + 1) * 4 > size) { *why = "function too small"; return NULL; }

    uint64_t pc = (uint64_t)(t + patch);
    rp_block_t* b = NULL;

    /* Patch kinds, by decreasing preference:
     *   short  : b island                         (4 bytes, island within +-128 MiB)
     *   medium : adrp x16, island; add; br x16    (12 bytes, island within +-4 GiB)
     *   page   : adrp x16, island; br x16         (8 bytes, page-aligned island, +-4 GiB)
     *   long   : ldr x16, #8; br x16; .quad island (16 bytes, anywhere)
     * Every instruction overwritten is relocated into the trampoline; no branch
     * of the function may land on the overwritten instructions (but the first). */
    static int force = -1;      /* RATELPROF_HOOK_FORCE_LONG: 1 skip short, 2 long only */
    if (force < 0) force = getenv("RATELPROF_HOOK_FORCE_LONG") ? atoi(getenv("RATELPROF_HOOK_FORCE_LONG")) : 0;
    const intptr_t range4g = (intptr_t)4000 << 20;
    size_t avail = size ? size / 4 - (size_t)patch : 0;     /* instructions available */
    enum { P_SHORT, P_MEDIUM, P_PAGE, P_LONG } kind = P_SHORT;
    uint8_t* mem = NULL;
    uint8_t* tramp = NULL;
    uint8_t* island = NULL;
    if (force == 0) mem = alloc_code((uintptr_t)target, 320, RP_HOOK_RANGE, false, &b);
    if (!mem && force < 2 && avail >= 3 && !a64_branch_into(t, size, pc, pc + 12)) {
        mem = alloc_code((uintptr_t)target, 320, range4g, false, &b);
        if (mem) kind = P_MEDIUM;
    }
    if (!mem && force < 2 && avail >= 2 && !a64_branch_into(t, size, pc, pc + 8)) {
        mem = alloc_code((uintptr_t)target, 320, range4g, true, &b);   /* island at the page start */
        if (mem) kind = P_PAGE;
    }
    if (!mem) {
        if (avail < 4) { *why = "function too small for the jump reachable from it"; return NULL; }
        if (a64_branch_into(t, size, pc, pc + 16)) { *why = "a branch targets the patched prologue"; return NULL; }
        mem = alloc_code(0, 320, RP_HOOK_RANGE, false, &b);
        if (!mem) { *why = "cannot allocate executable memory"; return NULL; }
        kind = P_LONG;
    }
    if (kind == P_PAGE) { island = mem; tramp = mem + 128; }
    else                { tramp = mem;  island = mem + 192; }
    const int nrel = kind == P_SHORT ? 1 : kind == P_MEDIUM ? 3 : kind == P_PAGE ? 2 : 4;

    /* Trampoline: [bti c] [prologue landing pad] relocated insns ; jump back */
    a64_asm_t a = {0};
    a64_emit(&a, A64_BTI_C);
    if (patch == 1 && auth) a64_emit(&a, first);          /* re-sign LR */
    for (int k = 0; k < nrel; k++)
        if (!a64_relocate(&a, t[patch + k], pc + 4u * (uint64_t)k)) { *why = "unsupported instruction"; return NULL; }
    a64_ldr_lit(&a, 17, pc + 4u * (uint64_t)nrel);
    a64_emit(&a, 0xd61f0220u);                            /* br x17 */
    if (a64_link(&a, tramp) > (kind == P_PAGE ? 192 : 192)) { *why = "trampoline too large"; return NULL; }

    /* Island: [undo PAC] slot <- trampoline ; br wrapper */
    a64_asm_t i = {0};
    if (auth) a64_emit(&i, auth);
    a64_set_slot_and_jump(&i, (uint64_t)tramp, (uint64_t)wrapper, slot_tls_offset());
    if (a64_link(&i, island) > 128) { *why = "island too large"; return NULL; }
    seal_code(b, mem, 320);

    uint32_t seq[4];
    size_t len;
    int64_t page_delta = (int64_t)(((uint64_t)island & ~0xfffull) - (pc & ~0xfffull)) >> 12;
    uint32_t adrp = 0x90000000u | 16u | (uint32_t)((page_delta & 3) << 29) | (uint32_t)(((page_delta >> 2) & 0x7ffff) << 5);
    if (kind == P_SHORT) {
        int64_t off = (int64_t)((uint64_t)island - pc);
        if (off >= ((int64_t)1 << 27) || off < -((int64_t)1 << 27)) { *why = "island out of range"; return NULL; }
        seq[0] = 0x14000000u | (uint32_t)((off >> 2) & 0x3ffffffu);
        len = 4;
    } else if (kind == P_MEDIUM || kind == P_PAGE) {
        if (page_delta >= ((int64_t)1 << 20) || page_delta < -((int64_t)1 << 20)) { *why = "island out of ADRP range"; return NULL; }
        seq[0] = adrp;
        if (kind == P_MEDIUM) {
            seq[1] = 0x91000210u | (uint32_t)(((uint64_t)island & 0xfff) << 10);   /* add x16, x16, #lo12 */
            seq[2] = 0xd61f0200u;                                                  /* br x16 */
            len = 12;
        } else {
            if (((uint64_t)island & 0xfff) != 0) { *why = "island not page aligned"; return NULL; }
            seq[1] = 0xd61f0200u;
            len = 8;
        }
    } else {
        seq[0] = 0x58000050u;            /* ldr x16, #8 */
        seq[1] = 0xd61f0200u;            /* br x16 */
        uint64_t isl = (uint64_t)island;
        memcpy(&seq[2], &isl, 8);
        len = 16;
    }
    if (!write_code((void*)pc, seq, len)) { *why = "cannot write code (mprotect)"; return NULL; }
    return tramp;
}

/* ------------------------------------------------------------------------- */
/* x86_64                                                                     */
/* ------------------------------------------------------------------------- */
#elif defined(__x86_64__)

enum { X86_PLAIN, X86_JMP, X86_CALL, X86_JCC, X86_BAD };

typedef struct {
    int      len;
    int      kind;       /* X86_* */
    int      rip_off;    /* offset of a RIP-relative disp32, -1 if none */
    int64_t  rel;        /* branch displacement (JMP/CALL/JCC) */
    uint8_t  cc;         /* condition code for JCC */
} x86_insn_t;

/* Decode the length of the instruction at @p p (64-bit mode). */
static bool x86_decode(const uint8_t* p, x86_insn_t* d)
{
    const uint8_t* s = p;
    bool opsz = false, adsz = false, rexw = false;
    d->kind = X86_PLAIN; d->rip_off = -1; d->rel = 0; d->cc = 0;

    for (;;) {                                  /* legacy prefixes */
        uint8_t b = *p;
        if (b == 0x66) opsz = true;
        else if (b == 0x67) adsz = true;
        else if (b == 0xf0 || b == 0xf2 || b == 0xf3 || b == 0x2e || b == 0x36 ||
                 b == 0x3e || b == 0x26 || b == 0x64 || b == 0x65) {}
        else break;
        p++;
    }
    if ((*p & 0xf0) == 0x40) { rexw = (*p & 8) != 0; p++; }

    int map = 0;                                /* 0: one byte, 1: 0F, 2: 0F38, 3: 0F3A */
    bool modrm = false;
    int imm = 0;
    uint8_t op;

    if (*p == 0xc5 || *p == 0xc4 || *p == 0x62) {           /* VEX / EVEX */
        if (*p == 0xc5) { map = 1; p += 2; }
        else if (*p == 0xc4) { map = p[1] & 0x1f; p += 3; }
        else { map = p[1] & 0x03; p += 4; }
        if (map < 1 || map > 3) return false;
        op = *p++;
        modrm = true;
        if (map == 3) imm = 1;
        else if (map == 1 && ((op >= 0x70 && op <= 0x73) || op == 0xc2 || op == 0xc4 || op == 0xc5 || op == 0xc6)) imm = 1;
    } else {
        op = *p++;
        if (op == 0x0f) {
            op = *p++;
            if (op == 0x38) { map = 2; op = *p++; modrm = true; }
            else if (op == 0x3a) { map = 3; op = *p++; modrm = true; imm = 1; }
            else {
                map = 1;
                if (op >= 0x80 && op <= 0x8f) {             /* Jcc rel32 */
                    d->kind = X86_JCC; d->cc = op & 0x0f;
                    d->rel = *(const int32_t*)p; p += 4;
                    d->len = (int)(p - s);
                    return true;
                }
                switch (op) {
                    case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: case 0x0b: case 0x0e:
                    case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: case 0x37:
                    case 0x77: case 0xa0: case 0xa1: case 0xa2: case 0xa8: case 0xa9: case 0xaa:
                    case 0xc8: case 0xc9: case 0xca: case 0xcb: case 0xcc: case 0xcd: case 0xce: case 0xcf:
                        modrm = false; break;
                    case 0x0f: return false;                     /* 3DNow! */
                    case 0x70: case 0x71: case 0x72: case 0x73: case 0xa4: case 0xac:
                    case 0xba: case 0xc2: case 0xc4: case 0xc5: case 0xc6:
                        modrm = true; imm = 1; break;
                    default: modrm = true; break;
                }
            }
        } else {
            int iz = opsz ? 2 : 4;                         /* imm16/32 */
            switch (op) {
                case 0x00: case 0x01: case 0x02: case 0x03: case 0x08: case 0x09: case 0x0a: case 0x0b:
                case 0x10: case 0x11: case 0x12: case 0x13: case 0x18: case 0x19: case 0x1a: case 0x1b:
                case 0x20: case 0x21: case 0x22: case 0x23: case 0x28: case 0x29: case 0x2a: case 0x2b:
                case 0x30: case 0x31: case 0x32: case 0x33: case 0x38: case 0x39: case 0x3a: case 0x3b:
                case 0x63: case 0x84: case 0x85: case 0x86: case 0x87: case 0x88: case 0x89: case 0x8a:
                case 0x8b: case 0x8c: case 0x8d: case 0x8e: case 0x8f: case 0xd0: case 0xd1: case 0xd2:
                case 0xd3: case 0xd8: case 0xd9: case 0xda: case 0xdb: case 0xdc: case 0xdd: case 0xde:
                case 0xdf: case 0xfe: case 0xff:
                    modrm = true; break;
                case 0x04: case 0x0c: case 0x14: case 0x1c: case 0x24: case 0x2c: case 0x34: case 0x3c:
                case 0x6a: case 0xa8: case 0xcd: case 0xe4: case 0xe5: case 0xe6: case 0xe7:
                    imm = 1; break;
                case 0x05: case 0x0d: case 0x15: case 0x1d: case 0x25: case 0x2d: case 0x35: case 0x3d:
                case 0x68: case 0xa9:
                    imm = iz; break;
                case 0x69: case 0x81: case 0xc7: modrm = true; imm = iz; break;
                case 0x6b: case 0x80: case 0x83: case 0xc0: case 0xc1: case 0xc6: modrm = true; imm = 1; break;
                case 0xc8: imm = 3; break;                                      /* enter */
                case 0xa0: case 0xa1: case 0xa2: case 0xa3: imm = adsz ? 4 : 8; break;
                case 0xf6: modrm = true; imm = ((p[0] >> 3) & 7) < 2 ? 1 : 0; break;
                case 0xf7: modrm = true; imm = ((p[0] >> 3) & 7) < 2 ? iz : 0; break;
                case 0xe8: d->kind = X86_CALL; d->rel = *(const int32_t*)p; p += 4; d->len = (int)(p - s); return true;
                case 0xe9: d->kind = X86_JMP;  d->rel = *(const int32_t*)p; p += 4; d->len = (int)(p - s); return true;
                case 0xeb: d->kind = X86_JMP;  d->rel = *(const int8_t*)p;  p += 1; d->len = (int)(p - s); return true;
                case 0xc2: case 0xc3: case 0xca: case 0xcb: case 0xcc: case 0xcf:  /* returns, int3 */
                case 0xe0: case 0xe1: case 0xe2: case 0xe3:                        /* loop/jrcxz */
                case 0x06: case 0x07: case 0x0e: case 0x16: case 0x17: case 0x1e: case 0x1f:
                case 0x27: case 0x2f: case 0x37: case 0x3f: case 0x60: case 0x61: case 0x82:
                case 0x9a: case 0xc4: case 0xc5: case 0xce: case 0xd4: case 0xd5: case 0xd6: case 0xea:
                    return false;
                default:
                    if (op >= 0x70 && op <= 0x7f) {                                 /* Jcc rel8 */
                        d->kind = X86_JCC; d->cc = op & 0x0f;
                        d->rel = *(const int8_t*)p; p += 1;
                        d->len = (int)(p - s);
                        return true;
                    }
                    if (op >= 0xb0 && op <= 0xb7) imm = 1;
                    else if (op >= 0xb8 && op <= 0xbf) imm = rexw ? 8 : iz;
                    /* 50-5f, 6c-6f, 90-9f, a4-a7, aa-af, c9, d7, ec-ef, f1, f4, f5, f8-fd: no operand */
                    break;
            }
        }
    }
    (void)map;
    if (modrm) {
        uint8_t m = *p++;
        int mod = m >> 6, rm = m & 7;
        if (mod != 3) {
            if (rm == 4) {                                   /* SIB */
                uint8_t sib = *p++;
                if (mod == 0 && (sib & 7) == 5) p += 4;
            } else if (mod == 0 && rm == 5) {                 /* RIP-relative */
                d->rip_off = (int)(p - s);
                p += 4;
            }
            if (mod == 1) p += 1;
            else if (mod == 2) p += 4;
        }
    }
    p += imm;
    d->len = (int)(p - s);
    return d->len <= 15;
}

/* Alignment padding: nop, int3, multi-byte nops (66.. 2e 0f 1f /0, 66 90). */
static bool x86_padding(const uint8_t* p, x86_insn_t* d)
{
    d->kind = X86_PLAIN; d->rip_off = -1; d->rel = 0;
    if (p[0] == 0x90 || p[0] == 0xcc) { d->len = 1; return true; }
    const uint8_t* q = p;
    while ((*q == 0x66 || *q == 0x2e) && q - p < 8) q++;
    if (!(q[0] == 0x90 || (q[0] == 0x0f && q[1] == 0x1f))) return false;
    x86_insn_t tmp;
    if (!x86_decode(p, &tmp)) return false;
    d->len = tmp.len;
    return true;
}

/* Linear sweep of a function: 1 if a direct branch lands strictly inside
 * (lo, hi), 0 if none does, -1 if the function could not be decoded. */
static int x86_branch_into(const uint8_t* fn, size_t size, uint64_t lo, uint64_t hi)
{
    size_t off = 0;
    while (off < size) {
        const uint8_t* p = fn + off;
        if (p[0] == 0xc3 || p[0] == 0xcc || p[0] == 0x90) { off++; continue; }
        if (p[0] == 0xc2) { off += 3; continue; }
        x86_insn_t d;
        if (!x86_decode(p, &d)) return -1;
        if (d.kind == X86_JMP || d.kind == X86_CALL || d.kind == X86_JCC) {
            uint64_t tgt = (uint64_t)(p + d.len) + (uint64_t)d.rel;
            if (tgt > lo && tgt < hi) return 1;
        }
        off += (size_t)d.len;
    }
    return 0;
}

typedef struct { uint8_t b[640]; int n; } x86_buf_t;   /* >= 16 insns x 16 bytes + jumps */
static void xb(x86_buf_t* o, const void* src, int n) { memcpy(o->b + o->n, src, (size_t)n); o->n += n; }
static void xb8(x86_buf_t* o, uint8_t v) { o->b[o->n++] = v; }
static void x86_abs_jmp(x86_buf_t* o, uint64_t target)
{
    static const uint8_t jmp[6] = { 0xff, 0x25, 0, 0, 0, 0 };     /* jmp [rip+0] */
    xb(o, jmp, 6); xb(o, &target, 8);
}

void* ratelprof_hook_make_stub(void* impl, void* wrapper)
{
    x86_buf_t o = {0};
    int64_t tls = slot_tls_offset();
    uint64_t im = (uint64_t)impl, wr = (uint64_t)wrapper;
    /* mov r11,[rip+off] ; mov r10,[rip+impl] ; mov fs:[r11],r10 ; jmp [rip+wrapper] */
    static const uint8_t code[] = {
        0x4c, 0x8b, 0x1d, 0x11, 0, 0, 0,          /* mov r11, [rip+17] -> lit0 (24) */
        0x4c, 0x8b, 0x15, 0x12, 0, 0, 0,          /* mov r10, [rip+18] -> lit1 (32) */
        0x64, 0x4d, 0x89, 0x13,                   /* mov fs:[r11], r10              */
        0xff, 0x25, 0x10, 0, 0, 0,                /* jmp [rip+16] -> lit2 (40)      */
    };
    xb(&o, code, sizeof(code));
    xb(&o, &tls, 8); xb(&o, &im, 8); xb(&o, &wr, 8);
    rp_block_t* b = NULL;
    pthread_mutex_lock(&hook_lock);
    uint8_t* p = alloc_code(0, 64, RP_HOOK_RANGE, false, &b);
    if (p) { memcpy(p, o.b, (size_t)o.n); seal_code(b, p, 64); }
    pthread_mutex_unlock(&hook_lock);
    return p;
}

static void* hook_impl(void* target, size_t size, void* wrapper, const char** why)
{
    uint8_t* t = (uint8_t*)target;
    int start = 0;
    if (t[0] == 0xf3 && t[1] == 0x0f && t[2] == 0x1e && t[3] == 0xfa) start = 4;   /* keep endbr64 */

    /* Short patch: jmp rel32 (5 bytes) to an island within +-2 GiB. Otherwise
     * long patch: jmp [rip+0]; .quad island (14 bytes). */
    rp_block_t* b = NULL;
    static int force_long = -1;
    if (force_long < 0) force_long = getenv("RATELPROF_HOOK_FORCE_LONG") && atoi(getenv("RATELPROF_HOOK_FORCE_LONG")) != 0;
    uint8_t* mem = force_long ? NULL : alloc_code((uintptr_t)target, 256, RP_HOOK_RANGE, false, &b);
    const int need = mem ? 5 : 14;

    /* Instructions covering the patch. With a known size, a short function may
     * end (ret) inside the patch if only alignment padding follows it, up to
     * the next 16-byte boundary (where the next function may start). */
    const uintptr_t fend  = (uintptr_t)t + size;
    const uintptr_t limit = size ? ((fend + 15) & ~(uintptr_t)15) : 0;
    x86_insn_t ins[16];
    int nins = 0, covered = 0;
    bool ended = false;                       /* past a ret / unconditional jmp */
    while (covered < need) {
        const uint8_t* at = t + start + covered;
        x86_insn_t* d = &ins[nins];
        if (nins == 16) { *why = "unsupported instruction in prologue"; return NULL; }
        if (size && (uintptr_t)at >= fend) {                        /* padding */
            if (!x86_padding(at, d) || (uintptr_t)at + (uintptr_t)d->len > limit) { *why = "function too small"; return NULL; }
        } else if (ended) {
            *why = "function too small"; return NULL;
        } else if (size && (at[0] == 0xc3 || at[0] == 0xcc)) {      /* ret, int3 */
            d->len = 1; d->kind = X86_PLAIN; d->rip_off = -1;
            ended = true;
        } else if (!x86_decode(at, d) || d->kind == X86_BAD) {
            *why = "unsupported instruction in prologue"; return NULL;
        } else if (d->kind == X86_JMP) {
            ended = true;
        }
        if (ended && !size && covered + d->len < need) { *why = "function too small"; return NULL; }
        covered += d->len;
        nins++;
    }
    if (size && (uintptr_t)(t + start + covered) > (ended ? limit : fend)) { *why = "function too small"; return NULL; }

    /* No branch of the function may land inside the overwritten bytes. */
    if (size) {
        int r = x86_branch_into(t, size, (uint64_t)(t + start), (uint64_t)(t + start + covered));
        if (r > 0) { *why = "a branch targets the patched prologue"; return NULL; }
        if (r < 0 && need > 5) { *why = "cannot verify the branch targets of the function"; return NULL; }
    }
    if (!mem) {
        if (!size) { *why = "no memory within +-2GiB and unknown function size"; return NULL; }
        mem = alloc_code(0, 256, RP_HOOK_RANGE, false, &b);
        if (!mem) { *why = "cannot allocate executable memory"; return NULL; }
    }
    uint8_t* tramp  = mem;
    uint8_t* island = mem + 192;

    x86_buf_t o = {0};
    static const uint8_t endbr[4] = { 0xf3, 0x0f, 0x1e, 0xfa };
    xb(&o, endbr, 4);
    int off = start;
    for (int k = 0; k < nins; k++) {
        const x86_insn_t* d = &ins[k];
        uint64_t pc = (uint64_t)(t + off);
        uint64_t next = pc + (uint64_t)d->len;
        if (d->kind == X86_JMP) {
            x86_abs_jmp(&o, next + (uint64_t)d->rel);
        } else if (d->kind == X86_CALL) {
            static const uint8_t call[8] = { 0xff, 0x15, 0x02, 0, 0, 0, 0xeb, 0x08 };  /* call [rip+2]; jmp +8 */
            uint64_t tgt = next + (uint64_t)d->rel;
            xb(&o, call, 8); xb(&o, &tgt, 8);
        } else if (d->kind == X86_JCC) {
            xb8(&o, (uint8_t)(0x70 | (d->cc ^ 1))); xb8(&o, 14);  /* j!cc over the abs jmp */
            x86_abs_jmp(&o, next + (uint64_t)d->rel);
        } else {
            int at = o.n;
            xb(&o, t + off, d->len);
            if (d->rip_off >= 0) {
                int64_t old = *(const int32_t*)(t + off + d->rip_off);
                int64_t nd = old + (int64_t)(pc - (uint64_t)(tramp + at));
                if (nd != (int32_t)nd) { *why = "RIP-relative operand out of range"; return NULL; }
                int32_t nd32 = (int32_t)nd;
                memcpy(o.b + at + d->rip_off, &nd32, 4);
            }
        }
        off += d->len;
    }
    x86_abs_jmp(&o, (uint64_t)(t + off));
    if (o.n > 192) { *why = "trampoline too large"; return NULL; }
    memcpy(tramp, o.b, (size_t)o.n);

    /* Island */
    int64_t tls = slot_tls_offset();
    uint64_t im = (uint64_t)tramp, wr = (uint64_t)wrapper;
    static const uint8_t code[] = {
        0x4c, 0x8b, 0x1d, 0x11, 0, 0, 0,
        0x4c, 0x8b, 0x15, 0x12, 0, 0, 0,
        0x64, 0x4d, 0x89, 0x13,
        0xff, 0x25, 0x10, 0, 0, 0,
    };
    memcpy(island, code, sizeof(code));
    memcpy(island + sizeof(code), &tls, 8);
    memcpy(island + sizeof(code) + 8, &im, 8);
    memcpy(island + sizeof(code) + 16, &wr, 8);
    seal_code(b, mem, 256);

    /* Patch: jump to the island, int3 over the rest of the relocated bytes. */
    uint8_t patch[48];
    int k0;
    if (need == 5) {
        int64_t rel = (int64_t)((uint64_t)island - (uint64_t)(t + start + 5));
        if (rel != (int32_t)rel) { *why = "island out of range"; return NULL; }
        int32_t rel32 = (int32_t)rel;
        patch[0] = 0xe9; memcpy(patch + 1, &rel32, 4);
        k0 = 5;
    } else {
        static const uint8_t jmp[6] = { 0xff, 0x25, 0, 0, 0, 0 };
        uint64_t isl = (uint64_t)island;
        memcpy(patch, jmp, 6); memcpy(patch + 6, &isl, 8);
        k0 = 14;
    }
    for (int k = k0; k < covered; k++) patch[k] = 0xcc;
    if (!write_code(t + start, patch, (size_t)covered)) { *why = "cannot write code (mprotect)"; return NULL; }
    return tramp;
}

#else
void* ratelprof_hook_make_stub(void* impl, void* wrapper) { (void)impl; (void)wrapper; return NULL; }
static void* hook_impl(void* target, size_t size, void* wrapper, const char** why)
{
    (void)target; (void)size; (void)wrapper;
    *why = "inline hooking is not supported on this architecture";
    return NULL;
}
#endif

void* ratelprof_hook_function(void* target, size_t size, void* wrapper, const char** why)
{
    const char* dummy;
    if (!why) why = &dummy;
    *why = NULL;
    if (!target || !wrapper) { *why = "null pointer"; return NULL; }
    pthread_mutex_lock(&hook_lock);
    void* tramp = hook_impl(target, size, wrapper, why);
    if (tramp) hook_count++;
    pthread_mutex_unlock(&hook_lock);
    return tramp;
}
