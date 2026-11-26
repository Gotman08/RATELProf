/* Unit test of the RATELProf inline hooking engine (ratelprof_hook.c). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "ratelprof/ratelprof_hook.h"

typedef struct { long a, b; } pair_t;
extern int g_counter;
int f_mul(int, int);
int f_global(int);
long f_many(long, long, long, long, long, long, long, long, long, long);
double f_float(double, float);
int f_loop(int);
int f_callee(int);
int f_call(int);
int f_tail(int);
int f_struct(pair_t, long*);
long f_fact(long);
size_t f_buf(const char*);

static int n_mul, n_global, n_many, n_float, n_loop, n_callee, n_call, n_tail, n_struct, n_fact, n_buf, n_stub;

#define IMPL(type) ((type)ratelprof_hook_take())
int    w_mul(int a, int b) { n_mul++; return IMPL(int (*)(int, int))(a, b); }
int    w_global(int a) { n_global++; return IMPL(int (*)(int))(a); }
long   w_many(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j)
{ n_many++; return IMPL(long (*)(long, long, long, long, long, long, long, long, long, long))(a, b, c, d, e, f, g, h, i, j); }
double w_float(double x, float y) { n_float++; return IMPL(double (*)(double, float))(x, y); }
int    w_loop(int n) { n_loop++; return IMPL(int (*)(int))(n); }
int    w_callee(int x) { n_callee++; return IMPL(int (*)(int))(x); }
int    w_call(int x) { n_call++; return IMPL(int (*)(int))(x); }
int    w_tail(int x) { n_tail++; return IMPL(int (*)(int))(x); }
int    w_struct(pair_t p, long* o) { n_struct++; return IMPL(int (*)(pair_t, long*))(p, o); }
long   w_fact(long n) { n_fact++; return IMPL(long (*)(long))(n); }
size_t w_buf(const char* s) { n_buf++; return IMPL(size_t (*)(const char*))(s); }
int    w_stub(int a, int b) { n_stub++; return IMPL(int (*)(int, int))(a, b); }

/* Size of a function from the executable's symbol table. */
static size_t fsize(const char* name)
{
    int fd = open("/proc/self/exe", O_RDONLY);
    struct stat sb; fstat(fd, &sb);
    const unsigned char* img = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)img;
    const Elf64_Shdr* sh = (const Elf64_Shdr*)(img + eh->e_shoff);
    size_t r = 0;
    for (int i = 0; i < eh->e_shnum && !r; i++) {
        if (sh[i].sh_type != SHT_SYMTAB) continue;
        const Elf64_Sym* sy = (const Elf64_Sym*)(img + sh[i].sh_offset);
        const char* str = (const char*)(img + sh[sh[i].sh_link].sh_offset);
        for (size_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym); k++)
            if (strcmp(str + sy[k].st_name, name) == 0) { r = sy[k].st_size; break; }
    }
    munmap((void*)img, sb.st_size);
    return r;
}

static int fails = 0, refused = 0;
#define CHECK(c, ...) do { if (getenv("HOOK_TRACE")) printf("[check line %d]\n", __LINE__); \
    if (!(c)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* A refusal is acceptable only when patching is constrained on purpose
 * (RATELPROF_HOOK_FORCE_LONG): the function must then keep working unhooked. */
static int hook(const char* name, void* f, void* w)
{
    if (getenv("HOOK_TRACE")) printf("[hook %s size %zu]\n", name, fsize(name));
    const char* why = NULL;
    void* t = ratelprof_hook_function(f, fsize(name), w, &why);
    if (!t) {
        const char* force = getenv("RATELPROF_HOOK_FORCE_LONG");
        bool expected = force && atoi(force) != 0;
        printf("  %s %s: %s\n", expected ? "refused (forced mode)" : "cannot hook", name, why);
        if (expected) refused++; else fails++;
        return 0;
    }
    return 1;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    /* Volatile pointers: the compiler cannot fold or inline the calls. */
    int (* volatile p_mul)(int, int) = f_mul;
    int (* volatile p_global)(int) = f_global;
    long (* volatile p_many)(long, long, long, long, long, long, long, long, long, long) = f_many;
    double (* volatile p_float)(double, float) = f_float;
    int (* volatile p_loop)(int) = f_loop;
    int (* volatile p_call)(int) = f_call;
    int (* volatile p_tail)(int) = f_tail;
    int (* volatile p_struct)(pair_t, long*) = f_struct;
    long (* volatile p_fact)(long) = f_fact;
    size_t (* volatile p_buf)(const char*) = f_buf;

    /* Reference results */
    pair_t pr = { 40, 2 }; long o1 = 0, o2 = 0;
    int r_mul = p_mul(6, 7), r_g = p_global(5);
    long r_many = p_many(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    double r_float = p_float(3.0, 16.f);
    int r_loop = p_loop(100), r_loop0 = p_loop(0), r_call = p_call(5), r_tail = p_tail(5), r_struct = p_struct(pr, &o1);
    long r_fact = p_fact(10);
    size_t r_buf = p_buf("ratelprof");
    g_counter = 3;

    int h_mul = hook("f_mul", (void*)f_mul, (void*)w_mul);
    int h_global = hook("f_global", (void*)f_global, (void*)w_global);
    int h_many = hook("f_many", (void*)f_many, (void*)w_many);
    int h_float = hook("f_float", (void*)f_float, (void*)w_float);
    int h_loop = hook("f_loop", (void*)f_loop, (void*)w_loop);
    int h_callee = hook("f_callee", (void*)f_callee, (void*)w_callee);
    int h_call = hook("f_call", (void*)f_call, (void*)w_call);
    int h_tail = hook("f_tail", (void*)f_tail, (void*)w_tail);
    int h_struct = hook("f_struct", (void*)f_struct, (void*)w_struct);
    int h_fact = hook("f_fact", (void*)f_fact, (void*)w_fact);
    int h_buf = hook("f_buf", (void*)f_buf, (void*)w_buf);

    /* Results must be unchanged; wrappers entered exactly once per call if hooked. */
    CHECK(p_mul(6, 7) == r_mul && n_mul == h_mul, "f_mul %d/%d n=%d", p_mul(6, 7), r_mul, n_mul);
    CHECK(p_global(5) == r_g && n_global == h_global, "f_global n=%d", n_global);
    CHECK(p_many(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == r_many && n_many == h_many, "f_many n=%d", n_many);
    CHECK(fabs(p_float(3.0, 16.f) - r_float) < 1e-12 && n_float == h_float, "f_float n=%d", n_float);
    CHECK(p_loop(100) == r_loop && p_loop(0) == r_loop0 && n_loop == 2 * h_loop, "f_loop n=%d", n_loop);
    n_callee = 0;
    CHECK(p_call(5) == r_call && n_call == h_call && n_callee == h_callee, "f_call n=%d callee=%d", n_call, n_callee);
    n_callee = 0;
    CHECK(p_tail(5) == r_tail && n_tail == h_tail && n_callee == h_callee, "f_tail n=%d callee=%d", n_tail, n_callee);
    CHECK(p_struct(pr, &o2) == r_struct && o2 == o1 && n_struct == h_struct, "f_struct n=%d", n_struct);
    CHECK(p_fact(10) == r_fact && n_fact == 10 * h_fact, "f_fact n=%ld (expected 10 recursive entries)", (long)n_fact);
    CHECK(p_buf("ratelprof") == r_buf && n_buf == h_buf, "f_buf n=%d", n_buf);

    /* dlsym-style stub: slot <- f_mul, then wrapper */
    void* stub = ratelprof_hook_make_stub((void*)f_mul, (void*)w_stub);
    int (*ps)(int, int) = (int (*)(int, int))stub;
    n_mul = 0;
    CHECK(stub && ps(3, 4) == 13 && n_stub == 1 && n_mul == h_mul, "stub n_stub=%d n_mul=%d", n_stub, n_mul);

    printf("%s: %zu functions hooked, %d refused in forced mode, %d failure(s)\n",
           fails ? "FAILED" : "PASSED", ratelprof_hook_count(), refused, fails);
    return fails != 0;
}
