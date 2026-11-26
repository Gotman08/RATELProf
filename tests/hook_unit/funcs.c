/* Functions with varied prologues, compiled with different flags. */
#include <math.h>
#include <string.h>

int  g_counter = 3;
static double g_scale = 2.5;

__attribute__((noipa)) int f_mul(int a, int b) { return a * b + 1; }

/* PC/RIP-relative data access in the first instructions */
__attribute__((noipa)) int f_global(int a) { g_counter += a; return g_counter * 2; }

/* 10 integer arguments: some are passed on the stack */
__attribute__((noipa)) long f_many(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j)
{
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i + 10 * j;
}

/* floating point arguments and result */
__attribute__((noipa)) double f_float(double x, float y) { return x * g_scale + sqrt((double)y); }

/* early conditional branch and a loop */
__attribute__((noipa)) int f_loop(int n)
{
    if (n <= 0) return -1;
    int s = 0;
    for (int k = 0; k < n; k++) s += k * k;
    return s;
}

/* calls another function right away */
__attribute__((noipa)) int f_callee(int x) { return x + 7; }
__attribute__((noipa)) int f_call(int x) { return f_callee(x) * 3; }

/* tail call */
__attribute__((noipa)) int f_tail(int x) { return f_callee(x + 1); }

/* struct by value + pointer output */
typedef struct { long a, b; } pair_t;
__attribute__((noipa)) int f_struct(pair_t p, long* out) { *out = p.a - p.b; return (int)(p.a + p.b); }

/* recursion through the hooked entry */
__attribute__((noipa)) long f_fact(long n) { long (* volatile self)(long) = f_fact; return n <= 1 ? 1 : n * self(n - 1); }

/* string handling with a local buffer (stack protector, big frame) */
__attribute__((noipa)) size_t f_buf(const char* s)
{
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    return strlen(buf) * 10;
}
