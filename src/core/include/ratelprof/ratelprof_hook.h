/**
 * @file ratelprof_hook.h
 * @brief Inline function hooking, for functions LD_PRELOAD cannot interpose.
 *
 * LD_PRELOAD only intercepts calls that go through the dynamic linker (PLT,
 * dlsym). Functions linked statically (libcublas_static.a, libnccl_static.a,
 * the plugins of libomptarget >= 19...) are called directly, and plugins
 * resolved before RATELProf's constructor keep raw pointers to them. Those
 * functions are patched in memory instead:
 *
 *   target:  first instruction replaced by a branch to a jump island
 *   island:  slot <- trampoline ; jump to the RATELProf wrapper
 *   wrapper: the usual interposition wrapper (same prototype as target)
 *   trampoline: relocated first instruction(s) of target ; jump back
 *
 * The wrapper then calls the implementation given by ratelprof_hook_take(),
 * i.e. the trampoline of the definition that was actually called, so several
 * definitions of the same function (a static copy and a shared library) each
 * keep their own implementation.
 *
 * Supported architectures: aarch64 and x86_64.
 */
#ifndef RATELPROF_HOOK_H
#define RATELPROF_HOOK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/** Implementation to call for the wrapper being entered, written by islands. */
extern __thread void* ratelprof_hook_slot __attribute__((tls_model("initial-exec")));

/** Read and clear the implementation stored by the island (NULL if none). */
static inline void* ratelprof_hook_take(void)
{
    void* impl = ratelprof_hook_slot;
    if (impl) ratelprof_hook_slot = NULL;
    return impl;
}

/** True when inline hooking is available for this architecture. */
bool ratelprof_hook_supported(void);

/**
 * Build a stub storing @p impl in the hook slot then jumping to @p wrapper.
 * Used to hand out wrappers for symbols resolved with dlsym(handle).
 * Stubs are cached by @p impl. Returns NULL on failure.
 */
void* ratelprof_hook_make_stub(void* impl, void* wrapper);

/**
 * Patch the function at @p target (@p size bytes long if known, 0 otherwise)
 * so that it enters @p wrapper with the hook slot set to a trampoline
 * executing the original function.
 *
 * @return the trampoline, or NULL on failure (@p why describes the reason).
 */
void* ratelprof_hook_function(void* target, size_t size, void* wrapper, const char** why);

/** Number of functions patched so far. */
size_t ratelprof_hook_count(void);

#endif // RATELPROF_HOOK_H
