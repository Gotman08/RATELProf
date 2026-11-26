/**
 * @file ratelprof_static_hooks.h
 * @brief Interception of the traced functions that LD_PRELOAD cannot reach.
 *
 * LD_PRELOAD interposition only sees calls resolved by the dynamic linker. This
 * module completes it, for every enabled API domain:
 *
 *  - static copies: definitions linked statically into the executable, or
 *    hidden/local definitions inside a shared library (libcublas_static.a,
 *    libnccl_static.a...) found in the ELF symbol tables are patched in memory
 *    (ratelprof_hook.h) to enter the RATELProf wrappers;
 *  - libomptarget plugins: their entry points are patched whatever the way
 *    they are reached (__tgt_rtl_* functions resolved before RATELProf starts,
 *    GenericPluginTy methods of LLVM >= 19 called directly);
 *  - dlopen/dlsym users (ctypes, cupy, plugins loaded on demand): dlsym returns
 *    a stub entering the wrapper with the implementation found by dlsym;
 *  - libraries loaded after start-up: newly loaded objects are scanned and
 *    unresolved functions are resolved again when first called.
 *
 * Set RATELPROF_STATIC_HOOKS=0 to disable in-memory patching.
 */
#ifndef RATELPROF_STATIC_HOOKS_H
#define RATELPROF_STATIC_HOOKS_H

#include <stdbool.h>
#include <stddef.h>

/** Adapter pair routing a C++ plugin method to an omp-target domain function. */
typedef struct {
    const char* name;      /**< method name, GenericPluginTy::<name> */
    const char* params;    /**< expected demangled parameter list */
    int         id;        /**< omp-target domain function id */
    void*       entry;     /**< replacement of the method (takes `this`) */
    void*       impl;      /**< implementation of the domain function (calls the method) */
} ratelprof_method_hook_t;

/** dlsym of the C library, bypassing the RATELProf dlsym wrapper. */
void* ratelprof_real_dlsym(void* handle, const char* symbol);

/** Remember that an API-table domain is traced (called when it is enabled). */
void ratelprof_static_hooks_domain_enabled(int domain, bool enabled);

/** Scan the loaded objects and patch what LD_PRELOAD cannot intercept. */
void ratelprof_static_hooks_activate(void);

/** Stop handing out dlsym stubs (hooks already installed stay in place). */
void ratelprof_static_hooks_deactivate(void);

/** Scan the objects loaded since the last scan (cheap when nothing changed). */
void ratelprof_static_hooks_rescan(void);

/**
 * dlsym post-processing: when @p symbol is a traced function and @p real its
 * implementation, return a stub entering the RATELProf wrapper instead.
 */
void* ratelprof_static_hooks_dlsym(void* handle, const char* symbol, void* real, void* caller);

#endif // RATELPROF_STATIC_HOOKS_H
