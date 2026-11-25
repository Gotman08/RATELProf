/**
 * @file ratelprof_cupti.c
 * @brief CUPTI backend of RATELProf (NVIDIA GPUs).
 *
 *  - CUDA Runtime / Driver API calls   : CUPTI Callback API (works with a
 *                                         statically linked cudart).
 *  - NVTX markers and ranges            : CUPTI NVTX domain (NVTX_INJECTION64_PATH
 *                                         must point to libcupti, set by `ratelprof profile`).
 *  - Kernels, memcpy, memset            : CUPTI Activity API (CONCURRENT_KERNEL, MEMCPY,
 *                                         MEMCPY2, MEMSET).
 *  - Synchronizations                   : CUPTI Activity API (SYNCHRONIZATION), reported
 *                                         in the Barrier AND (stream wait event) and
 *                                         Barrier OR (host synchronizations) domains.
 *
 * The CUPTI correlation ID of an API call is shared by the runtime call and
 * the driver calls it performs, and by the GPU records it produces. A table maps
 * it to the RATELProf ID of the outermost traced API call so that GPU events
 * get the right parent (corr_id) in the report.
 *
 * All timestamps are CLOCK_MONOTONIC nanoseconds: a CUPTI timestamp callback is
 * registered when available, otherwise CUPTI timestamps (CLOCK_REALTIME) are
 * shifted by a constant offset.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <dlfcn.h>
#include <wchar.h>

#include <cupti.h>
#include <nvtx3/nvToolsExt.h>
#include <generated_nvtx_meta.h>

#include <ratelprof.h>
#include "ratelprof_ext.h"
#include "ratelprof_ext/ratelprof_cupti.h"
#include "memory_structure/cht.h"

#include "cupti_records.h"
#include "cuda_rt_api_helper.h"
#include "cuda_drv_api_helper.h"

#define RP_CUPTI_CALL(call)                                                        \
    do {                                                                           \
        CUptiResult _r = (call);                                                   \
        if (_r != CUPTI_SUCCESS) {                                                 \
            const char* _s = NULL;                                                 \
            cuptiGetResultString(_r, &_s);                                         \
            LOG(LOG_LEVEL_WARN, "CUPTI: %s failed: %s (%d)\n", #call, _s ? _s : "?", (int)_r); \
        }                                                                          \
    } while (0)

#ifndef CUPTI_SYNCHRONIZATION_INVALID_VALUE
#define CUPTI_SYNCHRONIZATION_INVALID_VALUE ((uint32_t)0xFFFFFFFF)
#endif
#ifndef CUPTI_TIMESTAMP_UNKNOWN
#define CUPTI_TIMESTAMP_UNKNOWN 0ull
#endif

/* Optional CUPTI entry points (resolved at run time, absent from old CUPTI). */
typedef uint64_t (CUPTIAPI *rp_cupti_timestamp_cb_t)(void);
typedef CUptiResult (CUPTIAPI *rp_register_ts_cb_fn)(rp_cupti_timestamp_cb_t);
typedef CUptiResult (CUPTIAPI *rp_enable_latency_ts_fn)(uint8_t);

#define RP_DEFAULT_BUFFER_SIZE (8u << 20)

typedef struct {
    bool        enabled[RATELPROF_NB_DOMAIN_EXT];
    uint8_t     traced_rt[CUDA_RT_API_ID_NB_FUNCTION];
    uint8_t     traced_drv[CUDA_DRV_API_ID_NB_FUNCTION];

    CUpti_SubscriberHandle subscriber;
    bool        subscribed;
    bool        activity_enabled;
    _Atomic bool active;

    cht_t*      corr_map;       /* CUPTI correlation id -> RATELProf API activity id */
    cht_t*      ctx_dev_map;    /* CUPTI context id     -> device id + 1             */
    cht_t*      nvtx_ranges;    /* nvtxRangeId_t        -> ratelprof_roctx_activity_t* */
    cht_t*      nvtx_strings;   /* nvtxStringHandle_t   -> char*                     */
    cht_t*      nvtx_domains;   /* nvtxDomainHandle_t   -> char*                     */

    size_t      buffer_size;
    bool        ts_callback;
    int64_t     ts_offset;      /* cupti_ts - monotonic_ns, when no callback */
    bool        callsite;
    _Atomic uint64_t dropped;
    _Atomic uint64_t records;
} rp_cupti_state_t;

static rp_cupti_state_t g = {0};

/* ------------------------------------------------------------------------- */
/* Time                                                                       */
/* ------------------------------------------------------------------------- */

static inline uint64_t rp_monotonic_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint64_t CUPTIAPI rp_cupti_timestamp(void)
{
    return rp_monotonic_ns();
}

static inline ratelprof_time_t rp_ts(uint64_t cupti_ts)
{
    if (cupti_ts == 0) return 0;
    return g.ts_callback ? cupti_ts : (ratelprof_time_t)((int64_t)cupti_ts - g.ts_offset);
}

/* ------------------------------------------------------------------------- */
/* Domain enabling and filtering                                              */
/* ------------------------------------------------------------------------- */

bool ratelprof_cupti_is_domain_enabled(ratelprof_domain_t domain)
{
    return (int)domain >= 0 && (int)domain < RATELPROF_NB_DOMAIN_EXT && g.enabled[domain];
}

static void rp_build_filter(ratelprof_domain_t domain, uint8_t* traced, size_t n)
{
    const char* list   = get_function_filter(domains_data[domain].functions_filtered_envname);
    filter_t    filter = get_filter_type(domains_data[domain].filter_type_envname);

    memset(traced, filter == WHITELIST ? 0 : 1, n);
    if (filter == NO_FILTER || !list) return;

    char* copy = strdup(list);
    for (char* tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
        for (size_t id = 0; id < n; id++) {
            const char* name = ratelprof_get_funame_by_id(domain, id);
            if (name && strcmp(name, tok) == 0) traced[id] = (filter == WHITELIST) ? 1 : 0;
        }
    }
    free(copy);
}

ratelprof_status_t ratelprof_cupti_enable_domain(ratelprof_domain_t domain)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_CUDA_RT:
            rp_build_filter(domain, g.traced_rt, CUDA_RT_API_ID_NB_FUNCTION);
            break;
        case RATELPROF_DOMAIN_CUDA_DRV:
            rp_build_filter(domain, g.traced_drv, CUDA_DRV_API_ID_NB_FUNCTION);
            break;
        case RATELPROF_DOMAIN_BARRIERAND:
        case RATELPROF_DOMAIN_BARRIEROR:
            g.enabled[RATELPROF_DOMAIN_BARRIERAND] = true;
            g.enabled[RATELPROF_DOMAIN_BARRIEROR]  = true;
            return RATELPROF_STATUS_SUCCESS;
        case RATELPROF_DOMAIN_NVTX:
        case RATELPROF_DOMAIN_PROFILING:
        case RATELPROF_DOMAIN_MEMORY:
        case RATELPROF_DOMAIN_KERNEL:
            break;
        default:
            return RATELPROF_STATUS_UNKNOWN_DOMAIN;
    }
    g.enabled[domain] = true;
    return RATELPROF_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Correlation                                                                */
/* ------------------------------------------------------------------------- */

static inline uint64_t rp_lookup_corr(uint32_t cupti_corr)
{
    if (!g.corr_map || cupti_corr == 0) return 0;
    return (uint64_t)(uintptr_t)cht_get(g.corr_map, cupti_corr);
}

static inline void rp_register_corr(uint32_t cupti_corr, uint64_t id)
{
    if (!g.corr_map || cupti_corr == 0 || id == 0) return;
    if (cht_get(g.corr_map, cupti_corr) == NULL)
        cht_set(g.corr_map, cupti_corr, (void*)(uintptr_t)id);
}

static inline uint32_t rp_device_of_context(uint32_t context_id)
{
    uintptr_t v = (uintptr_t)cht_get(g.ctx_dev_map, context_id);
    return v ? (uint32_t)(v - 1) : 0;
}

/* ------------------------------------------------------------------------- */
/* CUDA Runtime / Driver API callbacks                                        */
/* ------------------------------------------------------------------------- */

static void rp_on_api(ratelprof_domain_t domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cb)
{
    const bool is_rt = (domain == RATELPROF_DOMAIN_CUDA_RT);
    const size_t nb  = is_rt ? CUDA_RT_API_ID_NB_FUNCTION : CUDA_DRV_API_ID_NB_FUNCTION;
    const uint8_t* traced = is_rt ? g.traced_rt : g.traced_drv;

    if (!g.enabled[domain] || cbid >= nb || !traced[cbid]) {
        if (cb->callbackSite == CUPTI_API_ENTER) *cb->correlationData = 0;
        return;
    }

    if (cb->callbackSite == CUPTI_API_ENTER) {
        const uint32_t argsz = is_rt ? cuda_rt_args_size[cbid] : cuda_drv_args_size[cbid];
        const uint8_t  flags = is_rt ? cuda_rt_flags[cbid]     : cuda_drv_flags[cbid];

        ratelprof_api_activity_t* act = (ratelprof_api_activity_t*)
            ratelprof_activity_pool_alloc_activity(sizeof(ratelprof_api_activity_t) + argsz);
        act->args = (void*)(act + 1);
        act->return_address = g.callsite ? ratelprof_nv_find_callsite() : NULL;

        ratelprof_on_enter_callbacks[domain](domain, cbid, act);

        if (flags & RP_CUDA_FLAG_MAY_LAUNCH) {
            /* GPU records of this call will reference this activity. */
            rp_register_corr(cb->correlationId, act->id);
            ratelprof_mark_current_has_children();
        }

        *cb->correlationData = (uint64_t)(uintptr_t)act;
        act->start_time = ratelprof_get_clock_now();
    } else {
        ratelprof_api_activity_t* act = (ratelprof_api_activity_t*)(uintptr_t)(*cb->correlationData);
        if (!act) return;
        act->stop_time = ratelprof_get_clock_now();

        int64_t ret = 0;
        if (cb->functionReturnValue) ret = (int64_t)(*(const int*)cb->functionReturnValue);
        if (is_rt) rp_cuda_rt_capture_args(cbid, cb->functionParams, ret, act->args);
        else       rp_cuda_drv_capture_args(cbid, cb->functionParams, ret, act->args);

        ratelprof_on_exit_callbacks[domain](domain, cbid, act);
    }
}

/* ------------------------------------------------------------------------- */
/* NVTX                                                                       */
/* ------------------------------------------------------------------------- */

#define RP_NVTX_STACK_CAPACITY 1024
#define RP_NVTX_MAX_DOMAINS    16

typedef struct {
    nvtxDomainHandle_t domain;   /* NULL for the default domain */
    ratelprof_stack_t  stack;    /* open ratelprof_roctx_activity_t* */
} rp_nvtx_domain_stack_t;

static __thread rp_nvtx_domain_stack_t rp_nvtx_stacks[RP_NVTX_MAX_DOMAINS];
static __thread int rp_nvtx_nstacks = 0;

static char* rp_strdup_wide(const wchar_t* w)
{
    if (!w) return strdup("<null>");
    size_t n = wcslen(w);
    char* s = (char*)malloc(n + 1);
    for (size_t i = 0; i < n; i++) s[i] = (w[i] < 128 && w[i] > 0) ? (char)w[i] : '?';
    s[n] = '\0';
    return s;
}

static char* rp_nvtx_message(const nvtxEventAttributes_t* a, nvtxDomainHandle_t domain)
{
    char* msg = NULL;
    if (!a) {
        msg = strdup("<nvtx>");
    } else {
        switch (a->messageType) {
            case NVTX_MESSAGE_TYPE_ASCII:
                msg = strdup(a->message.ascii ? a->message.ascii : "<null>");
                break;
            case NVTX_MESSAGE_TYPE_UNICODE:
                msg = rp_strdup_wide(a->message.unicode);
                break;
            case NVTX_MESSAGE_TYPE_REGISTERED: { /* enum value, not a macro */
                const char* s = (const char*)cht_get(g.nvtx_strings, (uint64_t)(uintptr_t)a->message.registered);
                msg = strdup(s ? s : "<registered string>");
                break;
            }
            default:
                msg = strdup("<nvtx>");
        }
    }
    if (domain) {
        const char* dname = (const char*)cht_get(g.nvtx_domains, (uint64_t)(uintptr_t)domain);
        if (dname) {
            size_t len = strlen(dname) + strlen(msg) + 4;
            char* full = (char*)malloc(len);
            snprintf(full, len, "%s: %s", dname, msg);
            free(msg);
            msg = full;
        }
    }
    return msg;
}

static ratelprof_roctx_activity_t* rp_nvtx_new_activity(char* message)
{
    ratelprof_roctx_activity_t* act = (ratelprof_roctx_activity_t*)
        ratelprof_activity_pool_alloc_activity(sizeof(ratelprof_roctx_activity_t));
    act->domain = RATELPROF_DOMAIN_NVTX;
    act->message = message;
    act->pid = get_pid();
    act->tid = get_tid();
    act->has_children = false;
    act->corr_id = 0;
    act->return_address = g.callsite ? ratelprof_nv_find_callsite() : NULL;
    return act;
}

/* NVTX push/pop ranges are nested per thread *and per domain*: a pop on the
 * default domain must not close a range a library pushed on its own domain
 * (NCCL, for instance, never pops its ncclCommInitAll range). */
static ratelprof_stack_t* rp_nvtx_stack_of(nvtxDomainHandle_t domain, bool create)
{
    for (int i = 0; i < rp_nvtx_nstacks; i++)
        if (rp_nvtx_stacks[i].domain == domain) return &rp_nvtx_stacks[i].stack;
    if (!create || rp_nvtx_nstacks == RP_NVTX_MAX_DOMAINS) return NULL;
    rp_nvtx_domain_stack_t* ds = &rp_nvtx_stacks[rp_nvtx_nstacks];
    if (ratelprof_stack_init(&ds->stack, RP_NVTX_STACK_CAPACITY) != RATELPROF_STATUS_SUCCESS) return NULL;
    ds->domain = domain;
    rp_nvtx_nstacks++;
    return &ds->stack;
}

static void rp_nvtx_push(char* message, nvtxDomainHandle_t domain)
{
    ratelprof_stack_t* st = rp_nvtx_stack_of(domain, true);
    if (!st || ratelprof_stack_is_full(st)) { free(message); return; }
    ratelprof_roctx_activity_t* act = rp_nvtx_new_activity(message);
    ratelprof_stack_push(st, (uint64_t)(uintptr_t)act);
    /* Nested events get this range as parent. Weak: if the event enclosing the
     * range ends before it is popped, the range is closed with it. */
    ratelprof_get_weak_id(&act->id);
    act->start_time = ratelprof_get_clock_now();
}

/* Weak ID handler: a range was left open by the event that enclosed it (NCCL
 * pushes a range in ncclCommInitAll and never pops it): close it now. */
static void rp_nvtx_weak_closed(uint64_t id, uint64_t corr_id, bool has_children)
{
    for (int i = 0; i < rp_nvtx_nstacks; i++) {
        ratelprof_stack_t* st = &rp_nvtx_stacks[i].stack;
        for (int j = st->top; j >= 0; j--) {
            ratelprof_roctx_activity_t* act = (ratelprof_roctx_activity_t*)(uintptr_t)st->items[j];
            if (!act || act->id != id) continue;
            for (int k = j; k < st->top; k++) st->items[k] = st->items[k + 1];
            st->top--;
            act->stop_time = ratelprof_get_clock_now();
            act->corr_id = corr_id;
            act->has_children = has_children;
            ratelprof_activity_pool_push_activity(act);
            return;
        }
    }
}

static void rp_nvtx_pop(nvtxDomainHandle_t domain)
{
    uint64_t popped = 0;
    ratelprof_stack_t* st = rp_nvtx_stack_of(domain, false);
    if (!st || ratelprof_stack_pop(st, &popped) != RATELPROF_STATUS_SUCCESS || !popped) return;
    ratelprof_roctx_activity_t* act = (ratelprof_roctx_activity_t*)(uintptr_t)popped;
    act->stop_time = ratelprof_get_clock_now();
    /* The range may not be on top of the ID stack if another domain left a
     * range open inside it: close it by ID. */
    ratelprof_close_id(act->id, &act->corr_id, &act->has_children);
    ratelprof_activity_pool_push_activity(act);
}

static void rp_nvtx_mark(char* message)
{
    ratelprof_roctx_activity_t* act = rp_nvtx_new_activity(message);
    act->start_time = ratelprof_get_clock_now();
    act->stop_time  = act->start_time;
    get_id(&act->id);
    get_correlation_id(&act->corr_id, &act->has_children);
    ratelprof_activity_pool_push_activity(act);
}

static void rp_nvtx_range_start(char* message, nvtxRangeId_t id)
{
    ratelprof_roctx_activity_t* act = rp_nvtx_new_activity(message);
    act->start_time = ratelprof_get_clock_now();
    /* Start/End ranges may cross threads: allocate the id and record the parent
     * without leaving anything on the ID stack. */
    get_id(&act->id);
    get_correlation_id(&act->corr_id, NULL);
    cht_set(g.nvtx_ranges, (uint64_t)id, act);
}

static void rp_nvtx_range_end(nvtxRangeId_t id)
{
    ratelprof_roctx_activity_t* act = (ratelprof_roctx_activity_t*)cht_get(g.nvtx_ranges, (uint64_t)id);
    if (!act) return;
    cht_set(g.nvtx_ranges, (uint64_t)id, NULL);
    act->stop_time = ratelprof_get_clock_now();
    ratelprof_activity_pool_push_activity(act);
}

static void rp_on_nvtx(CUpti_CallbackId cbid, const CUpti_NvtxData* d)
{
    const void* p = d->functionParams;
    const void* r = d->functionReturnValue;

    /* Bookkeeping callbacks are always handled (strings / domain names). */
    switch (cbid) {
        case CUPTI_CBID_NVTX_nvtxDomainCreateA:
            if (r && p) cht_set(g.nvtx_domains, (uint64_t)(uintptr_t)*(const nvtxDomainHandle_t*)r,
                                strdup(((const nvtxDomainCreateA_params*)p)->name));
            return;
        case CUPTI_CBID_NVTX_nvtxDomainRegisterStringA:
            if (r && p) cht_set(g.nvtx_strings, (uint64_t)(uintptr_t)*(const nvtxStringHandle_t*)r,
                                strdup(((const nvtxDomainRegisterStringA_params*)p)->string));
            return;
        case CUPTI_CBID_NVTX_nvtxDomainRegisterStringW:
            if (r && p) cht_set(g.nvtx_strings, (uint64_t)(uintptr_t)*(const nvtxStringHandle_t*)r,
                                rp_strdup_wide((const wchar_t*)((const nvtxDomainRegisterStringW_params*)p)->string));
            return;
        default: break;
    }

    if (!g.enabled[RATELPROF_DOMAIN_NVTX]) return;

    /* nvtxRangePop has no parameter struct: CUPTI may hand us NULL params. */
    if (cbid == CUPTI_CBID_NVTX_nvtxRangePop) { rp_nvtx_pop(NULL); return; }
    if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePop) {
        rp_nvtx_pop(p ? ((const nvtxDomainRangePop_params*)p)->domain : NULL);
        return;
    }
    if (!p) return;

    switch (cbid) {
        case CUPTI_CBID_NVTX_nvtxMarkA:
            rp_nvtx_mark(strdup(((const nvtxMarkA_params*)p)->message));
            break;
        case CUPTI_CBID_NVTX_nvtxMarkW:
            rp_nvtx_mark(rp_strdup_wide(((const nvtxMarkW_params*)p)->message));
            break;
        case CUPTI_CBID_NVTX_nvtxMarkEx:
            rp_nvtx_mark(rp_nvtx_message(((const nvtxMarkEx_params*)p)->eventAttrib, NULL));
            break;
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx: {
            const nvtxDomainMarkEx_params* q = (const nvtxDomainMarkEx_params*)p;
            rp_nvtx_mark(rp_nvtx_message(q->core.eventAttrib, q->domain));
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushA:
            rp_nvtx_push(strdup(((const nvtxRangePushA_params*)p)->message), NULL);
            break;
        case CUPTI_CBID_NVTX_nvtxRangePushW:
            rp_nvtx_push(rp_strdup_wide(((const nvtxRangePushW_params*)p)->message), NULL);
            break;
        case CUPTI_CBID_NVTX_nvtxRangePushEx:
            rp_nvtx_push(rp_nvtx_message(((const nvtxRangePushEx_params*)p)->eventAttrib, NULL), NULL);
            break;
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
            const nvtxDomainRangePushEx_params* q = (const nvtxDomainRangePushEx_params*)p;
            rp_nvtx_push(rp_nvtx_message(q->core.eventAttrib, q->domain), q->domain);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangeStartA:
            if (r) rp_nvtx_range_start(strdup(((const nvtxRangeStartA_params*)p)->message), *(const nvtxRangeId_t*)r);
            break;
        case CUPTI_CBID_NVTX_nvtxRangeStartW:
            if (r) rp_nvtx_range_start(rp_strdup_wide(((const nvtxRangeStartW_params*)p)->message), *(const nvtxRangeId_t*)r);
            break;
        case CUPTI_CBID_NVTX_nvtxRangeStartEx:
            if (r) rp_nvtx_range_start(rp_nvtx_message(((const nvtxRangeStartEx_params*)p)->eventAttrib, NULL), *(const nvtxRangeId_t*)r);
            break;
        case CUPTI_CBID_NVTX_nvtxDomainRangeStartEx: {
            const nvtxDomainRangeStartEx_params* q = (const nvtxDomainRangeStartEx_params*)p;
            if (r) rp_nvtx_range_start(rp_nvtx_message(q->core.eventAttrib, q->domain), *(const nvtxRangeId_t*)r);
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangeEnd:
            rp_nvtx_range_end(((const nvtxRangeEnd_params*)p)->id);
            break;
        case CUPTI_CBID_NVTX_nvtxDomainRangeEnd:
            rp_nvtx_range_end(((const nvtxDomainRangeEnd_params*)p)->core.id);
            break;
        default:
            break;
    }
}

/* ------------------------------------------------------------------------- */
/* Resource callbacks                                                         */
/* ------------------------------------------------------------------------- */

static void rp_on_resource(CUpti_CallbackId cbid, const CUpti_ResourceData* r)
{
    if (!r) return;
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
        uint32_t ctx_id = 0, dev = 0;
        if (cuptiGetContextId(r->context, &ctx_id) == CUPTI_SUCCESS &&
            cuptiGetDeviceId(r->context, &dev) == CUPTI_SUCCESS) {
            cht_set(g.ctx_dev_map, ctx_id, (void*)(uintptr_t)(dev + 1));
            ratelprof_nv_register_device(dev);
        }
    } else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
        /* Records of a context must be collected before it disappears. */
        if (g.activity_enabled) cuptiActivityFlushAll(0);
    }
}

static void CUPTIAPI rp_cupti_callback(void* userdata, CUpti_CallbackDomain domain,
                                       CUpti_CallbackId cbid, const void* cbdata)
{
    (void)userdata;
    if (!atomic_load_explicit(&g.active, memory_order_acquire)) return;
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            rp_on_api(RATELPROF_DOMAIN_CUDA_RT, cbid, (const CUpti_CallbackData*)cbdata);
            break;
        case CUPTI_CB_DOMAIN_DRIVER_API:
            rp_on_api(RATELPROF_DOMAIN_CUDA_DRV, cbid, (const CUpti_CallbackData*)cbdata);
            break;
        case CUPTI_CB_DOMAIN_NVTX:
            rp_on_nvtx(cbid, (const CUpti_NvtxData*)cbdata);
            break;
        case CUPTI_CB_DOMAIN_RESOURCE:
            rp_on_resource(cbid, (const CUpti_ResourceData*)cbdata);
            break;
        default:
            break;
    }
}

/* ------------------------------------------------------------------------- */
/* Activity records                                                           */
/* ------------------------------------------------------------------------- */

static ratelprof_gpu_activity_t* rp_new_gpu_activity(ratelprof_domain_ext_t domain, uint32_t cupti_corr,
                                                     uint64_t start, uint64_t end, uint32_t device,
                                                     uint32_t context, uint32_t stream)
{
    ratelprof_gpu_activity_t* a = (ratelprof_gpu_activity_t*)
        ratelprof_activity_pool_alloc_activity(sizeof(ratelprof_gpu_activity_t));
    memset(a, 0, sizeof(*a));
    a->domain        = domain;
    a->id            = ratelprof_new_id();
    a->corr_id       = rp_lookup_corr(cupti_corr);
    a->start_time    = rp_ts(start);
    a->stop_time     = rp_ts(end);
    if (a->stop_time < a->start_time) a->stop_time = a->start_time;
    a->device_id     = device;
    a->context_id    = context;
    a->stream_id     = stream;
    a->cupti_corr_id = cupti_corr;
    a->gpu_handle    = RATELPROF_NV_GPU_AGENT_HANDLE(device);
    ratelprof_nv_register_device(device);
    return a;
}

static void rp_copy_types(uint8_t copy_kind, uint8_t* src_type, uint8_t* dst_type)
{
    /* 0 = host, 1 = device (arrays are device memory) */
    switch (copy_kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: *src_type = 0; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: *src_type = 1; *dst_type = 0; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: *src_type = 0; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: *src_type = 1; *dst_type = 0; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: *src_type = 1; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: *src_type = 1; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: *src_type = 1; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: *src_type = 1; *dst_type = 1; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: *src_type = 0; *dst_type = 0; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: *src_type = 1; *dst_type = 1; break;
        default:                              *src_type = 0; *dst_type = 1; break;
    }
}

static void rp_process_kernel(const rp_cupti_kernel_t* k)
{
    if (!g.enabled[RATELPROF_DOMAIN_KERNEL]) return;
    ratelprof_gpu_activity_t* a = rp_new_gpu_activity(RATELPROF_DOMAIN_KERNEL, k->correlationId,
        k->start, k->end, k->deviceId, k->contextId, k->streamId);
    ratelprof_nv_kernel_args_t* ka = &a->args.kernel;
#ifdef RP_CUPTI_KERNEL_T_HAS_QUEUED
    ka->queued    = (k->queued == CUPTI_TIMESTAMP_UNKNOWN) ? 0 : rp_ts(k->queued);
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_SUBMITTED
    ka->submitted = (k->submitted == CUPTI_TIMESTAMP_UNKNOWN) ? 0 : rp_ts(k->submitted);
#endif
    ka->grid[0]  = k->gridX;  ka->grid[1]  = k->gridY;  ka->grid[2]  = k->gridZ;
    ka->block[0] = k->blockX; ka->block[1] = k->blockY; ka->block[2] = k->blockZ;
#ifdef RP_CUPTI_KERNEL_T_HAS_CLUSTERX
    ka->cluster[0] = k->clusterX; ka->cluster[1] = k->clusterY; ka->cluster[2] = k->clusterZ;
#endif
    ka->registers_per_thread  = k->registersPerThread;
    ka->static_shared_memory  = k->staticSharedMemory;
    ka->dynamic_shared_memory = k->dynamicSharedMemory;
#ifdef RP_CUPTI_KERNEL_T_HAS_LOCALMEMORYPERTHREAD
    ka->local_memory_per_thread = k->localMemoryPerThread;
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_LOCALMEMORYTOTAL
    ka->local_memory_total = k->localMemoryTotal;
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_GRAPHNODEID
    ka->graph_node_id = k->graphNodeId;
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_LAUNCHTYPE
    ka->launch_type = (uint8_t)k->launchType;
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_CACHECONFIG
    ka->cache_config_requested = k->cacheConfig.config.requested;
    ka->cache_config_executed  = k->cacheConfig.config.executed;
#endif
#ifdef RP_CUPTI_KERNEL_T_HAS_SHAREDMEMORYCONFIG
    ka->shared_memory_config = k->sharedMemoryConfig;
#endif
    ka->kernel_object = ratelprof_nv_intern_kernel_name(k->name ? k->name : "<unknown kernel>");
    ratelprof_activity_pool_push_activity(a);
}

static void rp_process_memcpy(const rp_cupti_memcpy_t* m)
{
    if (!g.enabled[RATELPROF_DOMAIN_MEMORY]) return;
    ratelprof_gpu_activity_t* a = rp_new_gpu_activity(RATELPROF_DOMAIN_MEMORY, m->correlationId,
        m->start, m->end, m->deviceId, m->contextId, m->streamId);
    ratelprof_nv_memory_args_t* ma = &a->args.memory;
    ma->memop     = RATELPROF_MEMORY_OP_CUDA_MEMCPY;
    ma->bytes     = m->bytes;
    ma->copy_kind = m->copyKind;
    ma->src_kind  = m->srcKind;
    ma->dst_kind  = m->dstKind;
    ma->is_async  = (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 1 : 0;
#ifdef RP_CUPTI_MEMCPY_T_HAS_GRAPHNODEID
    ma->graph_node_id = m->graphNodeId;
#endif
    rp_copy_types(m->copyKind, &ma->src_type, &ma->dst_type);
    ma->other_handle = (ma->src_type == 0 || ma->dst_type == 0) ? RATELPROF_NV_HOST_AGENT_HANDLE : a->gpu_handle;
    ratelprof_activity_pool_push_activity(a);
}

static void rp_process_memcpy_p2p(const rp_cupti_memcpy_p2p_t* m)
{
    if (!g.enabled[RATELPROF_DOMAIN_MEMORY]) return;
    ratelprof_gpu_activity_t* a = rp_new_gpu_activity(RATELPROF_DOMAIN_MEMORY, m->correlationId,
        m->start, m->end, m->deviceId, m->contextId, m->streamId);
    ratelprof_nv_memory_args_t* ma = &a->args.memory;
    ma->memop     = RATELPROF_MEMORY_OP_CUDA_MEMCPY_P2P;
    ma->bytes     = m->bytes;
    ma->copy_kind = m->copyKind;
    ma->src_kind  = m->srcKind;
    ma->dst_kind  = m->dstKind;
    ma->is_async  = (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 1 : 0;
#ifdef RP_CUPTI_MEMCPY_P2P_T_HAS_GRAPHNODEID
    ma->graph_node_id = m->graphNodeId;
#endif
    ma->src_type = 1;
    ma->dst_type = 1;
    uint32_t other = (m->srcDeviceId == m->deviceId) ? m->dstDeviceId : m->srcDeviceId;
    ratelprof_nv_register_device(other);
    ma->other_handle = RATELPROF_NV_GPU_AGENT_HANDLE(other);
    ratelprof_activity_pool_push_activity(a);
}

static void rp_process_memset(const rp_cupti_memset_t* m)
{
    if (!g.enabled[RATELPROF_DOMAIN_MEMORY]) return;
    ratelprof_gpu_activity_t* a = rp_new_gpu_activity(RATELPROF_DOMAIN_MEMORY, m->correlationId,
        m->start, m->end, m->deviceId, m->contextId, m->streamId);
    ratelprof_nv_memory_args_t* ma = &a->args.memory;
    ma->memop  = RATELPROF_MEMORY_OP_CUDA_MEMSET;
    ma->bytes  = m->bytes;
    ma->value  = m->value;
#ifdef RP_CUPTI_MEMSET_T_HAS_MEMORYKIND
    ma->dst_kind = (uint8_t)m->memoryKind;
#endif
#ifdef RP_CUPTI_MEMSET_T_HAS_FLAGS
    ma->is_async = (m->flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC) ? 1 : 0;
#endif
#ifdef RP_CUPTI_MEMSET_T_HAS_GRAPHNODEID
    ma->graph_node_id = m->graphNodeId;
#endif
    ma->src_type = 1;
    ma->dst_type = 1;
    ma->other_handle = a->gpu_handle;
    ratelprof_activity_pool_push_activity(a);
}

static void rp_process_sync(const rp_cupti_sync_t* y)
{
    if (!g.enabled[RATELPROF_DOMAIN_BARRIERAND]) return;
    const bool gpu_side = (y->type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
    uint32_t stream = (y->streamId == CUPTI_SYNCHRONIZATION_INVALID_VALUE) ? RATELPROF_NV_DEVICE_WIDE_STREAM : y->streamId;
    uint32_t dev = rp_device_of_context(y->contextId);
    ratelprof_gpu_activity_t* a = rp_new_gpu_activity(
        gpu_side ? RATELPROF_DOMAIN_BARRIERAND : RATELPROF_DOMAIN_BARRIEROR,
        y->correlationId, y->start, y->end, dev, y->contextId, stream);
    a->args.sync.type = (uint32_t)y->type;
    a->args.sync.cuda_event_id = y->cudaEventId;
    ratelprof_activity_pool_push_activity(a);
}

static void rp_process_record(const CUpti_Activity* r)
{
    atomic_fetch_add_explicit(&g.records, 1, memory_order_relaxed);
    switch (r->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            rp_process_kernel((const rp_cupti_kernel_t*)r);
            break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
            rp_process_memcpy((const rp_cupti_memcpy_t*)r);
            break;
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
            rp_process_memcpy_p2p((const rp_cupti_memcpy_p2p_t*)r);
            break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
            rp_process_memset((const rp_cupti_memset_t*)r);
            break;
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
            rp_process_sync((const rp_cupti_sync_t*)r);
            break;
        default:
            break;
    }
}

static void CUPTIAPI rp_buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records)
{
    *size = g.buffer_size;
    *buffer = (uint8_t*)aligned_alloc(8, g.buffer_size);
    *max_records = 0;
    if (!*buffer) {
        LOG(LOG_LEVEL_ERROR, "CUPTI: cannot allocate an activity buffer of %zu bytes.\n", g.buffer_size);
        *size = 0;
    }
}

static void rp_profiling_event(profiling_id_t id, ratelprof_clock_t start, ratelprof_clock_t stop)
{
    if (!g.enabled[RATELPROF_DOMAIN_PROFILING]) return;
    ratelprof_api_activity_t* act = (ratelprof_api_activity_t*)
        ratelprof_activity_pool_alloc_activity(sizeof(ratelprof_api_activity_t));
    memset(act, 0, sizeof(*act));
    act->domain = (ratelprof_domain_t)RATELPROF_DOMAIN_PROFILING;
    act->funid  = id;
    act->pid    = get_pid();
    act->tid    = get_tid();
    get_id(&act->id);
    get_correlation_id(&act->corr_id, &act->has_children);
    act->start_time = start;
    act->stop_time  = stop;
    act->return_address = NULL;
    act->args = NULL;
    ratelprof_activity_pool_push_activity(act);
}

static void CUPTIAPI rp_buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t* buffer,
                                         size_t size, size_t valid_size)
{
    (void)size;
    ratelprof_clock_t t0 = ratelprof_get_clock_now();
    CUpti_Activity* record = NULL;
    if (valid_size > 0) {
        while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
            rp_process_record(record);
        }
    }
    size_t dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped) == CUPTI_SUCCESS && dropped) {
        atomic_fetch_add_explicit(&g.dropped, dropped, memory_order_relaxed);
    }
    free(buffer);
    rp_profiling_event(PROFILING_ID_cuptiBufferCompleted, t0, ratelprof_get_clock_now());
}

/* ------------------------------------------------------------------------- */
/* Life cycle                                                                 */
/* ------------------------------------------------------------------------- */

ratelprof_status_t ratelprof_cupti_init(void)
{
    g.buffer_size = (size_t)envtoll("RATELPROF_CUPTI_BUFFER_SIZE", RP_DEFAULT_BUFFER_SIZE);
    if (g.buffer_size < (64u << 10)) g.buffer_size = 64u << 10;
    g.callsite = envtoi("RATELPROF_CUDA_CALLSITE", 1) != 0;

    g.corr_map     = cht_create(1u << 16);
    g.ctx_dev_map  = cht_create(64);
    g.nvtx_ranges  = cht_create(1024);
    g.nvtx_strings = cht_create(1024);
    g.nvtx_domains = cht_create(64);
    if (!g.corr_map || !g.ctx_dev_map || !g.nvtx_ranges || !g.nvtx_strings || !g.nvtx_domains)
        return RATELPROF_STATUS_MALLOC_FAILED;

    atomic_store(&g.active, false);
    return RATELPROF_STATUS_SUCCESS;
}

static int rp_count_devices(void)
{
    typedef int (*cuInit_t)(unsigned int);
    typedef int (*cuDeviceGetCount_t)(int*);
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h) return 0;
    cuInit_t p_init = (cuInit_t)dlsym(h, "cuInit");
    cuDeviceGetCount_t p_count = (cuDeviceGetCount_t)dlsym(h, "cuDeviceGetCount");
    int n = 0;
    if (p_init && p_count && p_init(0) == 0 && p_count(&n) == 0) return n;
    return 0;
}

ratelprof_status_t ratelprof_cupti_start(void)
{
    const bool want_api   = g.enabled[RATELPROF_DOMAIN_CUDA_RT] || g.enabled[RATELPROF_DOMAIN_CUDA_DRV];
    const bool want_nvtx  = g.enabled[RATELPROF_DOMAIN_NVTX];
    const bool want_gpu   = g.enabled[RATELPROF_DOMAIN_KERNEL] || g.enabled[RATELPROF_DOMAIN_MEMORY] ||
                            g.enabled[RATELPROF_DOMAIN_BARRIERAND];

    ratelprof_nv_register_agents(rp_count_devices());

    if (!want_api && !want_nvtx && !want_gpu) return RATELPROF_STATUS_SUCCESS;

    if (g.callsite && (want_api || want_nvtx)) ratelprof_nv_callsite_init();

    uint32_t version = 0;
    cuptiGetVersion(&version);
    LOG(LOG_LEVEL_DEBUG, "CUPTI: runtime version %u, built with CUPTI_API_VERSION %d (kernel record %s)\n",
        version, CUPTI_API_VERSION, RP_CUPTI_KERNEL_T_NAME);

    /* Timestamps in CLOCK_MONOTONIC (same clock as the GPU events of the AMD backend). */
    rp_register_ts_cb_fn reg_ts = (rp_register_ts_cb_fn)dlsym(RTLD_DEFAULT, "cuptiActivityRegisterTimestampCallback");
    if (reg_ts && envtoi("RATELPROF_CUPTI_TIMESTAMP_CALLBACK", 1) && reg_ts(rp_cupti_timestamp) == CUPTI_SUCCESS) {
        g.ts_callback = true;
    } else {
        uint64_t c0 = 0;
        uint64_t m0 = rp_monotonic_ns();
        cuptiGetTimestamp(&c0);
        uint64_t m1 = rp_monotonic_ns();
        g.ts_callback = false;
        g.ts_offset = (int64_t)c0 - (int64_t)((m0 + m1) / 2);
    }

    ratelprof_set_weak_id_handler(rp_nvtx_weak_closed);
    ratelprof_clock_t t0 = ratelprof_get_clock_now();
    RP_CUPTI_CALL(cuptiSubscribe(&g.subscriber, (CUpti_CallbackFunc)rp_cupti_callback, NULL));
    g.subscribed = true;
    RP_CUPTI_CALL(cuptiEnableDomain(1, g.subscriber, CUPTI_CB_DOMAIN_RESOURCE));
    if (g.enabled[RATELPROF_DOMAIN_CUDA_RT])  RP_CUPTI_CALL(cuptiEnableDomain(1, g.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    if (g.enabled[RATELPROF_DOMAIN_CUDA_DRV]) RP_CUPTI_CALL(cuptiEnableDomain(1, g.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    if (want_nvtx)                            RP_CUPTI_CALL(cuptiEnableDomain(1, g.subscriber, CUPTI_CB_DOMAIN_NVTX));
    rp_profiling_event(PROFILING_ID_cuptiSubscribe, t0, ratelprof_get_clock_now());

    if (want_gpu) {
        t0 = ratelprof_get_clock_now();
        RP_CUPTI_CALL(cuptiActivityRegisterCallbacks(rp_buffer_requested, rp_buffer_completed));
        rp_enable_latency_ts_fn latency = (rp_enable_latency_ts_fn)dlsym(RTLD_DEFAULT, "cuptiActivityEnableLatencyTimestamps");
        if (latency && g.enabled[RATELPROF_DOMAIN_KERNEL]) latency(1);
        if (g.enabled[RATELPROF_DOMAIN_KERNEL])
            RP_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        if (g.enabled[RATELPROF_DOMAIN_MEMORY]) {
            RP_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
            RP_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
            RP_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        }
        if (g.enabled[RATELPROF_DOMAIN_BARRIERAND])
            RP_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
        g.activity_enabled = true;
        rp_profiling_event(PROFILING_ID_cuptiActivityEnable, t0, ratelprof_get_clock_now());
    }

    atomic_store_explicit(&g.active, true, memory_order_release);
    return RATELPROF_STATUS_SUCCESS;
}

ratelprof_status_t ratelprof_cupti_stop(void)
{
    ratelprof_set_weak_id_handler(NULL);
    if (g.activity_enabled) {
        ratelprof_clock_t t0 = ratelprof_get_clock_now();
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        rp_profiling_event(PROFILING_ID_cuptiActivityFlushAll, t0, ratelprof_get_clock_now());
    }
    atomic_store_explicit(&g.active, false, memory_order_release);

    uint64_t dropped = atomic_load(&g.dropped);
    if (dropped) {
        LOG(LOG_LEVEL_WARN, "CUPTI dropped %lu activity records. Increase RATELPROF_CUPTI_BUFFER_SIZE (current %zu).\n",
            (unsigned long)dropped, g.buffer_size);
    }
    LOG(LOG_LEVEL_DEBUG, "CUPTI: %lu activity records processed.\n", (unsigned long)atomic_load(&g.records));
    return RATELPROF_STATUS_SUCCESS;
}

ratelprof_status_t ratelprof_cupti_fini(void)
{
    /* The CUPTI subscriber is intentionally kept: application threads may still
     * be inside CUDA calls while the process exits. Callbacks are disabled by
     * the 'active' flag. */
    return RATELPROF_STATUS_SUCCESS;
}
