/**
 * @file ratelprof_ext_nvidia.c
 * @brief Extension layer entry points of the NVIDIA backend.
 *
 * Implements the public RATELProf API on top of the core implementation
 * (__ratelprof_*_impl) exactly like src/ext/src/ratelprof_ext/ratelprof_ext.c
 * does for AMD, plus the CUPTI life cycle (start/stop).
 */
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <ratelprof.h>
#include "ratelprof_ext.h"
#include "ratelprof/ratelprof_static_hooks.h"
#include "ratelprof_internal.h"
#include "ratelprof_ext/ratelprof_cupti.h"
#include "memory_structure/cht.h"

/* ------------------------------------------------------------------------- */
/* Kernel names                                                               */
/* ------------------------------------------------------------------------- */

static cht_t* kernel_names = NULL;
static pthread_mutex_t kernel_names_lock = PTHREAD_MUTEX_INITIALIZER;

static uint64_t fnv1a(const char* s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    /* Kept below 2^48: the visualizer decodes it as a JavaScript number. */
    h &= 0xFFFFFFFFFFFFull;
    return h ? h : 1;
}

uint64_t ratelprof_nv_intern_kernel_name(const char* name)
{
    uint64_t key = fnv1a(name);
    for (;;) {
        const char* cur = (const char*)cht_get(kernel_names, key);
        if (cur == NULL) {
            pthread_mutex_lock(&kernel_names_lock);
            cur = (const char*)cht_get(kernel_names, key);
            if (cur == NULL) {
                cht_set(kernel_names, key, strdup(name));
                pthread_mutex_unlock(&kernel_names_lock);
                return key;
            }
            pthread_mutex_unlock(&kernel_names_lock);
        }
        if (strcmp(cur, name) == 0) return key;
        key = ((key + 1) & 0xFFFFFFFFFFFFull) | 1u; /* hash collision: linear probing */
    }
}

const char* get_kernel_name(uint64_t kernel_object)
{
    const char* name = kernel_names ? (const char*)cht_get(kernel_names, kernel_object) : NULL;
    return name ? name : "N/A";
}

char* get_copy_name(uint32_t src_type, uint32_t dst_type)
{
    static char* table[2][2] = {
        { "CopyHostToHost",   "CopyHostToDevice" },
        { "CopyDeviceToHost", "CopyDeviceToDevice" }
    };
    if (src_type > 1 || dst_type > 1) return "CopyUnknown";
    return table[src_type][dst_type];
}

/* ------------------------------------------------------------------------- */
/* Agents (host CPU + CUDA devices)                                           */
/* ------------------------------------------------------------------------- */

static pthread_mutex_t agents_lock = PTHREAD_MUTEX_INITIALIZER;

static void register_agent(uint64_t handle, uint32_t node, uint8_t type)
{
    ratelprof_agent_object_t* existing = NULL;
    pthread_mutex_lock(&agents_lock);
    if (ratelprof_object_tracking_pool_get_agent_from_handle(handle, &existing) != RATELPROF_STATUS_SUCCESS) {
        ratelprof_agent_object_t* a = (ratelprof_agent_object_t*)malloc(sizeof(ratelprof_agent_object_t));
        if (a) {
            a->agent.handle = handle;
            a->node = node;
            a->type = type;
            ratelprof_object_tracking_pool_map_new_agent_to_handle(handle, a);
        }
    }
    pthread_mutex_unlock(&agents_lock);
}

void ratelprof_nv_register_device(uint32_t device_id)
{
    register_agent(RATELPROF_NV_GPU_AGENT_HANDLE(device_id), RATELPROF_NV_GPU_NODE(device_id), 1);
}

void ratelprof_nv_register_agents(int ndevices)
{
    register_agent(RATELPROF_NV_HOST_AGENT_HANDLE, RATELPROF_NV_HOST_NODE, 0);
    for (int d = 0; d < ndevices; d++) ratelprof_nv_register_device((uint32_t)d);
}

ratelprof_status_t ratelprof_enable_node_id_tracking()
{
    /* Agents are registered when CUPTI starts (see ratelprof_start). */
    return RATELPROF_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Life cycle                                                                 */
/* ------------------------------------------------------------------------- */

ratelprof_status_t ratelprof_init()
{
    ratelprof_status_t status = RATELPROF_STATUS_SUCCESS;

    RATELPROF_TRY(__ratelprof_init_impl(RATELPROF_NB_DOMAIN_EXT));

    kernel_names = cht_create(envtoi("RATELPROF_NB_KERNEL_AVAILABLE", 1024) < 64 ? 64 :
                              (size_t)envtoi("RATELPROF_NB_KERNEL_AVAILABLE", 1024));
    if (!kernel_names) return RATELPROF_STATUS_MALLOC_FAILED;

    RATELPROF_TRY(ratelprof_object_tracking_pool_init(),
        LOG(LOG_LEVEL_ERROR, "Cannot init object tracking pool. %s (code %d)\n", get_error_string_ext(status), status));

    RATELPROF_TRY(ratelprof_cupti_init(),
        LOG(LOG_LEVEL_ERROR, "Cannot init the CUPTI backend. %s (code %d)\n", get_error_string_ext(status), status));

    return status;
}

ratelprof_status_t ratelprof_fini()
{
    ratelprof_status_t status = RATELPROF_STATUS_SUCCESS;
    RATELPROF_TRY(__ratelprof_fini_impl());
    RATELPROF_TRY(ratelprof_cupti_fini());
    RATELPROF_TRY(ratelprof_object_tracking_pool_fini());
    return status;
}

ratelprof_status_t ratelprof_start(void)
{
    ratelprof_status_t status = __ratelprof_start_impl();
    if (status != RATELPROF_STATUS_SUCCESS) return status;
    status = ratelprof_cupti_start();
    /* Functions LD_PRELOAD cannot reach: static copies, plugins, dlsym users. */
    ratelprof_static_hooks_activate();
    return status;
}

ratelprof_status_t ratelprof_stop(void)
{
    ratelprof_static_hooks_deactivate();
    ratelprof_cupti_stop();
    return __ratelprof_stop_impl();
}

/* ------------------------------------------------------------------------- */
/* Domains                                                                    */
/* ------------------------------------------------------------------------- */

const char* ratelprof_get_domain_name(ratelprof_domain_t domain)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_NVTX:       return RATELPROF_DOMAIN_NVTX_NAME;
        case RATELPROF_DOMAIN_MEMORY:     return RATELPROF_DOMAIN_MEMORY_NAME;
        case RATELPROF_DOMAIN_KERNEL:     return RATELPROF_DOMAIN_KERNEL_NAME;
        case RATELPROF_DOMAIN_BARRIEROR:  return RATELPROF_DOMAIN_BARRIEROR_NAME;
        case RATELPROF_DOMAIN_BARRIERAND: return RATELPROF_DOMAIN_BARRIERAND_NAME;
        case RATELPROF_DOMAIN_PROFILING:  return RATELPROF_DOMAIN_PROFILING_NAME;
        case RATELPROF_DOMAIN_OMP_REGION: return RATELPROF_DOMAIN_OMP_REGION_NAME;
        default: return __ratelprof_get_domain_name_impl(domain);
    }
}

ratelprof_status_t ratelprof_enable_domain(ratelprof_domain_t domain)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_CUDA_RT:
        case RATELPROF_DOMAIN_CUDA_DRV:
        case RATELPROF_DOMAIN_NVTX:
        case RATELPROF_DOMAIN_PROFILING:
        case RATELPROF_DOMAIN_MEMORY:
        case RATELPROF_DOMAIN_KERNEL:
        case RATELPROF_DOMAIN_BARRIERAND:
        case RATELPROF_DOMAIN_BARRIEROR:
            return ratelprof_cupti_enable_domain(domain);
        case RATELPROF_DOMAIN_OMP_REGION:
            return RATELPROF_STATUS_SUCCESS; // OMPT is enabled by ompt_start_tool
        default:
            if (domains_data[domain].api_table_addr == NULL) {
                LOG(LOG_LEVEL_WARN, "Domain '%s' is not available in this build.\n", ratelprof_get_domain_name(domain));
                return RATELPROF_STATUS_SUCCESS;
            }
            ratelprof_static_hooks_domain_enabled((int)domain, true);
            return __ratelprof_enable_domain_impl(domain);
    }
}

ratelprof_status_t ratelprof_disable_domain(ratelprof_domain_t domain)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_CUDA_RT:
        case RATELPROF_DOMAIN_CUDA_DRV:
        case RATELPROF_DOMAIN_NVTX:
        case RATELPROF_DOMAIN_PROFILING:
        case RATELPROF_DOMAIN_MEMORY:
        case RATELPROF_DOMAIN_KERNEL:
        case RATELPROF_DOMAIN_BARRIERAND:
        case RATELPROF_DOMAIN_BARRIEROR:
            return RATELPROF_STATUS_ERROR;
        case RATELPROF_DOMAIN_OMP_REGION:
            return RATELPROF_STATUS_SUCCESS;
        default:
            ratelprof_static_hooks_domain_enabled((int)domain, false);
            return __ratelprof_disable_domain_impl(domain);
    }
}

const char* ratelprof_get_funame_by_id(ratelprof_domain_t domain, ratelprof_api_id_t funid)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_PROFILING:  return get_profiling_funame_by_id((profiling_id_t)funid);
        case RATELPROF_DOMAIN_NVTX:       return "nvtx";
        case RATELPROF_DOMAIN_OMP_REGION: return get_ompt_funame_by_id(funid);
        default: return __ratelprof_get_funame_by_id_impl(domain, funid);
    }
}

ratelprof_api_id_t ratelprof_get_funid_by_name(ratelprof_domain_t domain, const char* funame)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_OMP_REGION:
        case RATELPROF_DOMAIN_PROFILING:
        case RATELPROF_DOMAIN_NVTX:
            return -1;
        default: return __ratelprof_get_funid_by_name_impl(domain, funame);
    }
}

void* ratelprof_get_funaddr_by_id(ratelprof_domain_t domain, ratelprof_api_id_t funid)
{
    switch ((int)domain) {
        case RATELPROF_DOMAIN_OMP_REGION:
        case RATELPROF_DOMAIN_PROFILING:
        case RATELPROF_DOMAIN_NVTX:
            return NULL;
        default: return __ratelprof_get_funaddr_by_id_impl(domain, funid);
    }
}
