/**
 * @file ratelprof_ext_nvidia.h
 * @brief RATELProf extension layer for NVIDIA GPUs (CUPTI backend).
 *
 * This header is included by ratelprof_ext.h when RATELPROF_VENDOR_NVIDIA is
 * defined. It exposes the same API as the AMD extension layer:
 *   - the extended domain enumeration (same numeric values as AMD),
 *   - the GPU / marker activity structures consumed by the report encoder,
 *   - kernel name / copy name helpers.
 *
 * GPU activities (kernels, memory copies, memsets, synchronizations) are
 * collected with the CUPTI Activity API, CUDA Runtime / Driver API calls with
 * the CUPTI Callback API and NVTX ranges/markers through the CUPTI NVTX
 * injection. See src/ext/src/nvidia/.
 */
#ifndef RATELPROF_EXT_NVIDIA_H
#define RATELPROF_EXT_NVIDIA_H

#include <stdint.h>
#include <stdbool.h>
#include <ratelprof.h>

#include "utils/mpi_helper.h"

#include "ratelprof_ext/ratelprof_ext_status.h"
#include "ratelprof_ext/ratelprof_gpu_handles.h"
#include "ratelprof_ext/ratelprof_object_tracking.h"

#include "domains/ompt_api_helper.h"
#include "domains/memory_op_helper.h"

#include "ratelprof_version.h"

#define RATELPROF_DOMAIN_MEMORY_NAME        "RATELPROF_DOMAIN_MEMORY"
#define RATELPROF_DOMAIN_KERNEL_NAME        "RATELPROF_DOMAIN_KERNEL"
#define RATELPROF_DOMAIN_BARRIEROR_NAME     "RATELPROF_DOMAIN_BARRIEROR"
#define RATELPROF_DOMAIN_BARRIERAND_NAME    "RATELPROF_DOMAIN_BARRIERAND"
#define RATELPROF_DOMAIN_PROFILING_NAME     "RATELPROF_DOMAIN_PROFILING"
#define RATELPROF_DOMAIN_OMP_REGION_NAME    "RATELPROF_DOMAIN_OMP_REGION"
#define RATELPROF_DOMAIN_NVTX_NAME          "RATELPROF_DOMAIN_NVTX"
#define RATELPROF_DOMAIN_ROCTX_NAME         RATELPROF_DOMAIN_NVTX_NAME

/*
 * Extended domains. Numeric values are identical to the AMD backend:
 *   ROCTX            <-> NVTX
 *   HSA for RATELProf <-> CUPTI for RATELProf (profiler overhead)
 *   BARRIERAND       <-> CUDA stream-wait-event synchronizations (GPU side)
 *   BARRIEROR        <-> CUDA host synchronizations (event/stream/context synchronize)
 */
typedef enum {
    RATELPROF_DOMAIN_OMP_REGION = RATELPROF_NB_DOMAIN,
    RATELPROF_DOMAIN_NVTX,
    RATELPROF_DOMAIN_PROFILING,
    RATELPROF_DOMAIN_MEMORY,
    RATELPROF_DOMAIN_KERNEL,
    RATELPROF_DOMAIN_BARRIERAND,
    RATELPROF_DOMAIN_BARRIEROR,
    RATELPROF_NB_DOMAIN_EXT,    /**< Total number including extensions */
    RATELPROF_DOMAIN_GPU        // Not a real domain.
} ratelprof_domain_ext_t;

#define RATELPROF_DOMAIN_ROCTX RATELPROF_DOMAIN_NVTX

/* ------------------------------------------------------------------------- */
/* Agents                                                                     */
/* ------------------------------------------------------------------------- */

/** Agent handle of CUDA device 'dev' (handles are never 0). */
#define RATELPROF_NV_GPU_AGENT_HANDLE(dev)  ((uint64_t)(dev) + 1u)
/** Node of CUDA device 'dev' (matches the lsgpu NVIDIA node numbering). */
#define RATELPROF_NV_GPU_NODE(dev)          ((uint32_t)(dev))
/** Agent handle and node used for the host CPU. */
#define RATELPROF_NV_HOST_AGENT_HANDLE      ((uint64_t)0xFFFF0000u)
#define RATELPROF_NV_HOST_NODE              (255u)
/** Sub unit (stream) used for device-wide synchronizations. */
#define RATELPROF_NV_DEVICE_WIDE_STREAM     (0u)

/* ------------------------------------------------------------------------- */
/* GPU activities                                                             */
/* ------------------------------------------------------------------------- */

typedef enum {
    RATELPROF_NV_LAUNCH_REGULAR      = 0,
    RATELPROF_NV_LAUNCH_COOPERATIVE  = 1,
    RATELPROF_NV_LAUNCH_CLUSTER      = 2,
} ratelprof_nv_launch_type_t;

typedef struct ratelprof_nv_kernel_args_s {
    ratelprof_time_t queued;            /**< Time the launch was queued by the host (ns, 0 if unknown). */
    ratelprof_time_t submitted;         /**< Time the launch was submitted to the GPU (ns, 0 if unknown). */
    uint32_t grid[3];                   /**< Grid dimension (blocks). */
    uint32_t block[3];                  /**< Block dimension (threads). */
    uint32_t cluster[3];                /**< Thread block cluster dimension (sm_90+, 0 otherwise). */
    uint32_t registers_per_thread;
    uint32_t static_shared_memory;      /**< Bytes. */
    uint32_t dynamic_shared_memory;     /**< Bytes. */
    uint32_t local_memory_per_thread;   /**< Bytes. */
    uint64_t local_memory_total;        /**< Bytes. */
    uint64_t kernel_object;             /**< Unique kernel identifier (interned name). */
    uint64_t graph_node_id;             /**< CUDA graph node (0 if not launched from a graph). */
    uint8_t  launch_type;               /**< ratelprof_nv_launch_type_t */
    uint8_t  cache_config_requested;
    uint8_t  cache_config_executed;
    uint8_t  shared_memory_config;
} ratelprof_nv_kernel_args_t;

typedef struct ratelprof_nv_memory_args_s {
    ratelprof_memory_op_t memop;        /**< CUDA_MEMCPY / CUDA_MEMCPY_P2P / CUDA_MEMSET */
    uint64_t bytes;
    uint64_t other_handle;              /**< Agent at the other end of the copy. */
    uint8_t  src_type;                  /**< 0 host, 1 device */
    uint8_t  dst_type;                  /**< 0 host, 1 device */
    uint8_t  copy_kind;                 /**< CUpti_ActivityMemcpyKind */
    uint8_t  src_kind;                  /**< CUpti_ActivityMemoryKind */
    uint8_t  dst_kind;                  /**< CUpti_ActivityMemoryKind */
    uint8_t  is_async;
    uint32_t value;                     /**< Memset value */
    uint64_t graph_node_id;
} ratelprof_nv_memory_args_t;

typedef struct ratelprof_nv_sync_args_s {
    uint32_t type;                      /**< CUpti_ActivitySynchronizationType */
    uint32_t cuda_event_id;
} ratelprof_nv_sync_args_t;

typedef struct ratelprof_gpu_activity_s {
    ratelprof_domain_ext_t domain;      /**< Must be the first member (activity pool dispatch). */
    uint64_t id;
    uint64_t corr_id;                   /**< RATELProf ID of the API call that launched the work. */
    ratelprof_time_t start_time;        /**< CLOCK_MONOTONIC ns */
    ratelprof_time_t stop_time;         /**< CLOCK_MONOTONIC ns */
    uint64_t gpu_handle;                /**< RATELPROF_NV_GPU_AGENT_HANDLE(device_id) */
    uint32_t device_id;
    uint32_t context_id;
    uint32_t stream_id;
    uint32_t cupti_corr_id;             /**< CUPTI correlation ID */
    union {
        ratelprof_nv_kernel_args_t kernel;
        ratelprof_nv_memory_args_t memory;
        ratelprof_nv_sync_args_t   sync;
    } args;
} ratelprof_gpu_activity_t;


/* ------------------------------------------------------------------------- */
/* NVTX activities (stored in the ROCTX slot of the report)                   */
/* ------------------------------------------------------------------------- */

typedef struct ratelprof_roctx_activity_s {
    ratelprof_domain_ext_t domain;
	uint64_t tid;
	uint64_t pid;
    uint64_t id;
    uint64_t corr_id;
    bool has_children;
    ratelprof_clock_t start_time;
    ratelprof_clock_t stop_time;
    void * return_address;
    const char* message;
} ratelprof_roctx_activity_t;

typedef ratelprof_roctx_activity_t ratelprof_nvtx_activity_t;


/* ------------------------------------------------------------------------- */
/* "CUPTI for RATELProf" domain (profiler self-overhead)                       */
/* ------------------------------------------------------------------------- */

#define FOR_EACH_PROFILING_FUNC(macro) \
    macro(cuptiBufferCompleted)        \
    macro(cuptiActivityFlushAll)       \
    macro(cuptiSubscribe)              \
    macro(cuptiActivityEnable)

typedef enum {
#define RP_PROF_ID(f) PROFILING_ID_##f,
    FOR_EACH_PROFILING_FUNC(RP_PROF_ID)
#undef RP_PROF_ID
    PROFILING_ID_NB_FUNCTION,
    PROFILING_ID_UNKNOWN
} profiling_id_t;

static inline const char* get_profiling_funame_by_id(profiling_id_t id)
{
    switch (id) {
#define RP_PROF_NAME(f) case PROFILING_ID_##f: return "rprof_" #f;
        FOR_EACH_PROFILING_FUNC(RP_PROF_NAME)
#undef RP_PROF_NAME
        default: return NULL;
    }
}

#include "utils/utils.h"

const char* get_kernel_name(uint64_t kernel_object);
char* get_copy_name(uint32_t src_type, uint32_t dst_type);
ratelprof_status_t ratelprof_enable_node_id_tracking();

/** Intern a kernel name and return its unique kernel object identifier. */
uint64_t ratelprof_nv_intern_kernel_name(const char* name);

#endif // RATELPROF_EXT_NVIDIA_H
