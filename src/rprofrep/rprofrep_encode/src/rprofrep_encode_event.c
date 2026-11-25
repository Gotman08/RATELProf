#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "rprofrep_core.h"

#include "utils/rprofrep_msgpack_ext.h"

#include "rprofrep_event_pool_buffer.h"
#include "rprofrep_encode_event.h"
#include "rprofrep_encode_type.h"

#ifdef RATELPROF_VENDOR_NVIDIA
/* ======================================================================== */
/* NVIDIA (CUPTI) GPU events                                                 */
/*                                                                          */
/* Layout of the fixed part is identical to the AMD backend (id, start, dur, */
/* cid, extra) so that the decoder and the visualizer stay vendor neutral.   */
/* The argument part carries the CUPTI information (see the label tables in  */
/* rprofrep_decode_event.c).                                                 */
/* ======================================================================== */

static inline uint64_t nv_norm(ratelprof_time_t t)
{
    return t ? ratelprof_get_normalized_monotonic_time(t) : 0;
}

/* CUpti_ActivitySynchronizationType values. */
static const char* nv_sync_name(uint32_t type)
{
    switch (type) {
        case 1:  return "Event Synchronize";
        case 2:  return "Stream Wait Event";
        case 3:  return "Stream Synchronize";
        case 4:  return "Context Synchronize";
        default: return "Synchronization";
    }
}

static void add_memory_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain,
    const ratelprof_gpu_activity_t* activity
) {
    const ratelprof_nv_memory_args_t* m = &activity->args.memory;
    ratelprof_time_t dur = activity->stop_time - activity->start_time;
    const char* event_name = NULL;

    if (m->memop == RATELPROF_MEMORY_OP_CUDA_MEMSET) {
        event_name = "Memset";
    } else if (m->memop == RATELPROF_MEMORY_OP_CUDA_MEMCPY_P2P) {
        event_name = "CopyPeerToPeer";
    } else {
        event_name = get_copy_name(m->src_type, m->dst_type);
    }

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, activity->gpu_handle, activity->stream_id);
    msgpack_buffer_t* evt = &entry->buffer;
    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);
    msgpack_encode_uint(evt, nv_norm(activity->start_time));
    msgpack_encode_uint(evt, dur);
    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);
    rprofrep_msgpack_ext_encode_string(ctx, evt, event_name);
    msgpack_encode_uint(evt, m->memop);

    if (m->memop == RATELPROF_MEMORY_OP_CUDA_MEMSET) {
        msgpack_encode_uint(evt, activity->cupti_corr_id);
        msgpack_encode_uint(evt, m->bytes);
        msgpack_encode_uint(evt, m->value);
        msgpack_encode_uint(evt, m->dst_kind);
        msgpack_encode_uint(evt, m->is_async);
        msgpack_encode_uint(evt, activity->context_id);
        msgpack_encode_uint(evt, m->graph_node_id);
    } else {
        msgpack_encode_uint(evt, activity->cupti_corr_id);
        msgpack_encode_uint(evt, m->bytes);
        msgpack_encode_uint(evt, m->other_handle);
        msgpack_encode_uint(evt, m->copy_kind);
        msgpack_encode_uint(evt, m->src_kind);
        msgpack_encode_uint(evt, m->dst_kind);
        msgpack_encode_uint(evt, m->is_async);
        msgpack_encode_uint(evt, activity->context_id);
        msgpack_encode_uint(evt, m->graph_node_id);
    }

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}


static void add_barrier_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain,
    const ratelprof_gpu_activity_t* activity
) {
    ratelprof_time_t dur = activity->stop_time - activity->start_time;

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, activity->gpu_handle, activity->stream_id);
    msgpack_buffer_t* evt = &entry->buffer;
    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);
    msgpack_encode_uint(evt, nv_norm(activity->start_time));
    msgpack_encode_uint(evt, dur);
    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);
    /* Name of the synchronization kind (decoded as the event name / ufunid). */
    rprofrep_msgpack_ext_encode_string(ctx, evt, nv_sync_name(activity->args.sync.type));

    msgpack_encode_uint(evt, activity->cupti_corr_id);
    msgpack_encode_uint(evt, nv_norm(activity->start_time));
    msgpack_encode_uint(evt, activity->args.sync.type);
    msgpack_encode_uint(evt, activity->args.sync.cuda_event_id);
    msgpack_encode_uint(evt, activity->context_id);

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}


static void add_kernel_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain,
    const ratelprof_gpu_activity_t* activity
) {
    const ratelprof_nv_kernel_args_t* k = &activity->args.kernel;
    ratelprof_time_t dur = activity->stop_time - activity->start_time;

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, activity->gpu_handle, activity->stream_id);
    msgpack_buffer_t* evt = &entry->buffer;
    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);
    msgpack_encode_uint(evt, nv_norm(activity->start_time));
    msgpack_encode_uint(evt, dur);
    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    /* Static kernel properties: private segment = local memory per thread,
     * group segment = static shared memory. */
    rprofrep_msgpack_ext_encode_kernel(ctx, evt,
        k->local_memory_per_thread,
        k->static_shared_memory,
        k->kernel_object
    );

    /* Dispatch time: time the launch was queued by the host (falls back on the
     * submission time, then on the start time). */
    ratelprof_time_t dispatch = k->queued ? k->queued : (k->submitted ? k->submitted : activity->start_time);

    msgpack_encode_uint(evt, activity->cupti_corr_id);
    msgpack_encode_uint(evt, nv_norm(dispatch));

    msgpack_encode_array(evt, 3);
    msgpack_encode_uint(evt, k->block[0]);
    msgpack_encode_uint(evt, k->block[1]);
    msgpack_encode_uint(evt, k->block[2]);

    msgpack_encode_array(evt, 3);
    msgpack_encode_uint(evt, k->grid[0]);
    msgpack_encode_uint(evt, k->grid[1]);
    msgpack_encode_uint(evt, k->grid[2]);

    msgpack_encode_uint(evt, nv_norm(k->submitted));
    msgpack_encode_uint(evt, k->registers_per_thread);
    msgpack_encode_uint(evt, k->static_shared_memory);
    msgpack_encode_uint(evt, k->dynamic_shared_memory);
    msgpack_encode_uint(evt, k->local_memory_per_thread);
    msgpack_encode_uint(evt, k->local_memory_total);

    msgpack_encode_array(evt, 3);
    msgpack_encode_uint(evt, k->cluster[0]);
    msgpack_encode_uint(evt, k->cluster[1]);
    msgpack_encode_uint(evt, k->cluster[2]);

    msgpack_encode_uint(evt, k->launch_type);
    msgpack_encode_uint(evt, activity->context_id);
    msgpack_encode_uint(evt, k->graph_node_id);
    msgpack_encode_uint(evt, k->cache_config_requested);
    msgpack_encode_uint(evt, k->cache_config_executed);
    msgpack_encode_uint(evt, k->shared_memory_config);

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}



#else /* AMD */
static uint64_t engine_id_from_mask(unsigned int mask) {
    if (mask == 0 || (mask & (mask - 1)) != 0) {
        return -1;
    }
    return -(__builtin_ctz(mask) + 2);
}


static void add_memory_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_gpu_activity_t* activity
) {
    uint64_t gpu_handle = 0;
    uint64_t other_handle = 0;
    char* event_name = NULL;
    ratelprof_time_t dur = activity->stop_time - activity->start_time;

    ratelprof_memory_op_t memop = activity->args.memory.memop;
    int64_t sub_unit = 0;
    if (memop == RATELPROF_MEMORY_OP_SDMA_COPY) {
        sub_unit = engine_id_from_mask(activity->args.memory.engine_id);
    } else {
        sub_unit = activity->args.memory.blit.queue_id;
    }

    if (is_blit_copy_kernel(memop) || memop == RATELPROF_MEMORY_OP_SDMA_COPY) {
        uint64_t src_agent_handle = activity->args.memory.copy.src_agent.handle;
        uint64_t dst_agent_handle = activity->args.memory.copy.dst_agent.handle;

        ratelprof_agent_object_t* src_agent_obj = NULL;
        ratelprof_agent_object_t* dst_agent_obj = NULL;

        ratelprof_object_tracking_pool_get_agent_from_handle(src_agent_handle, (void*)&src_agent_obj);
        ratelprof_object_tracking_pool_get_agent_from_handle(dst_agent_handle, (void*)&dst_agent_obj);

        uint8_t src_type = src_agent_obj ? src_agent_obj->type : 0;
        uint8_t dst_type = dst_agent_obj ? dst_agent_obj->type : 0;
        event_name = get_copy_name(src_type, dst_type);
        if (src_type == 1) {
            gpu_handle = src_agent_handle;
            other_handle = dst_agent_handle;
        } else {
            gpu_handle = dst_agent_handle;
            other_handle = src_agent_handle;
        }
    } else {
        gpu_handle = activity->args.memory.blit.agent.handle;
        event_name = "Memset";
    }

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, gpu_handle, sub_unit);

    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_monotonic_time(activity->start_time));
    
    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    rprofrep_msgpack_ext_encode_string(ctx, evt, event_name);

    msgpack_encode_uint(evt, memop);

    msgpack_encode_uint(evt, activity->completion_signal.handle);

    if (is_blit_copy_kernel(memop) || memop == RATELPROF_MEMORY_OP_SDMA_COPY) {
        msgpack_encode_uint(evt, activity->args.memory.copy.size);
        msgpack_encode_uint(evt, other_handle);
    } else if (is_blit_set_kernel(memop)) {
        msgpack_encode_uint(evt, activity->args.memory.fill.size);
        // msgpack_encode_uint(evt, activity->args.memory.fill.pattern_size);
        // msgpack_encode_uint(evt, activity->args.memory.fill.pattern); // TODO: encode pattern data in the future if needed
    }
    if (memop != RATELPROF_MEMORY_OP_SDMA_COPY) {
        ratelprof_time_t dispatch_time = ratelprof_get_time_ns(activity->args.memory.blit.dispatch_time);
        msgpack_encode_uint(evt, ratelprof_get_normalized_time(dispatch_time));
        msgpack_encode_uint(evt, activity->args.memory.blit.kernel.workgroup_size_x);
    }

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}


static void add_barrier_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_gpu_activity_t* activity
) {
    ratelprof_time_t dispatch_time = ratelprof_get_time_ns(activity->args.dispatch.dispatch_time);
    ratelprof_time_t dur = activity->stop_time - activity->start_time;

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, activity->args.dispatch.agent.handle, activity->args.dispatch.queue_id);
    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);
    
    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_monotonic_time(activity->start_time));

    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    msgpack_encode_uint(evt, activity->completion_signal.handle);

    msgpack_encode_uint(evt, ratelprof_get_normalized_time(dispatch_time));

    msgpack_encode_array(evt, 5);
    for (int i = 0; i < 5; i++)
    {
        msgpack_encode_uint(evt, activity->args.dispatch.barrier.dep_signal[i].handle);
    }

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}


static void add_kernel_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_gpu_activity_t* activity
) {
    ratelprof_time_t dispatch_time = ratelprof_get_time_ns(activity->args.dispatch.dispatch_time);
    ratelprof_time_t dur = activity->stop_time - activity->start_time;

    rprofrep_buffer_entry_t* entry = rprofrep_get_gpu_event_buffer(ctx, domain, activity->args.dispatch.agent.handle, activity->args.dispatch.queue_id);
    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_monotonic_time(activity->start_time));

    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    rprofrep_msgpack_ext_encode_kernel(ctx, evt,
        activity->args.dispatch.kernel.private_segment_size,
        activity->args.dispatch.kernel.group_segment_size,
        activity->args.dispatch.kernel.kernel_object
    );

    msgpack_encode_uint(evt, activity->completion_signal.handle);

    msgpack_encode_uint(evt, ratelprof_get_normalized_time(dispatch_time));

    const uint16_t wgx = activity->args.dispatch.kernel.workgroup_size_x;
    const uint16_t wgy = activity->args.dispatch.kernel.workgroup_size_y;
    const uint16_t wgz = activity->args.dispatch.kernel.workgroup_size_z;

    msgpack_encode_array(evt, 3);
    msgpack_encode_uint(evt, wgx);
    msgpack_encode_uint(evt, wgy);
    msgpack_encode_uint(evt, wgz);

    msgpack_encode_array(evt, 3);
    msgpack_encode_uint(evt, activity->args.dispatch.kernel.grid_size_x/wgx);
    msgpack_encode_uint(evt, activity->args.dispatch.kernel.grid_size_y/wgy);
    msgpack_encode_uint(evt, activity->args.dispatch.kernel.grid_size_z/wgz);

    msgpack_encode_uint(evt, (uintptr_t)activity->args.dispatch.kernel.kernarg_address);


    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, false);
}



#endif /* RATELPROF_VENDOR_NVIDIA */

static void add_api_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_api_activity_t* activity
) {
    ratelprof_time_t start = ratelprof_get_time_ns(activity->start_time);
    ratelprof_time_t stop  = ratelprof_get_time_ns(activity->stop_time);
    ratelprof_time_t dur   = stop - start;

    rprofrep_buffer_entry_t* entry = rprofrep_get_cpu_event_buffer(ctx, domain, activity->pid, activity->tid);
    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_time(start));

    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    rprofrep_msgpack_ext_encode_api_data(ctx, evt, domain, activity->funid);

    rprofrep_msgpack_ext_encode_loc(ctx, evt, activity->return_address);

    rprofrep_msgpack_ext_encode_args(ctx, evt, domain, activity->funid, activity->args);

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, activity->has_children);
}


static void add_roctx_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_roctx_activity_t* activity
) {
    ratelprof_time_t start = ratelprof_get_time_ns(activity->start_time);
    ratelprof_time_t stop  = ratelprof_get_time_ns(activity->stop_time);
    ratelprof_time_t dur   = stop - start;

    rprofrep_buffer_entry_t* entry = rprofrep_get_cpu_event_buffer(ctx, domain, activity->pid, activity->tid);
    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_time(start));

    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    rprofrep_msgpack_ext_encode_string(ctx, evt, activity->message);

    rprofrep_msgpack_ext_encode_loc(ctx, evt, activity->return_address);

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, activity->has_children);
}


static void add_profiling_activity_data_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain, 
    const ratelprof_api_activity_t* activity
) {
    ratelprof_time_t start = ratelprof_get_time_ns(activity->start_time);
    ratelprof_time_t stop  = ratelprof_get_time_ns(activity->stop_time);
    ratelprof_time_t dur   = stop - start;

    rprofrep_buffer_entry_t* entry = rprofrep_get_cpu_event_buffer(ctx, domain, activity->pid, activity->tid);
    msgpack_buffer_t* evt = &entry->buffer;

    size_t old_size = msgpack_size(evt);

    msgpack_encode_uint(evt, activity->id);

    msgpack_encode_uint(evt, ratelprof_get_normalized_time(start));

    msgpack_encode_uint(evt, dur);

    rprofrep_msgpack_ext_encode_cid(evt, activity->corr_id);

    rprofrep_msgpack_ext_encode_string(ctx, evt, ratelprof_get_funame_by_id(domain, activity->funid));

    rprofrep_msgpack_ext_encode_loc(ctx, evt, activity->return_address);

    rprofrep_msgpack_ext_encode_event(ctx, entry, msgpack_size(evt) - old_size, activity->id, activity->has_children);
}


typedef void (*encode_activity_data_to_buffer_cb_t)(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain,
    const void* activity
);

static encode_activity_data_to_buffer_cb_t encode_activity_data_to_buffer_cb[RATELPROF_NB_DOMAIN_EXT] = {
    [RATELPROF_DOMAIN_HSA]          = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_OMP_TGT_RTL]  = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_OMP_ROUTINE]  = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_HIP]          = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_MPI]          = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_OMP_REGION]   = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_RCCL]         = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_ROCBLAS]      = (encode_activity_data_to_buffer_cb_t) add_api_activity_data_to_buffer,
    [RATELPROF_DOMAIN_ROCTX]        = (encode_activity_data_to_buffer_cb_t) add_roctx_activity_data_to_buffer,
    [RATELPROF_DOMAIN_PROFILING]    = (encode_activity_data_to_buffer_cb_t) add_profiling_activity_data_to_buffer,
    [RATELPROF_DOMAIN_MEMORY]       = (encode_activity_data_to_buffer_cb_t) add_memory_activity_data_to_buffer,
    [RATELPROF_DOMAIN_KERNEL]       = (encode_activity_data_to_buffer_cb_t) add_kernel_activity_data_to_buffer,
    [RATELPROF_DOMAIN_BARRIERAND]   = (encode_activity_data_to_buffer_cb_t) add_barrier_activity_data_to_buffer,
    [RATELPROF_DOMAIN_BARRIEROR]    = (encode_activity_data_to_buffer_cb_t) add_barrier_activity_data_to_buffer,
};

void rprofrep_encode_activity_to_buffer(
    rprofrep_encode_context_t* ctx,
    ratelprof_domain_t domain,
    const void* activity
) {
    encode_activity_data_to_buffer_cb[domain](ctx, domain, activity);
}