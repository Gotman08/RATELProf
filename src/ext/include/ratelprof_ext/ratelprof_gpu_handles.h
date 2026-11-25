#ifndef RATELPROF_GPU_HANDLES_H
#define RATELPROF_GPU_HANDLES_H

/*
 * Vendor neutral handle types.
 *
 * The AMD backend uses the HSA types hsa_agent_t / hsa_signal_t (opaque
 * structures holding a 64-bit handle) in the shared parts of RATELProf (object
 * tracking, report encoder). The NVIDIA backend has no HSA runtime: it provides
 * structurally identical definitions so that the shared code compiles
 * unchanged.  For NVIDIA, an "agent" is either the host CPU or a CUDA device
 * (see ratelprof_cupti.h for the handle values).
 */

#include <stdint.h>

#ifdef RATELPROF_VENDOR_NVIDIA
typedef struct hsa_agent_s  { uint64_t handle; } hsa_agent_t;
typedef struct hsa_signal_s { uint64_t handle; } hsa_signal_t;
typedef int64_t hsa_signal_value_t;
#endif

#endif // RATELPROF_GPU_HANDLES_H
