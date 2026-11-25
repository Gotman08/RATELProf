#ifndef RATELPROF_CUPTI_H
#define RATELPROF_CUPTI_H

/*
 * Internal interface of the NVIDIA (CUPTI) backend.
 *
 *  ratelprof_cupti_init()          : called by ratelprof_init(); no CUDA call.
 *  ratelprof_cupti_enable_domain() : called for every domain requested by the user.
 *  ratelprof_cupti_start()         : called by ratelprof_start() once the activity
 *                                    pool exists; subscribes to CUPTI callbacks and
 *                                    enables the CUPTI activity kinds.
 *  ratelprof_cupti_stop()          : called by ratelprof_stop(); flushes the CUPTI
 *                                    activity buffers and stops recording.
 *  ratelprof_cupti_fini()          : called by ratelprof_fini().
 */

#include <stdbool.h>
#include <stdint.h>
#include <ratelprof.h>

ratelprof_status_t ratelprof_cupti_init(void);
ratelprof_status_t ratelprof_cupti_enable_domain(ratelprof_domain_t domain);
ratelprof_status_t ratelprof_cupti_start(void);
ratelprof_status_t ratelprof_cupti_stop(void);
ratelprof_status_t ratelprof_cupti_fini(void);
bool               ratelprof_cupti_is_domain_enabled(ratelprof_domain_t domain);

/* Call site of the user code that called the current CUDA API (best effort). */
void  ratelprof_nv_callsite_init(void);
void* ratelprof_nv_find_callsite(void);

/* Register the host and CUDA devices as RATELProf agents. */
void ratelprof_nv_register_agents(int ndevices);
void ratelprof_nv_register_device(uint32_t device_id);

#endif // RATELPROF_CUPTI_H
