#include <stdlib.h>

#include "utils/rprofrep_argument_manager.h"

#include "args_helper/ompt_args.h"
#ifdef RATELPROF_VENDOR_NVIDIA
#include "cuda_rt_args.h"
#include "cuda_drv_args.h"
#ifdef RATELPROF_HAVE_CUBLAS
#include "cublas_args.h"
#endif
#else
#include "args_helper/hsa_args.h"
#include "args_helper/hip_args.h"
#include "args_helper/rocblas_args.h"
#endif
#include "args_helper/omp_tgt_rtl_args.h"
#include "args_helper/omp_routine_args.h"
#include "args_helper/mpi_args.h"
#include "args_helper/rccl_args.h"

#include "rprofrep_encode_type.h"

#define SET_WRITER_AUX(name, ldomain, udomain) \
    arg_manager->__##ldomain##_api_arg_writer[udomain##_API_ID_##name].write_arg   = (rprofrep_write_arg_cb_t) write_##name##_args_to_buffer; \
    arg_manager->__##ldomain##_api_arg_writer[udomain##_API_ID_##name].write_param = (rprofrep_write_param_cb_t) write_##name##_params_to_buffer;


void rprofrep_init_argument_manager(rprofrep_argument_manager_t* arg_manager) 
{
#ifdef RATELPROF_VENDOR_NVIDIA
    // CUDA Runtime / Driver API argument writers (generated from the CUPTI headers)
    rprofrep_init_cuda_rt_arg_writers(arg_manager->__cuda_rt_api_arg_writer);
    rprofrep_init_cuda_drv_arg_writers(arg_manager->__cuda_drv_api_arg_writer);
#ifdef RATELPROF_HAVE_CUBLAS
    #define SET_WRITER(name) SET_WRITER_AUX(name, cublas, CUBLAS)
    FOR_EACH_CUBLAS_FUNC(SET_WRITER);
    #undef SET_WRITER
#endif
#else
    // Initialize HIP API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, hip, HIP)
    FOR_EACH_HIP_FUNC(SET_WRITER);
    #undef SET_WRITER

    // Initialize HSA API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, hsa, HSA)
    FOR_EACH_HSA_FUNC(SET_WRITER);
    #undef SET_WRITER
#endif

    // Initialize MPI API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, mpi, MPI)
    FOR_EACH_MPI_FUNC(SET_WRITER);
    #undef SET_WRITER

    // Initialize OMP TGT RTL API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, omp_tgt_rtl, OMP_TGT_RTL)
    FOR_EACH_OMP_TGT_RTL_FUNC(SET_WRITER);
    #undef SET_WRITER

    // Initialize OMP Routine API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, omp_routine, OMP_ROUTINE)
    FOR_EACH_OMP_ROUTINE_FUNC(SET_WRITER);
    #undef SET_WRITER
  
    // Initialize OMPT API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, ompt, OMPT)
    FOR_EACH_OMPT_FUNC(SET_WRITER);
    #undef SET_WRITER

#ifndef RATELPROF_VENDOR_NVIDIA
    // Initialize ROCBLAS API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, rocblas, ROCBLAS)
    FOR_EACH_ROCBLAS_FUNC(SET_WRITER);
    #undef SET_WRITER
#endif

    // Initialize RCCL API argument writers
    #define SET_WRITER(name) SET_WRITER_AUX(name, rccl, RCCL)
    FOR_EACH_RCCL_FUNC(SET_WRITER);
    #undef SET_WRITER

    // Map domain to corresponding argument writer arrays
    for (size_t i = 0; i < RATELPROF_NB_DOMAIN_EXT; i++) arg_manager->api_arg_writer[i] = NULL;
#ifdef RATELPROF_VENDOR_NVIDIA
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_CUDA_RT]     = arg_manager->__cuda_rt_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_CUDA_DRV]    = arg_manager->__cuda_drv_api_arg_writer;
#ifdef RATELPROF_HAVE_CUBLAS
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_CUBLAS]      = arg_manager->__cublas_api_arg_writer;
#endif
#else
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_HIP]         = arg_manager->__hip_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_HSA]         = arg_manager->__hsa_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_ROCBLAS]     = arg_manager->__rocblas_api_arg_writer;
#endif
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_MPI]         = arg_manager->__mpi_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_OMP_TGT_RTL] = arg_manager->__omp_tgt_rtl_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_OMP_ROUTINE] = arg_manager->__omp_routine_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_OMP_REGION]  = arg_manager->__ompt_api_arg_writer;
    arg_manager->api_arg_writer[RATELPROF_DOMAIN_RCCL]        = arg_manager->__rccl_api_arg_writer;
}



void rprofrep_write_args(
    rprofrep_encode_context_t* ctx,
    msgpack_buffer_t* buf,
    ratelprof_domain_t domain,
    ratelprof_api_id_t funid,
    const void* func_args
) {
    rprofrep_argument_manager_t* arg_manager = &ctx->arg_manager;
    rprofrep_write_arg_cb_t write_arg = arg_manager->api_arg_writer[domain][funid].write_arg;
    write_arg(func_args, ctx, buf);
}


void rprofrep_write_params(
    rprofrep_encode_context_t* ctx,
    msgpack_buffer_t* buf,
    ratelprof_domain_t domain,
    ratelprof_api_id_t funid
) {
    rprofrep_argument_manager_t* arg_manager = &ctx->arg_manager;
    rprofrep_write_param_cb_t write_param = arg_manager->api_arg_writer[domain][funid].write_param;
    write_param(ctx, buf);
}