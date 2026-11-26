# ---------------------------------------------------------------------------- #
#  RATELProf NVIDIA backend (CUPTI)                                            #
#                                                                              #
#  Inputs (cache variables):                                                   #
#    RATELPROF_CUDA_ROOT   CUDA toolkit root (default: $CUDA_HOME, $CUDA_PATH,  #
#                          or the parent of the 'nvcc' found in PATH)          #
#                                                                              #
#  Outputs:                                                                    #
#    RATELPROF_CUPTI_INCLUDE_DIR, RATELPROF_CUPTI_LIBRARY,                     #
#    RATELPROF_CUDA_INCLUDE_DIR, RATELPROF_NV_GEN_DIR,                         #
#    RATELPROF_NV_INCLUDE_DIRS, RATELPROF_HAVE_CUBLAS                          #
# ---------------------------------------------------------------------------- #

SET(RATELPROF_CUDA_ROOT "" CACHE PATH "CUDA toolkit root directory (NVIDIA backend)")

IF(NOT RATELPROF_CUDA_ROOT)
    IF(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}")
        SET(RATELPROF_CUDA_ROOT "$ENV{CUDA_HOME}")
    ELSEIF(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
        SET(RATELPROF_CUDA_ROOT "$ENV{CUDA_PATH}")
    ELSE()
        FIND_PROGRAM(RATELPROF_NVCC nvcc)
        IF(RATELPROF_NVCC)
            GET_FILENAME_COMPONENT(_nvcc_real "${RATELPROF_NVCC}" REALPATH)
            GET_FILENAME_COMPONENT(_nvcc_bin "${_nvcc_real}" DIRECTORY)
            GET_FILENAME_COMPONENT(RATELPROF_CUDA_ROOT "${_nvcc_bin}" DIRECTORY)
        ELSEIF(EXISTS "/usr/local/cuda")
            SET(RATELPROF_CUDA_ROOT "/usr/local/cuda")
        ENDIF()
    ENDIF()
ENDIF()

IF(NOT RATELPROF_CUDA_ROOT)
    MESSAGE(FATAL_ERROR "NVIDIA backend: CUDA toolkit not found. Set -DRATELPROF_CUDA_ROOT=<cuda> or CUDA_HOME.")
ENDIF()
MESSAGE(STATUS "NVIDIA backend: CUDA toolkit ${RATELPROF_CUDA_ROOT}")

FIND_PATH(RATELPROF_CUDA_INCLUDE_DIR cuda.h
    HINTS "${RATELPROF_CUDA_ROOT}/include" "${RATELPROF_CUDA_ROOT}/targets/x86_64-linux/include"
          "${RATELPROF_CUDA_ROOT}/targets/sbsa-linux/include" "${RATELPROF_CUDA_ROOT}/targets/aarch64-linux/include"
    NO_DEFAULT_PATH)
FIND_PATH(RATELPROF_CUPTI_INCLUDE_DIR cupti.h
    HINTS "${RATELPROF_CUDA_ROOT}/extras/CUPTI/include" "${RATELPROF_CUDA_ROOT}/include"
          "${RATELPROF_CUDA_ROOT}/targets/x86_64-linux/include" "${RATELPROF_CUDA_ROOT}/targets/sbsa-linux/include"
    NO_DEFAULT_PATH)
FIND_LIBRARY(RATELPROF_CUPTI_LIBRARY cupti
    HINTS "${RATELPROF_CUDA_ROOT}/extras/CUPTI/lib64" "${RATELPROF_CUDA_ROOT}/extras/CUPTI/lib"
          "${RATELPROF_CUDA_ROOT}/lib64" "${RATELPROF_CUDA_ROOT}/targets/x86_64-linux/lib"
          "${RATELPROF_CUDA_ROOT}/targets/sbsa-linux/lib"
    NO_DEFAULT_PATH)

IF(NOT RATELPROF_CUDA_INCLUDE_DIR OR NOT RATELPROF_CUPTI_INCLUDE_DIR OR NOT RATELPROF_CUPTI_LIBRARY)
    MESSAGE(FATAL_ERROR "NVIDIA backend: cuda.h / cupti.h / libcupti not found under ${RATELPROF_CUDA_ROOT}")
ENDIF()
GET_FILENAME_COMPONENT(RATELPROF_CUPTI_LIBRARY_DIR "${RATELPROF_CUPTI_LIBRARY}" DIRECTORY)
FILE(STRINGS "${RATELPROF_CUPTI_INCLUDE_DIR}/cupti_version.h" _cupti_ver REGEX "^#define CUPTI_API_VERSION [0-9]+")
STRING(REGEX REPLACE "^#define CUPTI_API_VERSION ([0-9]+).*" "\\1" RATELPROF_CUPTI_API_VERSION "${_cupti_ver}")
MESSAGE(STATUS "NVIDIA backend: CUPTI ${RATELPROF_CUPTI_LIBRARY}")

# ---- Code generation from the CUPTI headers ---------------------------------
FIND_PACKAGE(Python3 COMPONENTS Interpreter REQUIRED)
SET(RATELPROF_NV_GEN_DIR "${CMAKE_BINARY_DIR}/generated/nvidia")
EXECUTE_PROCESS(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_cupti_domains.py"
            --cupti-include "${RATELPROF_CUPTI_INCLUDE_DIR}" --out "${RATELPROF_NV_GEN_DIR}"
    RESULT_VARIABLE _gen_res
    OUTPUT_VARIABLE _gen_out
    ERROR_VARIABLE  _gen_err)
IF(NOT _gen_res EQUAL 0)
    MESSAGE(FATAL_ERROR "NVIDIA backend: CUPTI code generation failed:\n${_gen_out}\n${_gen_err}")
ENDIF()
MESSAGE(STATUS "${_gen_out}")
SET_PROPERTY(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_cupti_domains.py"
    "${RATELPROF_CUPTI_INCLUDE_DIR}/cupti_runtime_cbid.h"
    "${RATELPROF_CUPTI_INCLUDE_DIR}/cupti_driver_cbid.h")

# ---- cuBLAS domain (optional, generated from cublas_api.h) -------------------
OPTION(RATELPROF_BUILD_CUBLAS "Build the cuBLAS tracing domain (NVIDIA backend)" ON)
SET(RATELPROF_HAVE_CUBLAS OFF)
IF(RATELPROF_BUILD_CUBLAS AND EXISTS "${RATELPROF_CUDA_INCLUDE_DIR}/cublas_api.h"
   AND EXISTS "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_cublas_domain.py")
    EXECUTE_PROCESS(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_cublas_domain.py"
                --cuda-include "${RATELPROF_CUDA_INCLUDE_DIR}" --out "${RATELPROF_NV_GEN_DIR}"
        RESULT_VARIABLE _cublas_res
        OUTPUT_VARIABLE _cublas_out
        ERROR_VARIABLE  _cublas_err)
    IF(_cublas_res EQUAL 0)
        SET(RATELPROF_HAVE_CUBLAS ON)
        MESSAGE(STATUS "${_cublas_out}")
        ADD_COMPILE_DEFINITIONS(RATELPROF_HAVE_CUBLAS)
    ELSE()
        MESSAGE(WARNING "NVIDIA backend: cuBLAS domain generation failed, cuBLAS tracing disabled:\n${_cublas_err}")
    ENDIF()
ENDIF()

# ---- libomptarget >= 19 plugin method adapters (omp-target domain) ----------
EXECUTE_PROCESS(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_omp_tgt_methods.py"
            --wrapper "${CMAKE_SOURCE_DIR}/src/wrappers/omp_tgt_rtl_wrapper.c" --out "${RATELPROF_NV_GEN_DIR}"
    RESULT_VARIABLE _omp_res
    OUTPUT_VARIABLE _omp_out
    ERROR_VARIABLE  _omp_err)
IF(NOT _omp_res EQUAL 0)
    MESSAGE(FATAL_ERROR "NVIDIA backend: omp-target method adapters generation failed:
${_omp_out}
${_omp_err}")
ENDIF()
MESSAGE(STATUS "${_omp_out}")
SET_PROPERTY(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/nvidia/gen/gen_omp_tgt_methods.py"
    "${CMAKE_SOURCE_DIR}/src/wrappers/omp_tgt_rtl_wrapper.c")

SET(RATELPROF_NV_INCLUDE_DIRS
    "${RATELPROF_NV_GEN_DIR}"
    "${RATELPROF_CUPTI_INCLUDE_DIR}"
    "${RATELPROF_CUDA_INCLUDE_DIR}")

ADD_COMPILE_DEFINITIONS(RATELPROF_VENDOR_NVIDIA)
INCLUDE_DIRECTORIES(${RATELPROF_NV_INCLUDE_DIRS})

# CUPTI is found at run time through the RUNPATH of the RATELProf libraries.
SET(CMAKE_INSTALL_RPATH "${RATELPROF_CUPTI_LIBRARY_DIR}")
SET(CMAKE_BUILD_RPATH   "${RATELPROF_CUPTI_LIBRARY_DIR}")
