# RATELProf

RATELProf is a powerful and lightweight profiler designed specifically for AMD GPU applications. It enables detailed profiling by tracing GPU activities, and runtime API calls from HSA, HIP, and OpenMP. This tool is ideal for developers and researchers aiming to optimize GPU-accelerated applications.

##  Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [NVIDIA GPUs (CUPTI backend)](#nvidia-gpus-cupti-backend)
- [Comparison with ROCprof v3](#comparison-with-rocprof-v3)
- [Future Work and Improvement](#future-work-and-improvement)
- [Contribution](#contribution)

---

##  Features

RATELProf provides a comprehensive suite of tools to profile, analyze, and optimize AMD GPU applications. Key features include:

#### **GPU Activity Profiling**
- Monitor kernel dispatches, barriers and memory transfers to identify bottlenecks and optimize GPU performance.

#### **Runtime API Tracing**
- Trace calls from HSA, HIP, and OpenMP runtimes, offering deep insights into application behavior.

#### **Commands**
RATELProf includes four core commands to streamline your profiling workflow:

1. **`profile`**  
   - Profiles the application and generates a detailed .rprof-rep report.  
   - This command captures kernel execution, memory transfers, and runtime API details.

2. **`stats`**  
   - Computes statistical summaries from a .rprof-rep report created by the `profile` command.  
   - Output includes detailed metrics similar to the `stat` command from NVIDIA's Nsight Systems.

3. **`analyze`**  
   - Analyze profiling results from a .rprof-rep report created by the `profile` command.  
   - Output advices to optimize your CPU/GPU code.

4. **`breakdown`**  
   - Give time breakdown information for each PID or GPU.  

5. **`summarize`**  
   - Run `stats`, `analyze` and `breakdown` command to give in terminal view a summary of your profiling report.  

6. **`visualize`**  
   - Generates an interactive HTML timeline report for the .rprof-rep profile report created by the `profile` command.  
   - Perfect for visualizing application details.

7. **`inspect`**  
   - Inspects the application binary and outputs a CSV/JSON report containing detailed kernel information.  
   - Use this command to analyze static kernel properties.

8. **`export`**  
   - Export the .rprof-rep report to another type of report (json, arg-info, ...).

---

## **Getting Started**

Installing RATELProf is simple and requires running the provided `install.sh` script.

### Prerequisites

Ensure you have the following installed on your system before proceeding:
- **CMake** (version 3.13 or later)
- **Lua** (version 5.1)
- **LuaJIT** (optional)

### Installation

1. Clone the repository:

```bash
git clone --recurse-submodules https://github.com/Kassouley/RATELProf.git
cd RATELProf
```

2. Run the installation script:

```bash
lua sett_install.lua
```
By default, the installation script will use the sett.config config file.
Modify this file to custom your installation.

```bash
lua set_install.lua <sett.config>
```

3. Set ENV

If needed, set up your environment variable:
```bash
export PATH=<path/to/ratelprof/bin>:$PATH
export PATH=<path/to/lua5.1/bin>:$PATH
```

##  NVIDIA GPUs (CUPTI backend)

This fork adds an NVIDIA backend built on **CUPTI** (callback + activity APIs). The same
binary report format, Lua front-end and RATELProf-Visualize web app are used, so every
command (`profile`, `stats`, `analyze`, `breakdown`, `summarize`, `visualize`, `inspect`,
`export`) works on NVIDIA reports.

### Build

Requirements: a CUDA toolkit with CUPTI (`extras/CUPTI`), CMake ≥ 3.13, Python 3 (build-time
code generators), Lua 5.1 (with `dlopen` support).

```bash
git clone --recurse-submodules -b nvidia-port https://github.com/Gotman08/RATELProf.git
cd RATELProf
lua sett_install.lua sett.nvidia.config        # or:
cmake -B build -DRATELPROF_GPU_VENDOR=NVIDIA -DRATELPROF_CUDA_ROOT=$CUDA_HOME \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local/ratelprof
cmake --build build -j && cmake --install build
```

The CUDA runtime/driver callback IDs, their parameter structs, the CUPTI activity record
versions and the cuBLAS API are **generated at configure time from the headers of the CUDA
toolkit in use** (`src/nvidia/gen/`). Nothing is hard-coded to a CUDA version: the backend
was built and tested with CUDA 12.6.3 and 13.1.1.

### Traces

| `--trace`          | NVIDIA domain                                   | AMD equivalent  |
|--------------------|-------------------------------------------------|-----------------|
| `kernel`           | Kernel executions (CUPTI concurrent kernel)     | kernel          |
| `memory`           | Memcpy / P2P memcpy / memset                    | memory          |
| `sync`, `barrier`  | Stream wait event, host synchronizations        | barrier         |
| `cuda`             | CUDA Runtime API (with arguments, return value) | hip             |
| `cuda-driver`      | CUDA Driver API                                 | hsa             |
| `nvtx`             | NVTX push/pop, start/end, marks, domains, registered strings | roctx |
| `cublas`           | cuBLAS (interposed, all `cublas*` functions)    | rocblas         |
| `nccl`             | NCCL                                            | rccl            |
| `mpi`              | MPI                                             | mpi             |
| `omp-region`, `omp-routine`, `omp-target` | OMPT regions, OpenMP device routines, libomptarget RTL | same |
| `cupti-for-rprof`  | Profiler self-overhead (CUPTI buffers, flushes) | hsa-for-rprof   |

GPU events are correlated to the API call that launched them (CUPTI correlation IDs), and
each API call keeps its call site in the application (also through a statically linked
`cudart`). Kernel events carry queued/submitted/start/end timestamps, grid/block/cluster
sizes, registers, static/dynamic shared memory, local memory, launch type and cache config.

Environment variables: `RATELPROF_CUPTI_BUFFER_SIZE` (activity buffer size, default 8 MiB),
`RATELPROF_CUDA_CALLSITE=0` (disable call-site resolution), `RATELPROF_LLVM_LIB` (directory of
`libomptarget.rtl.cuda.so` for `omp-target`).

### Inspect

`ratelprof inspect` reads the fatbinaries embedded in executables and shared libraries
(`.nv_fatbin`, `__nv_relfatbin`, standalone `.fatbin`/`.cubin`, LZ4 and ZSTD compressed entries)
and parses each CUDA ELF: registers, shared/constant/local memory, `.nv.info` EIATTR attributes
(max threads, required block size, cluster dims, spills, ...), theoretical occupancy per
architecture (sm_50 to sm_120). With `--sass`, kernels are disassembled with `nvdisasm`
(or `cuobjdump -sass`) and an instruction-mix analysis is added (FP32/FP64/FP16, tensor-core
MMA, TMA, global/shared/local/constant memory, atomics, texture, barriers, control flow,
predication) with findings such as local-memory spills or FP64 usage; `--sass-dump` keeps the
listings.

### Tested

On the ROMEO supercomputer (NVIDIA GH200, Hopper sm_90, aarch64): CUDA 12.6.3 and 13.1.1,
multi-stream/event app, cuBLAS + NCCL + NVTX on 2 GPUs, MPI + CUDA (OpenMPI, 2 ranks),
OpenMP offload with LLVM 16 and 19. `inspect` was validated on 141 kernel images from sm_50 to
sm_120 (compared against `cuobjdump -res-usage` / `cuobjdump -elf`). CUPTI itself supports
Maxwell and later GPUs; profiling has only been run on Hopper so far.

### Known limitations

- No hardware counters / PC sampling yet (CUPTI Range Profiler / PC Sampling APIs are not wired).
- `omp-target` intercepts the `__tgt_rtl_*` plugin interface that libomptarget `dlopen`s. Upstream
  LLVM ≥ 19 links its plugins statically and no longer has that interface; LLVM 16 resolves it in
  a constructor that runs before RATELProf's. Use `omp-region` + `cuda-driver` there.
  OMPT target-region callbacks require LLVM ≥ 17.
- Ranges a library pushes but never pops (NCCL's `ncclCommInitAll` range) are not reported.

### Fixes in shared code

The port also fixes issues that affect the AMD build: NCCL/RCCL `stream` arguments truncated to
`int`, uninitialized agent count in object tracking, interposition wrappers called before their
API table is populated (crash with OMPT under LLVM 16), `ratelprof.fs.concat_path` not defined in
`inspect`, aarch64 timer (`cntvct_el0`) and glibc `dlsym` symbol versions.

##  Comparison with ROCprof v3

| Attribute                         | ROCprof v3                                      | RATELProf                                    |
|-----------------------------------|-------------------------------------------------|----------------------------------------------|
| **GPU Architecture Support**      | AMD RDNA, CDNA (ROCm-compatible GPUs)           | AMD RDNA, CDNA (ROCm-compatible GPUs)        |
| **HIP Tracing**                   | ✅                                             | ✅                                           |
| **HSA Tracing**                   | ✅                                             | ✅                                           |
| **rocBLAS Tracing**               | ❌                                             | ✅                                           |
| **RCCL Tracing**                  | ✅                                             | ✅                                           |
| **Marker Tracing**                | ✅ (ROCTx)                                     | ✅ (ROCTx)                                   |
| **OpenMP Routine Tracing**        | ❌                                             | ✅                                           |
| **OpenMP Target RTL Tracing**     | ❌                                             | ✅                                           |
| **OMPT Integration**              | ✅                                             | ✅                                           |
| **Scratch Memory Tracing**        | ✅                                             | ❌                                           |
| **Memory transfers Profiling**    | ✅                                             | ✅                                           |
| **Kernel dispatch Profiling**     | ✅                                             | ✅                                           |
| **Barrier dispatch Profiling**    | ❌                                             | ✅                                           |
| **Trace filtering**               | ❌                                             | ✅                                           |
| **PC Sampling**                   | ✅ (Beta)                                      | ❌                                           |
| **HW Counter**                    | ✅                                             | ❌ (WIP)                                     |
| **Statistical post processing**   | ✅ but really simple post processing           | ✅                                           |
| **Post processing analysis**      | ❌                                             | ✅                                           |
| **Output Formats**                | CSV, JSON                                       | Binary (rprof-rep), CSV, TSV, JSON, TXT      |
| **Output Size**                   | Large                                           | Small (binary format)                        |
| **Visualization Tools**           | External (Perfetto)                             | Integrated                                   |
| **Ease of Use**                   | Medium (requires scripting for deeper analysis) | Easy, run and play                           |
| **Overhead**                      | Low to Medium (depends on config)               | Low                                          |


##  Future Work and Improvement

While the current version provides a functional profiling workflow, there are several areas identified for future enhancement:

- **Hardware Counter Support**: Support for hardware performance counters is a work in progress. These metrics are crucial for low-level performance analysis and are planned to be implemented in the next major release.

- **Barrier Dispatch Reliability**: Certain applications may encounter issues with barrier dispatch tracking. Investigating edge cases is on the roadmap.

- **Documentation**: A full and detailed documentation is work in progress.

Community feedback and contributions are welcome to help guide and accelerate these improvements.

##  Contribution

Contributions to RATELProf are welcome and appreciated! Whether you're fixing bugs, improving documentation, adding new features, or optimizing performance, your input helps make this tool better for everyone.

### Guidelines

- Follow existing code style and structure.
- Keep changes focused and well-documented.
- For major changes, open an issue to discuss first.

Thanks for contributing!
