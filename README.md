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
`RATELPROF_CUDA_CALLSITE=0` (disable call-site resolution), `RATELPROF_STATIC_HOOKS=0` (disable
in-memory interception, see below), `RATELPROF_STATIC_HOOKS_VERBOSE=1` (list what is intercepted).

### Statically linked libraries and functions LD_PRELOAD cannot reach

LD_PRELOAD only sees calls resolved by the dynamic linker. The CUDA runtime/driver and NVTX do not
depend on it (CUPTI callbacks also cover a statically linked `cudart`), but the other domains do.
RATELProf completes the interposition at start-up, for every enabled domain:

- **Static copies** (`libcublas_static.a`, `libnccl_static.a`, MPI or OpenMP linked statically,
  hidden copies inside a shared library): the symbol tables (`.symtab`, `.dynsym`) of the loaded
  objects are scanned and these functions are **patched in memory** to enter the RATELProf
  wrappers. Each patched definition keeps its own implementation (a trampoline running its
  relocated first instructions), so a static copy and a shared library of the same API coexist.
- **libomptarget plugins**: the `__tgt_rtl_*` entry points (LLVM ≤ 18, even when resolved before
  RATELProf starts) and the `GenericPluginTy::*` C++ methods of LLVM ≥ 19 (plugins linked into
  `libomptarget.so`) are patched; methods are matched by demangled signature.
- **dlopen/dlsym users** (ctypes, cupy, plugins loaded on demand): `dlsym(handle, ...)` returns a
  stub entering the wrapper with the implementation dlsym found.
- **Libraries loaded later** (including with `RTLD_LOCAL`): new objects are scanned when they appear
  and unresolved functions are resolved again on their first call.

The patching engine (`src/core/src/ratelprof/ratelprof_hook.c`) supports aarch64 and x86_64:
branch to a jump island (`b` ±128 MiB / `adrp` ±4 GiB / absolute jump on aarch64, `jmp rel32` /
absolute jump on x86_64), relocation of PC-relative instructions (ADR/ADRP, literal loads,
B/BL/B.cond/CBZ/TBZ, RIP-relative operands, jcc/jmp/call), BTI and pointer-authentication
prologues (`bti c`, `paciasp`) and CET `endbr64`. It refuses to patch when a branch of the function
targets the overwritten instructions. `tests/hook_unit/run.sh` checks it on 11 functions with
varied prologues (stack arguments, floats, loops, calls, tail calls, recursion, PC/RIP-relative
data) for 5 compilation flag sets x PIE/non-PIE x 3 patch modes, on aarch64 (incl. BTI+PAC) and
x86_64 (incl. CET).

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
OpenMP offload with LLVM 16 and 19 (`omp-target` included), statically linked cuBLAS (CUDA 12.6
and 13.1) and NCCL, cuBLAS used through dlopen/dlsym and from a library loaded at run time.
The x86_64 build was compiled (no NVIDIA GPU node of that architecture on ROMEO) and its patching
engine unit-tested. `inspect` was validated on 141 kernel images from sm_50 to
sm_120 (compared against `cuobjdump -res-usage` / `cuobjdump -elf`). CUPTI itself supports
Maxwell and later GPUs; profiling has only been run on Hopper so far.

### Known limitations

- No hardware counters / PC sampling yet (CUPTI Range Profiler / PC Sampling APIs are not wired;
  hardware counters are also work in progress in upstream RATELProf).
- OMPT target-region callbacks (`omp-region`) require an LLVM whose libomptarget emits them
  (LLVM ≥ 17); `omp-target` works with LLVM 16 and 19.
- In-memory interception needs the symbol of the function: a static copy inside a fully stripped
  binary (no `.symtab`, not exported) cannot be found. x86_64 functions shorter than 5 bytes
  without alignment padding after them (typical of `-Os`) cannot be patched.
- A range that a library pushes and never pops (NCCL's `ncclCommInitAll`) is closed when the API
  call that contains it returns.

### Fixes in shared code

The port also fixes issues that affect the AMD build: NCCL/RCCL `stream` arguments truncated to
`int`, uninitialized agent count in object tracking, interposition wrappers called before their
API table is populated (crash with OMPT under LLVM 16), `ratelprof.fs.concat_path` not defined in
`inspect`, aarch64 timer (`cntvct_el0`), glibc `dlsym` symbol versions, `dlsym(RTLD_NEXT)`
answered relative to RATELProf instead of the caller (infinite loop in libomp), the decoder
looping forever on a missing or self-referencing parent event, and libraries loaded after start-up
calling into unresolved functions (fatal error).

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
