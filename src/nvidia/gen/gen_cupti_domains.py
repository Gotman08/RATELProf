#!/usr/bin/env python3
"""
gen_cupti_domains.py - RATELProf NVIDIA backend code generator.

The CUDA Runtime and CUDA Driver API domains of the NVIDIA backend are traced
through the CUPTI Callback API (this is the only robust way to trace the CUDA
runtime when it is statically linked into the application, which is nvcc's
default).  CUPTI identifies each API function by a "callback id" (CBID) and
gives access to a C structure holding the parameters of the call.  Both the
CBID enumerations and the parameter structures change from one CUDA release to
the other, so this script is run at configure time against the CUPTI headers
of the CUDA toolkit that RATELProf is built with.

Generated files (all in --out):
  cuda_rt_api_helper.h / cuda_drv_api_helper.h
      Domain helpers used by the RATELProf core (ids, names, lookups).
  cuda_api_tables.c
      Name tables, parameter-structure sizes, flags and the argument capture
      functions executed in the CUPTI exit callback.
  cuda_rt_args.h / cuda_drv_args.h
      rprofrep argument/parameter writers (one per CBID).
  cupti_records.h
      Typedefs to the CUPTI activity record structures matching the headers
      (CUpti_ActivityKernel9 for CUDA 12.x, CUpti_ActivityKernel11 for 13.x...).

Only the Python standard library is used.
"""

import argparse
import os
import re
import sys

# --------------------------------------------------------------------------- #
#  Parsing helpers                                                            #
# --------------------------------------------------------------------------- #

def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_cbids(text, prefix):
    """Return a list of (name, value) for an enum like CUPTI_RUNTIME_TRACE_CBID_x = n."""
    res = []
    pat = re.compile(r"^\s*" + re.escape(prefix) + r"(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)\s*,?", re.M)
    for m in pat.finditer(text):
        name, val = m.group(1), int(m.group(2), 0)
        if name in ("SIZE", "FORCE_INT", "INVALID"):
            continue
        res.append((name, val))
    size_m = re.search(r"^\s*" + re.escape(prefix) + r"SIZE\s*=\s*(\d+)", text, re.M)
    size = int(size_m.group(1)) if size_m else (max(v for _, v in res) + 1 if res else 0)
    return res, size


def parse_param_structs(text):
    """Return {struct_base_name: [(type, name), ...]} for 'typedef struct X_params_st {...} X_params;'."""
    structs = {}
    pat = re.compile(r"typedef\s+struct\s+(\w+)_params_st\s*\{(.*?)\}\s*(\w+)_params\s*;", re.S)
    for m in pat.finditer(text):
        base, body = m.group(1), m.group(2)
        fields = []
        for line in body.split(";"):
            line = " ".join(line.split())
            if not line or line.startswith("//"):
                continue
            # remove comments
            line = re.sub(r"/\*.*?\*/", "", line).strip()
            if not line:
                continue
            fields.append(split_decl(line))
        structs[base] = fields
    return structs


def split_decl(decl):
    """Split 'const void *src' -> ('const void *', 'src').  Handles arrays and function pointers."""
    decl = decl.strip()
    fp = re.match(r"(.*)\(\s*\*\s*(\w+)\s*\)\s*\((.*)\)$", decl)
    if fp:
        return (fp.group(1).strip() + " (*)(" + fp.group(3).strip() + ")", fp.group(2), "fnptr")
    arr = re.match(r"(.*?)(\w+)\s*(\[.*\])$", decl)
    if arr:
        return (arr.group(1).strip() + " " + arr.group(3), arr.group(2), "array")
    m = re.match(r"(.*?)(\w+)$", decl)
    if not m:
        return (decl, "unnamed", "unknown")
    typ = m.group(1).strip()
    # Normalise pointer spacing: 'void *' / 'const char **'
    typ = re.sub(r"\s*\*", " *", typ).replace("* *", "**").replace(" **", " **")
    typ = re.sub(r"\s+", " ", typ).strip()
    return (typ, m.group(2), "value")


def parse_activity_records(text):
    """Map CUPTI_ACTIVITY_KIND_x -> record struct name using the doc comment of the enumerator."""
    res = {}
    m = re.search(r"typedef\s+enum\s*\{(.*?)\}\s*CUpti_ActivityKind\s*;", text, re.S)
    if not m:
        return res
    body = m.group(1)
    comment = ""
    for line in body.splitlines():
        s = line.strip()
        em = re.match(r"(CUPTI_ACTIVITY_KIND_\w+)\s*=", s)
        if em:
            rm = re.search(r"\\ref\s+(CUpti_Activity\w+)", comment)
            if rm:
                res[em.group(1)] = rm.group(1)
            comment = ""
        else:
            comment += " " + s.lstrip("/*").rstrip("*/")
    return res


# --------------------------------------------------------------------------- #
#  Code generation helpers                                                    #
# --------------------------------------------------------------------------- #

VERSION_SUFFIX = re.compile(r"_v\d+$")

# Name heuristic for API calls that may produce GPU activity records
# (kernels, copies, memsets, synchronizations). Those API events are marked
# as having children so that GPU events can be correlated to them.
LAUNCH_RE = re.compile(
    r"(Launch|Memcpy|Memset|memcpy|memset|Synchronize|WaitEvent|Prefetch|"
    r"GraphLaunch|StreamWaitValue|StreamWriteValue|BatchMemOp|MemcpyBatch|"
    r"cuLaunch|cudaLaunch|Graph.*Upload|ConditionalHandle)")

# Out-parameters whose pointed value is captured at API exit (address returned
# by allocation functions).
DEREF_NAMES = {"devPtr", "ptr", "pHost", "dptr", "pp", "pDevice"}
DEREF_TYPES = {"void **", "CUdeviceptr *"}


def display_name(cbid_name):
    return VERSION_SUFFIX.sub("", cbid_name)


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Domain:
    def __init__(self, key, upper, cbid_prefix, cbids, size, structs, ret_type):
        self.key = key            # cuda_rt / cuda_drv
        self.upper = upper        # CUDA_RT / CUDA_DRV
        self.cbid_prefix = cbid_prefix
        self.cbids = cbids        # [(name, value)]
        self.size = size
        self.structs = structs    # {name: fields}
        self.ret_type = ret_type  # cudaError_t / CUresult

    def struct_for(self, cbid_name):
        if cbid_name in self.structs:
            return cbid_name
        return None

    def deref_fields(self, fields):
        return [f for f in fields if f[2] == "value" and f[0] in DEREF_TYPES and f[1] in DEREF_NAMES]


def gen_helper_header(d):
    out = []
    guard = d.upper + "_API_HELPER_H"
    out.append("/* Generated by src/nvidia/gen/gen_cupti_domains.py - DO NOT EDIT. */")
    out.append("#ifndef " + guard)
    out.append("#define " + guard)
    out.append("")
    out.append("#include <stddef.h>")
    out.append("#include <stdint.h>")
    out.append("#include <string.h>")
    out.append("#include <stdbool.h>")
    out.append("")
    out.append("#define {0}_API_ID_NB_FUNCTION {1}".format(d.upper, d.size))
    out.append("#define {0}_API_ID_UNKNOWN ((size_t)-1)".format(d.upper))
    out.append("#define {0}_ARGS_MAX_SIZE {1}_args_max_size".format(d.upper, d.key))
    out.append("")
    out.append("typedef size_t {0}_api_id_t;".format(d.key))
    out.append("")
    out.append("#define RP_CUDA_FLAG_MAY_LAUNCH  0x1  /* API may produce GPU activity records */")
    out.append("#define RP_CUDA_FLAG_HAS_PARAMS  0x2  /* CUPTI provides a parameter structure */")
    out.append("")
    out.append("extern const char*    const {0}_names[{1}_API_ID_NB_FUNCTION];".format(d.key, d.upper))
    out.append("extern const uint32_t       {0}_args_size[{1}_API_ID_NB_FUNCTION];".format(d.key, d.upper))
    out.append("extern const uint8_t        {0}_flags[{1}_API_ID_NB_FUNCTION];".format(d.key, d.upper))
    out.append("extern const uint32_t       {0}_args_max_size;".format(d.key))
    out.append("")
    out.append("/* Copy the CUPTI parameter structure of 'cbid' (and dereferenced out-params) into 'dst'. */")
    out.append("void rp_{0}_capture_args(uint32_t cbid, const void* params, int64_t retval, void* dst);".format(d.key))
    out.append("")
    out.append("static inline const char* get_{0}_funame_by_id({0}_api_id_t id) {{".format(d.key))
    out.append("    if (id >= {0}_API_ID_NB_FUNCTION) return NULL;".format(d.upper))
    out.append("    return {0}_names[id];".format(d.key))
    out.append("}")
    out.append("")
    out.append("static inline {0}_api_id_t get_{0}_funid_by_name(const char* name) {{".format(d.key))
    out.append("    if (!name) return {0}_API_ID_UNKNOWN;".format(d.upper))
    out.append("    for (size_t i = 0; i < {0}_API_ID_NB_FUNCTION; i++)".format(d.upper))
    out.append("        if ({0}_names[i] && strcmp({0}_names[i], name) == 0) return i;".format(d.key))
    out.append("    return {0}_API_ID_UNKNOWN;".format(d.upper))
    out.append("}")
    out.append("")
    out.append("static inline void* get_{0}_funaddr_by_id({0}_api_id_t id) {{ (void)id; return NULL; }}".format(d.key))
    out.append("")
    out.append("#endif /* " + guard + " */")
    return "\n".join(out) + "\n"


def gen_tables_c(domains):
    out = []
    out.append("/* Generated by src/nvidia/gen/gen_cupti_domains.py - DO NOT EDIT. */")
    out.append("#include <stdint.h>")
    out.append("#include <string.h>")
    out.append("#include <cupti.h> /* runtime/driver parameter structures (+ obsolete v1 types) */")
    out.append('#include "cuda_rt_api_helper.h"')
    out.append('#include "cuda_drv_api_helper.h"')
    out.append("")
    out.append("#define RP_ALIGN8(x) (((x) + 7u) & ~7u)")
    out.append("")
    for d in domains:
        names = ["NULL"] * d.size
        sizes = ["8"] * d.size
        flags = ["0"] * d.size
        max_terms = ["8"]
        for name, val in d.cbids:
            if val >= d.size:
                continue
            names[val] = c_str(display_name(name))
            st = d.struct_for(name)
            fl = []
            if LAUNCH_RE.search(name):
                fl.append("RP_CUDA_FLAG_MAY_LAUNCH")
            if st:
                fl.append("RP_CUDA_FLAG_HAS_PARAMS")
                nderef = len(d.deref_fields(d.structs[st]))
                expr = "RP_ALIGN8(sizeof({0}_params)) + 8 + {1}".format(st, 8 * nderef)
                sizes[val] = expr
                max_terms.append(expr)
            flags[val] = " | ".join(fl) if fl else "0"
        out.append("const char* const {0}_names[{1}_API_ID_NB_FUNCTION] = {{".format(d.key, d.upper))
        out.extend("    " + n + "," for n in names)
        out.append("};")
        out.append("")
        out.append("const uint32_t {0}_args_size[{1}_API_ID_NB_FUNCTION] = {{".format(d.key, d.upper))
        out.extend("    " + s + "," for s in sizes)
        out.append("};")
        out.append("")
        out.append("const uint8_t {0}_flags[{1}_API_ID_NB_FUNCTION] = {{".format(d.key, d.upper))
        out.extend("    " + f + "," for f in flags)
        out.append("};")
        out.append("")
        # max size computed at compile time with a union of the structures
        out.append("typedef union {")
        out.append("    char __rp_dummy[8];")
        for name, val in d.cbids:
            st = d.struct_for(name)
            if st and val < d.size:
                out.append("    {0}_params __{1};".format(st, name))
        out.append("}} __rp_{0}_all_params_t;".format(d.key))
        out.append("const uint32_t {0}_args_max_size = RP_ALIGN8(sizeof(__rp_{0}_all_params_t)) + 8 + 8 * 8;".format(d.key))
        out.append("")
        out.append("void rp_{0}_capture_args(uint32_t cbid, const void* params, int64_t retval, void* dst) {{".format(d.key))
        out.append("    uint8_t* out = (uint8_t*)dst;")
        out.append("    switch (cbid) {")
        for name, val in d.cbids:
            st = d.struct_for(name)
            if not st or val >= d.size:
                continue
            deref = d.deref_fields(d.structs[st])
            out.append("    case {0}: {{".format(val))
            out.append("        const size_t off = RP_ALIGN8(sizeof({0}_params));".format(st))
            out.append("        if (params) memcpy(out, params, sizeof({0}_params)); else memset(out, 0, sizeof({0}_params));".format(st))
            out.append("        memcpy(out + off, &retval, sizeof(retval));")
            for i, (typ, fname, _) in enumerate(deref):
                out.append("        {{ uint64_t v = 0; const {0}_params* p = (const {0}_params*)params;".format(st))
                out.append("          if (p && p->{0} && retval == 0) memcpy(&v, p->{0}, sizeof(*p->{0}));".format(fname))
                out.append("          memcpy(out + off + 8 + {0}, &v, sizeof(v)); }}".format(8 * i))
            out.append("        return; }")
        out.append("    default:")
        out.append("        memcpy(out, &retval, sizeof(retval));")
        out.append("        return;")
        out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out) + "\n"


def gen_args_header(d):
    out = []
    guard = d.upper + "_ARGS_H"
    out.append("/* Generated by src/nvidia/gen/gen_cupti_domains.py - DO NOT EDIT. */")
    out.append("#ifndef " + guard)
    out.append("#define " + guard)
    out.append("")
    out.append("#include <ratelprof.h>")
    out.append("#include <ratelprof_ext.h>")
    out.append("#include <cupti.h> /* runtime/driver parameter structures (+ obsolete v1 types) */")
    out.append('#include "rprofrep_encode_context.h"')
    out.append('#include "utils/rprofrep_msgpack_ext.h"')
    out.append('#include "utils/rprofrep_cuda_value.h"')
    out.append('#include "msgpack.h"')
    out.append("")
    # default writers (CBID without parameter structure)
    out.append("static void write_{0}_default_params_to_buffer(rprofrep_encode_context_t* ctx, msgpack_buffer_t* buf) {{".format(d.key))
    out.append("    msgpack_encode_uint(buf, 1);")
    out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str(d.ret_type)))
    out.append('    rprofrep_msgpack_ext_encode_string(ctx, buf, "retval");')
    out.append("}")
    out.append("static void write_{0}_default_args_to_buffer(const void* args, rprofrep_encode_context_t* ctx, msgpack_buffer_t* buf) {{".format(d.key))
    out.append("    (void)ctx; int64_t r = 0; memcpy(&r, args, sizeof(r)); msgpack_encode_int(buf, r);")
    out.append("}")
    out.append("")
    for name, val in d.cbids:
        st = d.struct_for(name)
        if not st or val >= d.size:
            continue
        fields = d.structs[st]
        deref = d.deref_fields(fields)
        out.append("/* {0} (cbid {1}) */".format(display_name(name), val))
        out.append("static void write_{0}_{1}_params_to_buffer(rprofrep_encode_context_t* ctx, msgpack_buffer_t* buf) {{".format(d.key, val))
        out.append("    msgpack_encode_uint(buf, {0});".format(len(fields) + 1 + len(deref)))
        for typ, fname, _ in fields:
            out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str(typ)))
            out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str(fname)))
        out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str(d.ret_type)))
        out.append('    rprofrep_msgpack_ext_encode_string(ctx, buf, "retval");')
        for typ, fname, _ in deref:
            out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str(typ[:-1].strip() if typ.endswith("*") else typ)))
            out.append("    rprofrep_msgpack_ext_encode_string(ctx, buf, {0});".format(c_str("*" + fname)))
        out.append("}")
        out.append("static void write_{0}_{1}_args_to_buffer(const void* args, rprofrep_encode_context_t* ctx, msgpack_buffer_t* buf) {{".format(d.key, val))
        out.append("    (void)ctx;")
        out.append("    const {0}_params* p = (const {0}_params*)args;".format(st))
        out.append("    const uint8_t* extra = (const uint8_t*)args + RP_ALIGN8(sizeof({0}_params));".format(st))
        for typ, fname, kind in fields:
            if kind in ("array", "unknown"):
                out.append("    msgpack_encode_nil(buf); /* {0} */".format(fname))
            else:
                out.append("    RP_ENC_VALUE(buf, p->{0});".format(fname))
        out.append("    { int64_t r = 0; memcpy(&r, extra, sizeof(r)); msgpack_encode_int(buf, r); }")
        for i, _ in enumerate(deref):
            out.append("    {{ uint64_t v = 0; memcpy(&v, extra + 8 + {0}, sizeof(v)); msgpack_encode_uint(buf, v); }}".format(8 * i))
        out.append("}")
        out.append("")
    out.append("static inline void rprofrep_init_{0}_arg_writers(rprofrep_arg_writer_t* w) {{".format(d.key))
    out.append("    for (size_t i = 0; i < {0}_API_ID_NB_FUNCTION; i++) {{".format(d.upper))
    out.append("        w[i].write_arg   = (rprofrep_write_arg_cb_t)   write_{0}_default_args_to_buffer;".format(d.key))
    out.append("        w[i].write_param = (rprofrep_write_param_cb_t) write_{0}_default_params_to_buffer;".format(d.key))
    out.append("    }")
    for name, val in d.cbids:
        st = d.struct_for(name)
        if not st or val >= d.size:
            continue
        out.append("    w[{1}].write_arg = (rprofrep_write_arg_cb_t) write_{0}_{1}_args_to_buffer; w[{1}].write_param = (rprofrep_write_param_cb_t) write_{0}_{1}_params_to_buffer;".format(d.key, val))
    out.append("}")
    out.append("")
    out.append("#endif /* " + guard + " */")
    return "\n".join(out) + "\n"


RECORD_KINDS = [
    ("CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL", "rp_cupti_kernel_t",     "CUpti_ActivityKernel4"),
    ("CUPTI_ACTIVITY_KIND_MEMCPY",            "rp_cupti_memcpy_t",     "CUpti_ActivityMemcpy"),
    ("CUPTI_ACTIVITY_KIND_MEMCPY2",           "rp_cupti_memcpy_p2p_t", "CUpti_ActivityMemcpy2"),
    ("CUPTI_ACTIVITY_KIND_MEMSET",            "rp_cupti_memset_t",     "CUpti_ActivityMemset"),
    ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",   "rp_cupti_sync_t",       "CUpti_ActivitySynchronization"),
]


def parse_struct_fields(texts, struct_name):
    """Return the set of field names of 'typedef struct ... { ... } struct_name;'."""
    for text in texts:
        end = re.search(r"\}\s*" + re.escape(struct_name) + r"\s*;", text)
        if not end:
            continue
        # walk back to the matching opening brace
        depth, i = 0, end.start()
        while i >= 0:
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        if i < 0:
            continue
        raw = text[i + 1:end.start()]
        # keep only top-level declarations (drop nested struct/union bodies)
        flat, depth = [], 0
        for ch in raw:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0:
                flat.append(ch)
        if True:
            body = re.sub(r"/\*.*?\*/", "", "".join(flat), flags=re.S)
            body = re.sub(r"//[^\n]*", "", body)
            names = set()
            for decl in body.split(";"):
                decl = decl.strip()
                if not decl:
                    continue
                fm = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*(:\s*\d+)?$", decl)
                if fm:
                    names.add(fm.group(1))
            return names
    return set()


RECORD_FIELDS = ["queued", "submitted", "clusterX", "graphNodeId", "launchType", "isSharedMemoryCarveoutRequested",
                 "localMemoryTotal", "localMemoryPerThread", "sharedMemoryConfig", "cacheConfig", "srcKind", "dstKind",
                 "memoryKind", "flags", "channelID", "cudaEventId", "streamId", "deviceId", "contextId"]


def gen_records_header(records, texts=()):
    out = []
    out.append("/* Generated by src/nvidia/gen/gen_cupti_domains.py - DO NOT EDIT. */")
    out.append("#ifndef RP_CUPTI_RECORDS_H")
    out.append("#define RP_CUPTI_RECORDS_H")
    out.append("#include <cupti.h>")
    out.append("")
    for kind, tname, fallback in RECORD_KINDS:
        struct = records.get(kind, fallback)
        out.append("typedef {0} {1}; /* {2} */".format(struct, tname, kind))
        out.append("#define {0}_NAME \"{1}\"".format(tname.upper(), struct))
        fields = parse_struct_fields(texts, struct)
        for f in RECORD_FIELDS:
            if f in fields:
                out.append("#define {0}_HAS_{1} 1".format(tname.upper(), f.upper()))
    out.append("")
    out.append("#endif")
    return "\n".join(out) + "\n"


def write(path, content):
    old = None
    if os.path.exists(path):
        old = read(path)
    if old != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cupti-include", required=True, help="CUPTI include directory (extras/CUPTI/include)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    inc = args.cupti_include
    os.makedirs(args.out, exist_ok=True)

    rt_cbids, rt_size = parse_cbids(read(os.path.join(inc, "cupti_runtime_cbid.h")), "CUPTI_RUNTIME_TRACE_CBID_")
    drv_cbids, drv_size = parse_cbids(read(os.path.join(inc, "cupti_driver_cbid.h")), "CUPTI_DRIVER_TRACE_CBID_")
    rt_structs = parse_param_structs(read(os.path.join(inc, "generated_cuda_runtime_api_meta.h")))
    drv_structs = parse_param_structs(read(os.path.join(inc, "generated_cuda_meta.h")))
    activity_text = read(os.path.join(inc, "cupti_activity.h"))
    deprecated_path = os.path.join(inc, "cupti_activity_deprecated.h")
    texts = [activity_text] + ([read(deprecated_path)] if os.path.exists(deprecated_path) else [])
    records = parse_activity_records(activity_text)

    rt = Domain("cuda_rt", "CUDA_RT", "CUPTI_RUNTIME_TRACE_CBID_", rt_cbids, rt_size, rt_structs, "cudaError_t")
    drv = Domain("cuda_drv", "CUDA_DRV", "CUPTI_DRIVER_TRACE_CBID_", drv_cbids, drv_size, drv_structs, "CUresult")

    write(os.path.join(args.out, "cuda_rt_api_helper.h"), gen_helper_header(rt))
    write(os.path.join(args.out, "cuda_drv_api_helper.h"), gen_helper_header(drv))
    write(os.path.join(args.out, "cuda_api_tables.c"), gen_tables_c([rt, drv]))
    write(os.path.join(args.out, "cuda_rt_args.h"), gen_args_header(rt))
    write(os.path.join(args.out, "cuda_drv_args.h"), gen_args_header(drv))
    write(os.path.join(args.out, "cupti_records.h"), gen_records_header(records, texts))

    nrt = sum(1 for n, v in rt_cbids if rt.struct_for(n))
    ndrv = sum(1 for n, v in drv_cbids if drv.struct_for(n))
    print("gen_cupti_domains: runtime {0} cbids ({1} with params), driver {2} cbids ({3} with params)".format(
        len(rt_cbids), nrt, len(drv_cbids), ndrv))
    for kind, tname, _ in RECORD_KINDS:
        print("gen_cupti_domains: {0} -> {1}".format(kind, records.get(kind, "(fallback)")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
