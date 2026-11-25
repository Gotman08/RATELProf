-- inspect_nvidia.lua
--
-- `ratelprof inspect` for NVIDIA binaries.
--
-- * Static resources of every kernel, read directly from the fat binaries
--   embedded in the application (liborkh: .nv_fatbin / __nv_relfatbin parsing,
--   LZ4/ZSTD decompression, CUDA ELF + EIATTR attributes). No CUDA tool needed,
--   works for every architecture (sm_30 ... sm_120).
-- * Optional SASS analysis (--sass): the cubins are disassembled with nvdisasm
--   (or cuobjdump as a fallback) and every kernel gets an instruction mix
--   (FP32/FP64/FP16/INT/tensor cores/memory spaces/atomics/control/barriers/
--   TMA...), its local memory traffic (register spills) and a list of findings.
-- * Theoretical occupancy estimate from registers / shared memory / launch
--   bounds and the per-architecture SM limits.

local liborkh = require("liborkh_lua")

local inspect_nvidia = {}

-- ------------------------------------------------------------------------- --
-- Architecture data                                                          --
-- ------------------------------------------------------------------------- --

local ARCH_FAMILY = {
    [30] = "Kepler", [32] = "Kepler", [35] = "Kepler", [37] = "Kepler",
    [50] = "Maxwell", [52] = "Maxwell", [53] = "Maxwell",
    [60] = "Pascal", [61] = "Pascal", [62] = "Pascal",
    [70] = "Volta", [72] = "Volta", [75] = "Turing",
    [80] = "Ampere", [86] = "Ampere", [87] = "Ampere", [88] = "Ampere", [89] = "Ada Lovelace",
    [90] = "Hopper", [100] = "Blackwell", [101] = "Blackwell", [103] = "Blackwell",
    [110] = "Blackwell", [120] = "Blackwell", [121] = "Blackwell",
}

-- SM limits (CUDA occupancy calculator data):
-- threads/SM, blocks/SM, registers/SM, max shared memory/SM (bytes),
-- shared memory reserved per block (bytes), FP64 rate (fraction of FP32)
local ARCH_LIMITS = {
    [30] = {2048, 16, 65536,  49152,    0, 1/24}, [32] = {2048, 16, 65536,  49152,    0, 1/24},
    [35] = {2048, 16, 65536,  49152,    0, 1/3},  [37] = {2048, 16, 131072, 114688,   0, 1/3},
    [50] = {2048, 32, 65536,  65536,    0, 1/32}, [52] = {2048, 32, 65536,  98304,    0, 1/32},
    [53] = {2048, 32, 65536,  65536,    0, 1/32}, [60] = {2048, 32, 65536,  65536,    0, 1/2},
    [61] = {2048, 32, 65536,  98304,    0, 1/32}, [62] = {2048, 32, 65536,  65536,    0, 1/32},
    [70] = {2048, 32, 65536,  98304,    0, 1/2},  [72] = {2048, 32, 65536,  98304,    0, 1/32},
    [75] = {1024, 16, 65536,  65536,    0, 1/32}, [80] = {2048, 32, 65536,  167936, 1024, 1/2},
    [86] = {1536, 16, 65536,  102400, 1024, 1/64},[87] = {1536, 16, 65536,  167936, 1024, 1/64},
    [88] = {1536, 16, 65536,  102400, 1024, 1/64},[89] = {1536, 24, 65536,  102400, 1024, 1/64},
    [90] = {2048, 32, 65536,  233472, 1024, 1/2}, [100] = {2048, 32, 65536, 233472, 1024, 1/2},
    [101] = {2048, 32, 65536, 233472, 1024, 1/2}, [103] = {2048, 32, 65536, 233472, 1024, 1/2},
    [110] = {2048, 32, 65536, 233472, 1024, 1/64},[120] = {1536, 32, 65536, 102400, 1024, 1/64},
    [121] = {1536, 32, 65536, 102400, 1024, 1/64},
}

local function arch_name(sm, kind)
    local prefix = (kind == "ptx") and "compute_" or "sm_"
    return prefix .. tostring(sm)
end

-- Theoretical occupancy (active warps / max warps) of a kernel for a block size.
local function occupancy(sm, regs, smem, block)
    local lim = ARCH_LIMITS[sm]
    if not lim or block <= 0 then return nil end
    local max_threads, max_blocks, regs_sm, smem_sm, smem_reserved = lim[1], lim[2], lim[3], lim[4], lim[5]
    local warps_per_block = math.ceil(block / 32)
    local max_warps = max_threads / 32

    local by_threads = math.floor(max_threads / (warps_per_block * 32))
    local by_regs = max_blocks
    if regs > 0 then
        local regs_per_warp = math.ceil(regs * 32 / 256) * 256
        local warps_by_regs = math.floor(regs_sm / regs_per_warp)
        by_regs = math.floor(warps_by_regs / warps_per_block)
    end
    local by_smem = max_blocks
    if smem > 0 or smem_reserved > 0 then
        local smem_block = math.ceil((smem + smem_reserved) / 128) * 128
        by_smem = math.floor(smem_sm / smem_block)
    end
    local blocks = math.min(max_blocks, by_threads, by_regs, by_smem)
    local limiter = "threads"
    if blocks == by_regs and by_regs < by_threads then limiter = "registers" end
    if blocks == by_smem and by_smem < math.min(by_threads, by_regs) then limiter = "shared memory" end
    if blocks == max_blocks and max_blocks < math.min(by_threads, by_regs, by_smem) then limiter = "blocks/SM" end
    local active_warps = blocks * warps_per_block
    return math.min(1, active_warps / max_warps), limiter, blocks
end

-- ------------------------------------------------------------------------- --
-- SASS analysis                                                              --
-- ------------------------------------------------------------------------- --

-- Instruction classes (base opcode, before the first '.'), from Kepler to Blackwell.
local SASS_CLASS = {}
local function class(name, ops) for _, op in ipairs(ops) do SASS_CLASS[op] = name end end

class("fp32",      {"FADD", "FMUL", "FFMA", "FMNMX", "FMNMX3", "FSET", "FSETP", "FSEL", "FCHK", "FADD32I", "FMUL32I",
                    "FFMA32I", "FSWZADD", "FCMP", "FADD2", "FMUL2", "FFMA2"})
class("fp64",      {"DADD", "DMUL", "DFMA", "DMNMX", "DSET", "DSETP"})
class("fp16",      {"HADD2", "HMUL2", "HFMA2", "HMNMX2", "HSET2", "HSETP2", "HADD2_32I", "HMUL2_32I", "HFMA2_32I",
                    "HADD2_F32", "HFMA2_MMA"})
class("int",       {"IADD", "IADD3", "IADD32I", "IMAD", "IMADSP", "IMUL", "IMUL32I", "IMAD32I", "ISCADD", "ISCADD32I",
                    "ISET", "ISETP", "IMNMX", "IMNMX3", "LEA", "LEA_HI", "LOP", "LOP3", "LOP32I", "SHF", "SHL", "SHR",
                    "BFE", "BFI", "POPC", "FLO", "BREV", "IABS", "IDP", "IDP4A", "VABSDIFF", "VABSDIFF4", "XMAD",
                    "BMSK", "SGXT", "VIADD", "VIADDMNMX", "VIMNMX", "VIMNMX3", "ICMP", "VMNMX", "VSET", "VSETP"})
class("convert",   {"F2F", "F2I", "I2F", "I2I", "F2IP", "I2FP", "F2FP", "FRND", "DRND", "F2F64", "I2F64"})
class("sfu",       {"MUFU", "RRO"})
class("tensor",    {"HMMA", "IMMA", "DMMA", "BMMA", "HGMMA", "IGMMA", "QGMMA", "BGMMA", "OMMA", "QMMA", "HMMA2",
                    "UTCHMMA", "UTCQMMA", "UTCIMMA", "UTCOMMA", "UTCMMA", "UTCBAR", "UTCCP", "UTCSHIFT"})
class("ld_global", {"LDG", "LD", "LDGSTS", "LDGDEPBAR", "LDGMC"})
class("st_global", {"STG", "ST"})
class("ld_shared", {"LDS", "LDSM", "LDSU"})
class("st_shared", {"STS", "STSM"})
class("ld_local",  {"LDL"})
class("st_local",  {"STL"})
class("ld_const",  {"LDC", "ULDC"})
class("atomic",    {"ATOM", "ATOMG", "ATOMS", "RED", "REDG", "REDAS", "ATOMR"})
class("texture",   {"TEX", "TLD", "TLD4", "TXQ", "TXD", "TMML", "TEXS", "TLDS", "TLD4S", "SULD", "SUST", "SURED",
                    "SUATOM", "SUQUERY", "SULDGA", "SUSTGA"})
class("tma",       {"UTMALDG", "UTMASTG", "UTMAPF", "UTMACCTL", "UTMACMDFLUSH", "UTMAREDG", "UBLKCP", "UBLKRED", "UBLKPF"})
class("control",   {"BRA", "BRX", "BRXU", "JMP", "JMX", "JMXU", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "BREAK", "SSY",
                    "SYNC", "PBK", "BRK", "PCNT", "CONT", "KILL", "NANOSLEEP", "BPT", "YIELD", "JCAL", "PRET", "PEXIT",
                    "RTT", "ACQBULK", "ENDCOLLECTIVE", "CALL.ABS"})
class("barrier",   {"BAR", "MEMBAR", "DEPBAR", "ERRBAR", "CCTL", "CCTLL", "CCTLT", "WARPSYNC", "SYNCS", "ARRIVES",
                    "FENCE", "UCGABAR_ARV", "UCGABAR_WAIT", "BARRIER"})
class("warp",      {"SHFL", "VOTE", "VOTEU", "MATCH", "REDUX", "ELECT", "CREDUX"})
class("move",      {"MOV", "MOV32I", "S2R", "S2UR", "CS2R", "P2R", "R2P", "R2UR", "PLOP3", "PSETP", "SEL", "PRMT",
                    "NOP", "IMAD.MOV", "PMTRIG", "LEPC", "GETLMEMBASE", "SETLMEMBASE"})

local CLASS_ORDER = {"fp32", "fp64", "fp16", "int", "convert", "sfu", "tensor",
                     "ld_global", "st_global", "ld_shared", "st_shared", "ld_local", "st_local", "ld_const",
                     "atomic", "texture", "tma", "control", "barrier", "warp", "uniform", "move", "other"}

local CLASS_LABEL = {
    fp32 = "FP32", fp64 = "FP64", fp16 = "FP16/BF16", int = "INT", convert = "Conversion", sfu = "SFU (MUFU)",
    tensor = "Tensor Core", ld_global = "Global Load", st_global = "Global Store", ld_shared = "Shared Load",
    st_shared = "Shared Store", ld_local = "Local Load", st_local = "Local Store", ld_const = "Constant Load",
    atomic = "Atomic", texture = "Texture/Surface", tma = "TMA/Bulk Copy", control = "Control Flow",
    barrier = "Barrier/Fence", warp = "Warp-level", uniform = "Uniform Datapath", move = "Move/Predicate",
    other = "Other",
}

local function classify(op)
    local c = SASS_CLASS[op]
    if c then return c end
    if op:sub(1, 1) == "U" and #op > 1 then return "uniform" end
    return "other"
end

local function which(tool)
    local p = ratelprof.utils.execute_command("command -v " .. tool .. " 2>/dev/null")
    p = p and p:gsub("%s+$", "") or ""
    if p ~= "" then return p end
    local root = ratelprof.consts._CUDA_ROOT
    if root and root ~= "" and ratelprof.fs.exists(root .. "/bin/" .. tool) then return root .. "/bin/" .. tool end
    return nil
end

-- Parse a SASS listing (nvdisasm or cuobjdump -sass). Returns {name -> stats}.
local function parse_sass(text)
    local funcs = {}
    local cur = nil
    for line in text:gmatch("[^\n]+") do
        local fname = line:match("^%.text%.([^:%s]+):%s*$") or line:match("^%s*Function%s*:%s*(%S+)")
        if fname then
            cur = funcs[fname] or { total = 0, classes = {}, opcodes = {}, predicated = 0, listing = {} }
            funcs[fname] = cur
        elseif cur then
            local offset, ins = line:match("/%*(%x+)%*/%s+(.-)%s*;")
            if offset and ins and ins ~= "" then
                table.insert(cur.listing, string.format("/*%s*/ %s", offset, ins))
                local body = ins
                if body:match("^@") then
                    if not body:match("^@!?U?PT%s") then cur.predicated = cur.predicated + 1 end
                    body = body:gsub("^@%S+%s+", "")
                end
                body = body:gsub("^{%s*", "")
                local full = body:match("^([%w_%.]+)")
                if full then
                    local base = full:match("^([%w_]+)")
                    local c = (full:match("^IMAD%.MOV") and "move") or classify(base)
                    cur.total = cur.total + 1
                    cur.classes[c] = (cur.classes[c] or 0) + 1
                    cur.opcodes[base] = (cur.opcodes[base] or 0) + 1
                end
            end
        end
    end
    return funcs
end

local function disassemble(cubin, sm)
    local nvdisasm = which("nvdisasm")
    if nvdisasm then
        local out = ratelprof.utils.execute_command(string.format("%s -c '%s' 2>/dev/null", nvdisasm, cubin))
        if out and out:find("/%*%x+%*/") then return out, "nvdisasm" end
    end
    local cuobjdump = which("cuobjdump")
    if cuobjdump then
        local out = ratelprof.utils.execute_command(string.format("%s -sass '%s' 2>/dev/null", cuobjdump, cubin))
        if out and out:find("/%*%x+%*/") then return out, "cuobjdump" end
    end
    return nil, nil
end

-- Findings in the spirit of a SASS audit.
local function findings(k, sass, sm)
    local f = {}
    local lim = ARCH_LIMITS[sm]
    if k.frame_size > 0 or (sass and ((sass.classes.st_local or 0) + (sass.classes.ld_local or 0)) > 0) then
        local st = sass and (sass.classes.st_local or 0) or 0
        local ld = sass and (sass.classes.ld_local or 0) or 0
        table.insert(f, string.format("local memory: %d B/thread stack frame, %d STL / %d LDL (register spills or local arrays)",
            k.frame_size, st, ld))
    end
    if k.registers >= 128 then
        table.insert(f, string.format("high register pressure (%d registers/thread)", k.registers))
    end
    if sass then
        local fp64 = sass.classes.fp64 or 0
        if fp64 > 0 and lim and lim[6] <= 1/32 then
            table.insert(f, string.format("%d FP64 instructions on %s (FP64 rate 1/%d of FP32)", fp64, arch_name(sm), math.floor(1 / lim[6] + 0.5)))
        end
        if (sass.classes.tensor or 0) > 0 then
            table.insert(f, string.format("uses tensor cores (%d MMA instructions)", sass.classes.tensor))
        end
        if (sass.classes.tma or 0) > 0 then table.insert(f, "uses TMA / bulk asynchronous copies") end
        if (sass.classes.atomic or 0) > 0 then
            table.insert(f, string.format("%d atomic/reduction instructions", sass.classes.atomic))
        end
        if (sass.classes.texture or 0) > 0 then table.insert(f, "uses texture/surface instructions") end
        if (sass.opcodes.LDGSTS or 0) > 0 then table.insert(f, "uses cp.async (LDGSTS)") end
    end
    if k.cluster_dim[1] > 0 then
        table.insert(f, string.format("thread block cluster %dx%dx%d", k.cluster_dim[1], k.cluster_dim[2], k.cluster_dim[3]))
    end
    if k.coop_groups then table.insert(f, "uses cooperative groups grid synchronization") end
    if k.atomic_sys then table.insert(f, "uses system-scope atomics") end
    return f
end

-- ------------------------------------------------------------------------- --
-- Public API                                                                 --
-- ------------------------------------------------------------------------- --

function inspect_nvidia.get_kernel_number(binary_app)
    local ok, n = pcall(liborkh.nv_kernel_count, binary_app)
    if not ok then return 0 end
    return n or 0
end

local function parse_archs(archs)
    local set = nil
    if archs and #archs > 0 then
        set = {}
        for _, a in ipairs(archs) do
            local n = tonumber(tostring(a):match("(%d+)"))
            if n then set[n] = true end
        end
    end
    return set
end

local function csv_escape(v)
    v = tostring(v)
    v = v:gsub('^"(.*)"$', "%1") -- get_kernel_name() returns quoted names
    if v:find('[,"\n]') then v = '"' .. v:gsub('"', '""') .. '"' end
    return v
end

function inspect_nvidia.inspect_kernels(application, opt, options_helper)
    local output_dir = ratelprof.get_opt_val(opt, "output_dir") or "."
    local mangled    = ratelprof.get_opt_val(opt, "mangled")
    local trunc      = ratelprof.get_opt_val(opt, "trunc")
    local save_json  = ratelprof.get_opt_val(opt, "save-json")
    local do_sass    = ratelprof.get_opt_val(opt, "sass") or ratelprof.get_opt_val(opt, "sass-dump")
    local sass_dump  = ratelprof.get_opt_val(opt, "sass-dump")
    local archs      = parse_archs(options_helper.parse_list_option(opt, "archs"))

    Message:print("RPROF: Inspecting '" .. application .. "' (NVIDIA fat binaries) . . .")

    local tmpdir = nil
    if do_sass then
        tmpdir = os.tmpname()
        os.remove(tmpdir)
        ratelprof.fs.mkdir(tmpdir)
    end

    local images, err = liborkh.nv_inspect(application, { dump_dir = tmpdir, attributes = save_json and true or false })
    if not images then
        Message:error("Cannot read '" .. application .. "': " .. tostring(err))
        os.exit(1)
    end
    if #images == 0 then
        Message:print("RPROF: No CUDA fat binary found in '" .. application .. "'.")
    end

    ratelprof.fs.mkdir(output_dir)
    local basename = ratelprof.fs.basename(application)
    local json_doc = { application = application, vendor = "NVIDIA", images = {} }
    local sass_tool_used = nil
    local nimages = 0

    for id, img in ipairs(images) do
        if (not archs or archs[img.sm]) and #img.kernels > 0 then
            nimages = nimages + 1
            local arch = arch_name(img.sm, img.kind)
            local family = ARCH_FAMILY[img.sm] or "Unknown"
            local sass_funcs = nil
            if do_sass and img.kind == "cubin" and img.file then
                local text, tool = disassemble(img.file, img.sm)
                if text then
                    sass_tool_used = tool
                    sass_funcs = parse_sass(text)
                    if sass_dump then
                        local fsass = ratelprof.fs.join(output_dir, string.format("inspect_%s_%d_%s.sass", basename, id, arch))
                        local f = ratelprof.fs.open_file(fsass, "w")
                        f:write(text)
                        f:close()
                        Message:print("RPROF: SASS written to " .. fsass)
                    end
                else
                    Message:warn("nvdisasm/cuobjdump not found or failed: no SASS analysis for " .. arch .. ".")
                end
            end

            local header = { "Name", "Arch", "Kind", "Registers", "Static Shared Mem (B)", "Local Mem/Thread (B)",
                             "Max Stack (B)", "Constant[0] (B)", "Param Size (B)", "Num Params", "Max Reg Count",
                             "Launch Bounds", "Cluster Dim", "Code Size (B)", "Occupancy@LB/256 (%)", "Occupancy Limiter" }
            if sass_funcs then
                table.insert(header, "SASS Instructions")
                for _, c in ipairs(CLASS_ORDER) do table.insert(header, CLASS_LABEL[c]) end
            end
            table.insert(header, "Findings")
            local rows = { table.concat(header, ", ") }

            local json_img = { arch = arch, family = family, kind = img.kind, compression = img.compression,
                               origin = img.origin, size = img.size, debug = img.debug, kernels = {} }

            table.sort(img.kernels, function(a, b) return a.name < b.name end)
            for _, k in ipairs(img.kernels) do
                if k.is_entry or img.kind == "ptx" then
                    local name = ratelprof.utils.get_kernel_name(k.name, trunc, mangled)
                    local block = (k.req_ntid[1] > 0 and k.req_ntid[1] * k.req_ntid[2] * k.req_ntid[3])
                               or (k.max_threads[1] > 0 and k.max_threads[1] * k.max_threads[2] * k.max_threads[3])
                               or 256
                    local occ, limiter, blocks = nil, "", nil
                    if img.kind == "cubin" then occ, limiter, blocks = occupancy(img.sm, k.registers, k.shared_static, block) end
                    local lb = (k.max_threads[1] > 0) and string.format("%dx%dx%d", k.max_threads[1], k.max_threads[2], k.max_threads[3]) or "-"
                    local cl = (k.cluster_dim[1] > 0) and string.format("%dx%dx%d", k.cluster_dim[1], k.cluster_dim[2], k.cluster_dim[3]) or "-"
                    local sass = sass_funcs and sass_funcs[k.name] or nil
                    local fnd = findings(k, sass, img.sm)

                    local row = { csv_escape(name), arch, img.kind, k.registers, k.shared_static,
                                  math.max(k.frame_size, k.local_size), k.max_stack_size, k.constant0_size,
                                  k.param_size, k.num_params, k.max_reg_count, lb, cl, k.text_size,
                                  occ and string.format("%.1f", occ * 100) or "-", limiter ~= "" and limiter or "-" }
                    if sass_funcs then
                        table.insert(row, sass and sass.total or 0)
                        for _, c in ipairs(CLASS_ORDER) do table.insert(row, sass and (sass.classes[c] or 0) or 0) end
                    end
                    table.insert(row, csv_escape(table.concat(fnd, "; ")))
                    table.insert(rows, table.concat(row, ", "))

                    local jk = {
                        name = k.name, demangled = ratelprof.utils.get_kernel_name(k.name, false, false),
                        registers = k.registers, shared_static = k.shared_static, frame_size = k.frame_size,
                        local_size = k.local_size, min_stack_size = k.min_stack_size, max_stack_size = k.max_stack_size,
                        crs_stack_size = k.crs_stack_size, constant0_size = k.constant0_size, param_size = k.param_size,
                        num_params = k.num_params, max_reg_count = k.max_reg_count, max_threads = k.max_threads,
                        req_ntid = k.req_ntid, cluster_dim = k.cluster_dim, max_cluster_rank = k.max_cluster_rank,
                        num_barriers = k.num_barriers, num_exits = k.num_exits, code_size = k.text_size,
                        ctaidz_used = k.ctaidz_used, wmma_used = k.wmma_used, coop_groups = k.coop_groups,
                        atomic_sys = k.atomic_sys, mbarrier = k.mbarrier,
                        occupancy = occ and { block_size = block, occupancy = occ, blocks_per_sm = blocks, limiter = limiter } or nil,
                        findings = fnd, attributes = k.attributes,
                    }
                    if sass then
                        jk.sass = { instructions = sass.total, predicated = sass.predicated, classes = sass.classes, opcodes = sass.opcodes }
                    end
                    table.insert(json_img.kernels, jk)
                end
            end

            local suffix = string.format("%d_%s", id, arch)
            local output_file_csv = ratelprof.fs.join(output_dir, "inspect_" .. basename .. "_" .. suffix .. ".csv")
            local file = ratelprof.fs.open_file(output_file_csv, "w")
            file:write(table.concat(rows, "\n"))
            file:close()
            Message:print(string.format("RPROF: %s %s (%s, %s, %d kernels) -> %s",
                img.kind, arch, family, img.compression ~= "none" and (img.compression .. " compressed") or "uncompressed",
                #json_img.kernels, output_file_csv))
            table.insert(json_doc.images, json_img)
        end
    end

    if sass_tool_used then Message:print("RPROF: SASS analysis done with " .. sass_tool_used .. ".") end
    if nimages == 0 and #images > 0 then Message:print("RPROF: No image matches the selected architectures.") end

    if save_json then
        local output_file_json = ratelprof.fs.join(output_dir, "inspect_" .. basename .. ".json")
        local f = ratelprof.fs.open_file(output_file_json, "w")
        f:write(ratelprof.utils.generate_json(json_doc))
        f:close()
        Message:print("RPROF: JSON output written to " .. output_file_json)
    end

    if tmpdir then os.execute("rm -rf '" .. tmpdir .. "'") end
end

return inspect_nvidia
