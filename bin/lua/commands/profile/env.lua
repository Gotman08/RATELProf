local inspect = require ("commands.inspect")

local env = {}

env.env_value = {}

-- Function to set environment variable
function env.set_env(var, value)
    if not var then return end
    table.insert(env.env_value, var.."="..value)
end

function env.get_env()
    return table.concat(env.env_value, " ")
end

function env.set_filter_env_var(filename, filter_type)
    local file = ratelprof.fs.open_file(filename, "r")
    local categories = {}
    local current_category = nil

    for line in file:lines() do
        line = line:trim()

        if line:find(":") then
            -- Found a new category
            local category, names = line:match("^(.-):%s*(.*)$")
            current_category = category:trim()
            categories[current_category] = {}

            if names and names ~= "" then
                -- Add any names that are on the same line as the category
                for name in names:gmatch("[^%s*]+") do
                    table.insert(categories[current_category], name:trim())
                end
            end
        elseif current_category and line ~= "" then
            -- Add names to the current category if they are on separate lines
            for name in line:gmatch("[^%s*]+") do
                table.insert(categories[current_category], name:trim())
            end
        end
    end
    file:close()
    
    for category, names in pairs(categories) do
        local upper_category = string.upper(category)
        env.set_env(string.format(ratelprof.consts._ENV.FUNCTION_FILTERED, upper_category), table.concat(names,","))
        env.set_env(string.format(ratelprof.consts._ENV.FILTER_TYPE, upper_category), filter_type)
    end
end

function env.set_number_of_kernel_env_var(application)
    if ratelprof.consts.IS_NVIDIA then
        -- Only used to size the kernel name table: never fatal on NVIDIA
        -- (kernels may also be JIT compiled or loaded at run time).
        local ok, n = pcall(inspect.get_kernel_number, application, true)
        if not ok or not n or n <= 0 then n = 4096 end
        env.set_env(ratelprof.consts._ENV.NB_KERNEL_AV, math.max(n, 1024))
        return
    end
    local n = inspect.get_kernel_number(application)
    env.set_env(ratelprof.consts._ENV.NB_KERNEL_AV, n)
end

-- NVIDIA: NVTX markers/ranges are collected through the CUPTI NVTX injection.
function env.set_nvidia_env_var(enabled_vars)
    if not ratelprof.consts.IS_NVIDIA then return end
    if enabled_vars[ratelprof.consts._ENV.DOMAIN_ROCTX] then
        local cupti = ratelprof.consts._CUPTI_LIBRARY
        if cupti ~= "" and ratelprof.fs.exists(cupti) then
            env.set_env("NVTX_INJECTION64_PATH", cupti)
        else
            Message:warn("CUPTI library not found ('" .. cupti .. "'): NVTX tracing disabled. Set RATELPROF_CUPTI_LIBRARY.")
        end
    end
end


function env.set_number_of_queue_env_var()
    local num_queue = os.getenv("GPU_MAX_HW_QUEUES") or 4
    env.set_env(ratelprof.consts._ENV.NB_QUEUE_AV, tonumber(num_queue) + 1)
end

return env