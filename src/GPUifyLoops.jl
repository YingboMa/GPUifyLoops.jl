module GPUifyLoops

abstract type Device end
struct CPU <: Device end

abstract type GPU <: Device end
struct CUDA <: GPU end

#=
# Hopefully we can eventually support AMDGPUs through ROCm
struct ROCm <: GPU end
=#

export CPU, CUDA, Device

using StaticArrays
using Requires

export @setup, @loop, @synchronize
export @scratch, @shmem, launch

"""
   launch(f, args..., kwargs...)

Launch a kernel on the GPU. `kwargs` are passed to `@cuda`
`kwargs` can be any of the compilation and runtime arguments
normally passed to `@cuda`.
"""
launch(f, args...; kwargs...) = error("GPU support not available")

"""
    launch_config(::F, maxthreads, args...; kwargs...)

Calculate a valid launch configuration based on the typeof(F), the
maximum number of threads, the functions arguments and the particular
launch configuration passed to the call.

Return a NamedTuple that has `blocks`, `threads`, `shmem`, and `stream`.
All arguments are optional, but blocks and threads is recommended.
"""
function launch_config(@nospecialize(f), maxthreads, args...; kwargs...)
    return kwargs
end

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    function launch(f::F, args...; kwargs...) where F
        compiler_kwargs, call_kwargs = CUDAnative.split_kwargs(kwargs)
        GC.@preserve args begin
            kernel_args = map(cudaconvert, args)
            kernel_tt = Tuple{map(Core.Typeof, kernel_args)...}
            kernel = cufunction(f, kernel_tt; compilation_kwargs)

            maxthreads = CUDAnative.maxthreads(kernel)
            config = launch_config(f, maxthreads, args...; call_kwargs)

            kernel(kernel_args..., config...)
        end
        return nothing
    end
end

iscpu(::GPU) = false
iscpu(::CPU) = true
sync(::CPU) = nothing
sync(::CUDA) = sync_threads()

@deprecate iscpu(::Val{:GPU}) iscpu(CUDA())
@deprecate iscpu(::Val{:CPU}) iscpu(CPU())
@deprecate sync(::Val{:GPU}) sync(CUDA())
@deprecate sync(::Val{:CPU}) sync(CPU())


"""
    @setup Dev

Setups some hidden state within the function that allows the other macros to
properly work.

```julia
function kernel(::Dev, A) where {Dev}
    @setup Dev
    # ...
end

kernel(A::Array) = kernel(CPU(), A)
kernel(A::CuArray) = @cuda kernel(GPU(), A)
```
"""
macro setup(sym)
    esc(:(local __DEVICE = $sym()))
end

"""
    @syncronize

Calls `sync_threads()` on the GPU and nothing on the CPU.
"""
macro synchronize()
    esc(:($sync(__DEVICE)))
end

# TODO:
# - check if __DEVICE is defined
"""
    @loop for i in (A; B)
        # body
    end

Take a `for i in (A; B)` expression and on the CPU lowers it to:

```julia
for i in A
    # body
end
```

and on the GPU:
```julia
for i in B
    if !(i in A)
        continue
    end
    # body
end
```
"""
macro loop(expr)
    if expr.head != :for
        error("Syntax error: @loop needs a for loop")
    end

    induction = expr.args[1]
    body = expr.args[2]

    if induction.head != :(=)
        error("Syntax error: @loop needs a induction variable")
    end

    rhs = induction.args[2]
    if rhs.head == :block
        @assert length(rhs.args) == 3
        # rhs[2] is a linenode
        cpuidx = rhs.args[1]
        gpuidx = rhs.args[3]

        rhs = Expr(:if, :($iscpu(__DEVICE)), cpuidx, gpuidx)
        induction.args[2] = rhs

        # use cpuidx calculation to check bounds of on GPU.
        bounds_chk = quote
            if !$iscpu(__DEVICE) && !($gpuidx in $cpuidx)
                continue
            end
        end

        pushfirst!(body.args, bounds_chk)
    end

    return esc(Expr(:for, induction, body))
end

include("scratch.jl")
include("shmem.jl")

end
