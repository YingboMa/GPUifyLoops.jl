using GPUifyLoops

kernel(A::Array) = kernel(CPU(), A)
function kernel(::Dev, A) where Dev
    @setup Dev

    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

data = Array{Float32}(undef, 1024)
kernel(data)

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    function kernel(A::CuArray)
        launch(kernel, CPU(), A, threads=length(A))
    end

    data = CuArray{Float32}(undef, 1024)
    kernel(data)
end

