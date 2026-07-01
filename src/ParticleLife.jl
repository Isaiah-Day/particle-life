module ParticleLife
using KernelAbstractions
using Metal

# GPUArray(backend, T, D) = MtlArray(T,D)
GPUArray(backend, contents) = MtlArray(contents)

# function get_gpuarray(backend)
#     if backend isa KernelAbstractions.GPU.MetalBackend
#         return MtlArray
#     elseif backend isa KernelAbstractions.GPU.CUDABackend
#         return CuArray
#     elseif backend isa KernelAbstractions.CPU
#         return Array
#     else
#         error("Unsupported backend: $backend")
#     end
# end



include("physics/model.jl")
include("physics/cpu.jl")
include("physics/step.jl")
# include("analysis/organisms.jl")
include("analysis/species.jl")
include("display/display.jl")

export download_positions!, heatmap, heatmap_slow, get_ptypes
export ParticleModel, create_model, randomize_matrix!, reset_particles!
export model_step!
export find_species
# export find_organisms

end
