export ParticleModel, create_model, randomize_matrix!, reset_particles!

using Random
using KernelAbstractions
using GPUSelect
# Threadgroup tile width.  
const TILE_SIZE = 128

#   Model struct                                
mutable struct ParticleModel{}
    WORLD_SIZE::Float16
    # GPU BUFFERS                    
    px::AbstractVector{Float32}   # positions
    py::AbstractVector{Float32}
    vx::AbstractVector{Float32}   # velocities
    vy::AbstractVector{Float32}
    ptypes::AbstractVector{Int32}   # species index 1..num_types
    gpu_fx::AbstractVector{Float32}   # forces
    gpu_fy::AbstractVector{Float32}
    gpu_attr::AbstractVector{Float32}   # flattened attraction matrix

    #  CPU BUFFERS                    ─
    # Reused every frame, download_positions! writes into these in-place.
    cpu_px::Vector{Float32}
    cpu_py::Vector{Float32}
    # Reused during reset, avoids allocation inside reset_particles!.
    cpu_scratch::Vector{Float32}
    cpu_ptypes::Vector{Int32}

    #   Simulation parameters                         ─
    attraction_matrix::Matrix{Float32}
    num_types::Int32
    num_particles::Int32
    dt::AbstractFloat
    friction::AbstractFloat
    max_radius::AbstractFloat
    min_radius::AbstractFloat
    force_scale::AbstractFloat
    steps_per_frame::Int    # display hint: sim steps between position downloads TODO MOVE
    step_count::Int

    rng::MersenneTwister

    #   Cached dispatch geometry                        
    groups::Int    # cld(num_particles, TILE_SIZE), constant

    #   Dirty flag                               
    # Set to true whenever attraction_matrix changes on the CPU side.
    # model_step! syncs gpu_attr and clears the flag.
    attr_dirty::Bool

    # Species proportions: length num_types, sums to 1.
    # Used by reset_particles! to assign species with weighted probability.
    species_weights::Vector{Float32}

    # GRID STUFF
    # Grid dimensions (recomputed when max_radius changes).
    grid_w::Int32
    grid_h::Int32
    # Per-particle cell id (0-based flat index into grid).
    gpu_cell_ids::AbstractVector{Int32}    # [n]
    gpu_sorted_order::AbstractVector{Int32}  # [n]  1-based particle indices sorted by cell
    gpu_cell_start::AbstractVector{Int32}    # [grid_w*grid_h]  first sorted_order index for cell
    gpu_cell_end::AbstractVector{Int32}      # [grid_w*grid_h]  one-past-last index for cell
    cpu_cell_ids::Vector{Int32}
    cpu_sorted_order::Vector{Int32}
    cpu_cell_start::Vector{Int32}
    cpu_cell_end::Vector{Int32}


    # Backend
    backend::Union{<:KernelAbstractions.GPU, KernelAbstractions.CPU}
end

function create_model(;
    num_particles=5000,
    num_types=6,
    world_size=1.0f0,
    max_radius=0.114f0,
    min_radius=0.025f0,
    friction=0.3f0,
    dt=0.0064f0,
    force_scale=8.0f0,
    attraction_matrix=nothing,
    species_weights=nothing,
    seed=42,
    backend = GPUSelect.Backend()
)
    rng = MersenneTwister(seed)

    mat = if attraction_matrix === nothing
        rand(rng, Float32, num_types, num_types) .* 2.0f0 .- 1.0f0
    else
        Float32.(attraction_matrix)
    end

    weights = if species_weights === nothing
        fill(1.0f0 / Float32(num_types), num_types)
    else
        w = Float32.(species_weights)
        w ./ sum(w)
    end

    px, py, vx, vy, ptypes = _make_particles(
        rng, num_particles, num_types, world_size, weights, backend
    )

    n = num_particles
    gw = max(Int32(1), Int32(floor(world_size / max_radius)))
    gh = gw
    ncells = Int(gw) * Int(gh)

    return ParticleModel(
        world_size,
        # GPU buffers
        px,
        py,
        vx,
        vy,
        ptypes,
        GPUArray(backend, zeros(Float32, n)),   # gpu_fx
        GPUArray(backend, zeros(Float32, n)),   # gpu_fy
        GPUArray(backend, vec(mat)),            # gpu_attr
        # CPU buffers
        Vector{Float32}(undef, n),     # cpu_px
        Vector{Float32}(undef, n),     # cpu_py
        Vector{Float32}(undef, n),     # cpu_scratch
        Vector{Int32}(undef, n),       # cpu_ptypes
        # Params
        mat,
        Int32(num_types),
        Int32(n),
        dt,
        friction,
        max_radius,
        min_radius,
        force_scale,
        2,   # steps_per_frame
        0,   # step_count
        rng,
        cld(n, TILE_SIZE),
        false,
        weights,
        # Spatial hash
        gw,
        gh,
        GPUArray(backend, zeros(Int32, n)),         # gpu_cell_ids
        GPUArray(backend, zeros(Int32, n)),         # gpu_sorted_order
        GPUArray(backend, zeros(Int32, ncells)),    # gpu_cell_start
        GPUArray(backend, zeros(Int32, ncells)),    # gpu_cell_end
        Vector{Int32}(undef, n),           # cpu_cell_ids
        Vector{Int32}(undef, n),           # cpu_sorted_order
        Vector{Int32}(undef, ncells),      # cpu_cell_start
        Vector{Int32}(undef, ncells),      # cpu_cell_end
        backend
    )
end

function _make_particles(rng, n, nt, world_size, weights, backend = GPUSelect.Backend())
    # Build species assignments using weighted sampling via cumulative distribution
    cdf = cumsum(weights)
    cdf[end] = 1.0f0  # guard against float rounding
    species = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        r = rand(rng, Float32)
        species[i] = Int32(searchsortedfirst(cdf, r))
    end
    return (
        GPUArray(backend, rand(rng, Float32, n) .* world_size),
        GPUArray(backend, rand(rng, Float32, n) .* world_size),
        GPUArray(backend, zeros(Float32, n)),
        GPUArray(backend, zeros(Float32, n)),
        GPUArray(backend, species),
    )
end

function randomize_matrix!(model::ParticleModel)
    nt = Int(model.num_types)
    model.attraction_matrix .= rand(model.rng, Float32, nt, nt) .* 2.0f0 .- 1.0f0
    return model.attr_dirty = true # gpu pays attention
end

function reset_particles!(model::ParticleModel)
    n = Int(model.num_particles)
    nt = Int(model.num_types)
    s = model.cpu_scratch

    # Positions x
    rand!(model.rng, s)
    s .*= Float32(model.WORLD_SIZE)
    copyto!(model.px, s)

    # Positions y
    rand!(model.rng, s)
    s .*= Float32(model.WORLD_SIZE)
    copyto!(model.py, s)

    # Velocities
    fill!(model.vx, 0.0f0)
    fill!(model.vy, 0.0f0)

    # Species: weighted sampling using species_weights
    cdf = cumsum(model.species_weights)
    cdf[end] = 1.0f0
    @inbounds for i in 1:n
        r = rand(model.rng, Float32)
        model.cpu_ptypes[i] = Int32(searchsortedfirst(cdf, r))
    end
    copyto!(model.ptypes, model.cpu_ptypes)

    model.step_count = 0
    return nothing
end
