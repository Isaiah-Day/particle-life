
using CUDA
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @synchronize
using Random
using AcceleratedKernels

#export create_model, model_step!, randomize_matrix!, reset_particles!, download_positions!, get_ptypes




### CONSTANTS ####
const NUM_TYPES = 6
const NUM_PARTICLES = 5000
const WORLD_SIZE = 1.0f0
const MAX_RADIUS = 0.114f0
const MIN_RADIUS = 0.025f0
const FRICTION = 0.08f0
const DT = 0.0064f0
const FORCE_SCALE = 8.0f0

const GPU_BACKEND = CUDABackend()
# Threadgroup tile width.  
const TILE_SIZE = 256

#   Model struct                                
mutable struct ParticleModel
  # GPU BUFFERS
  px::CuArray{Float32,1}# positions
  py::CuArray{Float32,1}
  vx::CuArray{Float32,1}# velocities
  vy::CuArray{Float32,1}
  ptypes::CuArray{Int32,1}#species index (1..num_types)
  gpu_fx::CuArray{Float32,1}#forces
  gpu_fy::CuArray{Float32,1}
  gpu_attr::CuArray{Float32,1}# flattened attr

  # CPU BUFFERS
  cpu_px::Vector{Float32}
  cpu_py::Vector{Float32}
  # Reused during reset, avoids allocation inside reset_particles!.
  cpu_scratch::Vector{Float32}
  cpu_ptypes::Vector{Int32}

  #   Simulation parameters                         ─
  attraction_matrix::Matrix{Float32}
  num_types::Int32
  num_particles::Int32
  dt::Float32
  friction::Float32
  max_radius::Float32
  min_radius::Float32
  force_scale::Float32
  steps_per_frame::Int
  step_count::Int

  rng::MersenneTwister

  #   Cached dispatch geometry                        
  groups::Int    # cld(num_particles, TILE_SIZE) — constant, precomputed once

  #   Dirty flag                               
  # Set to true whenever attraction_matrix changes on the CPU side.
  # model_step! syncs gpu_attr and clears the flag.
  attr_dirty::Bool
end


### CREATE MODEL ###
function create_model(;
  num_particles=NUM_PARTICLES,
  num_types=NUM_TYPES,
  attraction_matrix=nothing,
  seed=42,
)
  rng = MersenneTwister(seed)

  mat = if attraction_matrix === nothing
    rand(rng, Float32, num_types, num_types) .* 2.0f0 .- 1.0f0
  else
    Float32.(attraction_matrix)
  end

  px, py, vx, vy, ptypes = _make_particles(rng, num_particles, num_types)

  ParticleModel(
    # GPU buffers
    px, py, vx, vy, ptypes,
    CuArray(zeros(Float32, num_particles)),   # gpu_fx
    CuArray(zeros(Float32, num_particles)),   # gpu_fy
    CuArray(vec(mat)),                         # gpu_attr
    # CPU buffers
    Vector{Float32}(undef, num_particles),    # cpu_px
    Vector{Float32}(undef, num_particles),    # cpu_py
    Vector{Float32}(undef, num_particles),    # cpu_scratch
    Vector{Int32}(undef, num_particles),      # cpu_ptypes
    # Params
    mat, # attr
    Int32(num_types), Int32(num_particles),
    DT, FRICTION, MAX_RADIUS, MIN_RADIUS, FORCE_SCALE,
    2,   # steps_per_frame default
    0,   # step_count
    rng,
    # Cached geometry
    cld(num_particles, TILE_SIZE),
    # Dirty flag — false because we just uploaded mat above
    false
  )
end

function _make_particles(rng, n, nt)
  return (
    CuArray(rand(rng, Float32, n) .* WORLD_SIZE),
    CuArray(rand(rng, Float32, n) .* WORLD_SIZE),
    CuArray(zeros(Float32, n)),
    CuArray(zeros(Float32, n)),
    CuArray(Int32[rand(rng, 1:nt) for _ in 1:n]),
  )
end


### FORCE KERNEL ###
@kernel function _force_kernel!(
  fx, fy, px, py, ptypes, attraction,
  n::Int32, num_types::Int32,
  max_r::Float32, max_r_sq::Float32,
  half_w::Float32, world_sz::Float32, fscale::Float32,
  beta::Float32, inv_beta::Float32,
  inv_mr::Float32, mid::Float32, inv_hmb::Float32,
)
  # Threadgroup shared memory for one tile of particle data.
  tile_px = @localmem Float32 (TILE_SIZE,)
  tile_py = @localmem Float32 (TILE_SIZE,)
  tile_pt = @localmem Int32   (TILE_SIZE,)

  i  = @index(Global, Linear)   # 1-based global thread index
  li = @index(Local,  Linear)   # 1-based index within the workgroup

  valid_i = i <= n
  fxi = 0.0f0 # force accumalators
  fyi = 0.0f0
  ti = valid_i ? ptypes[i] : Int32(1)
  pxi = valid_i ? px[i] : 0.0f0
  pyi = valid_i ? py[i] : 0.0f0

  num_tiles = cld(n, Int32(TILE_SIZE))

  for tile in Int32(1):num_tiles

    # Each thread loads one particle into shared memory.
    j_load = (tile - Int32(1)) * Int32(TILE_SIZE) + li # particle index in px, py and ptypes
    if j_load <= n
      tile_px[li] = px[j_load]
      tile_py[li] = py[j_load]
      tile_pt[li] = ptypes[j_load]
    else
      # Padding, tile_count below ensures these are never visited.
      tile_px[li] = 0.0f0
      tile_py[li] = 0.0f0
      tile_pt[li] = Int32(1)
    end

    @synchronize()   # all threads ready before reading shared memory

    # Accumulate forces from this tile
    if valid_i
      # How many real particles are in this (possibly partial last) tile
      tile_count = min(Int32(TILE_SIZE), n - (tile - Int32(1)) * Int32(TILE_SIZE))

      for lj in Int32(1):tile_count
        # displacement from i to j
        dx = tile_px[lj] - pxi
        dy = tile_py[lj] - pyi

        # space wrapping
        if dx > half_w
          dx -= world_sz
        end
        if dx < -half_w
          dx += world_sz
        end
        if dy > half_w
          dy -= world_sz
        end
        if dy < -half_w
          dy += world_sz
        end

        dist2 = dx * dx + dy * dy
        (dist2 == 0.0f0 || dist2 >= max_r_sq) && continue  # don't calculate i on i force

        
        #  we do it this way bc we need inv_dist later
        inv_dist = 1.0f0 / sqrt(dist2)
        dist = dist2 * inv_dist

        tj = tile_pt[lj]
        a  = attraction[(tj - Int32(1)) * num_types + ti]

        # ACTUAL PL COMPUTATION (remember mult is faster on gpu than div, so we mult by inv_X)
        r = dist * inv_mr           
        f = if r < beta
          -(1.0f0 - r * inv_beta)       
        else
          a * (1.0f0 - abs(r - mid) * inv_hmb)
        end

        inv_d = fscale * inv_dist     
        fxi += f * dx * inv_d
        fyi += f * dy * inv_d
      end
    end

    @synchronize()   # all threads done before next tile overwrites shared memory
  end

  if valid_i
    fx[i] = fxi
    fy[i] = fyi
  end
end


### INTEGRATE KERNEL ###
@kernel function _integrate_kernel!(
  px, py, vx, vy, fx, fy,
  n::Int32, dt::Float32, damping::Float32, world_sz::Float32,
)
  i = @index(Global, Linear)

  if i <= n
    nvx = (vx[i] + fx[i] * dt) * damping
    nvy = (vy[i] + fy[i] * dt) * damping
    nx = px[i] + nvx * dt
    ny = py[i] + nvy * dt

    # Wrapping (doesn't use mod bc it's too expensive)
    if nx >= world_sz
      nx -= world_sz
    end
    if nx < 0.0f0
      nx += world_sz
    end
    if ny >= world_sz
      ny -= world_sz
    end
    if ny < 0.0f0
      ny += world_sz
    end

    px[i] = nx;  py[i] = ny
    vx[i] = nvx; vy[i] = nvy
  end
end


### STEP ###
# Compile kernel instances once per Julia session (KA memoizes them).
const _force_kernel_inst     = _force_kernel!(GPU_BACKEND, TILE_SIZE)
const _integrate_kernel_inst = _integrate_kernel!(GPU_BACKEND, TILE_SIZE)

function model_step!(model)
  max_r    = model.max_radius
  max_r_sq = max_r * max_r
  beta     = model.min_radius / max_r
  inv_beta = 1.0f0 / beta
  inv_mr   = 1.0f0 / max_r
  mid      = (1.0f0 + beta) * 0.5f0
  inv_hmb  = 2.0f0 / (1.0f0 - beta)
  damping  = 1.0f0 - model.friction

  if model.attr_dirty
    copyto!(model.gpu_attr, vec(model.attraction_matrix))
    model.attr_dirty = false
  end

  n      = model.num_particles
  ndrange = model.groups * TILE_SIZE   # padded to full tiles; kernel guards i <= n

  _force_kernel_inst(
    model.gpu_fx, model.gpu_fy,
    model.px, model.py, model.ptypes, model.gpu_attr,
    n, model.num_types,
    max_r, max_r_sq,
    WORLD_SIZE * 0.5f0, WORLD_SIZE, model.force_scale,
    beta, inv_beta, inv_mr, mid, inv_hmb;
    ndrange = ndrange,
  )

  _integrate_kernel_inst(
    model.px, model.py,
    model.vx, model.vy,
    model.gpu_fx, model.gpu_fy,
    n, model.dt, damping, WORLD_SIZE;
    ndrange = ndrange,
  )

  model.step_count += 1
  return nothing
end









### MOVE TO CPU ###

function download_positions!(model)
  copyto!(model.cpu_px, model.px)
  copyto!(model.cpu_py, model.py)
  return nothing
end

get_ptypes(model) = Array(model.ptypes)

"""
    heatmap(model, n, threshold) -> Matrix{Float32}

Returns an n²×n² matrix of particle density with adaptive refinement.

The world is first divided into an n×n coarse grid.  Each coarse cell whose
particle count exceeds `threshold` is further subdivided into its own n×n
sub-grid, giving up to n² detail cells per coarse cell.  Cells below the
threshold are filled with a uniform value (the coarse count spread evenly
across the n² sub-cells they own).

The returned matrix is indexed [row, col] with row 1 at the top (y-max).
"""
function heatmap(model, n::Int, threshold::Real)
  N   = n * n
  out = zeros(Float32, N, N)

  px  = model.cpu_px
  py  = model.cpu_py
  np  = Int(model.num_particles)
  ws  = WORLD_SIZE
  inv_ws = 1.0f0 / ws

  # ── Coarse pass ───────────────────────────────────────────────────────────
  coarse = zeros(Int32, n, n)
  for k in 1:np
    cx = clamp(floor(Int, px[k] * inv_ws * n) + 1, 1, n)
    cy = clamp(floor(Int, py[k] * inv_ws * n) + 1, 1, n)
    coarse[cy, cx] += Int32(1)
  end

  # ── Fine pass: collect sub-cell counts only for hot coarse cells ──────────
  fine = zeros(Int32, n, n, n, n)
  coarse_cell_size = ws / n
  inv_ccs = 1.0f0 / coarse_cell_size

  if any(coarse .> threshold)
    for k in 1:np
      cx = clamp(floor(Int, px[k] * inv_ws * n) + 1, 1, n)
      cy = clamp(floor(Int, py[k] * inv_ws * n) + 1, 1, n)
      coarse[cy, cx] > threshold || continue
      local_x = px[k] - (cx - 1) * coarse_cell_size
      local_y = py[k] - (cy - 1) * coarse_cell_size
      fx = clamp(floor(Int, local_x * inv_ccs * n) + 1, 1, n)
      fy = clamp(floor(Int, local_y * inv_ccs * n) + 1, 1, n)
      fine[fy, fx, cy, cx] += Int32(1)
    end
  end

  # ── Assemble output ───────────────────────────────────────────────────────
  for cy in 1:n, cx in 1:n
    row0 = (n - cy) * n + 1
    col0 = (cx - 1) * n + 1

    if coarse[cy, cx] > threshold
      for fy in 1:n, fx in 1:n
        out[row0 + (n - fy), col0 + (fx - 1)] = Float32(fine[fy, fx, cy, cx])
      end
    else
      val = Float32(coarse[cy, cx]) / Float32(n * n)
      for fy in 1:n, fx in 1:n
        out[row0 + (n - fy), col0 + (fx - 1)] = val
      end
    end
  end

  return out
end


### UTILITIES ###
function randomize_matrix!(model)
  nt = Int(model.num_types)
  model.attraction_matrix .= rand(model.rng, Float32, nt, nt) .* 2.0f0 .- 1.0f0
  model.attr_dirty = true # gpu pays attention
end


function reset_particles!(model)
  n = Int(model.num_particles)
  nt = Int(model.num_types)
  s  = model.cpu_scratch

  # Velocities
  rand!(model.rng, s); s .*= WORLD_SIZE; copyto!(model.px, s)
  rand!(model.rng, s); s .*= WORLD_SIZE; copyto!(model.py, s)
  fill!(model.vx, 0.0f0)
  fill!(model.vy, 0.0f0)

  rand!(model.rng, model.cpu_ptypes, Int32(1):Int32(nt))
  copyto!(model.ptypes, model.cpu_ptypes)

  model.step_count = 0
  return nothing
end

