
using Metal
using Random

# export create_model, model_step!, randomize_matrix!, reset_particles!, download_positions!, get_ptypes


# Threadgroup tile width.  
const TILE_SIZE = 256

#   Model struct                                
mutable struct ParticleModel
  WORLD_SIZE::Float16
  # GPU BUFFERS                    
  px::MtlArray{Float32,1}   # positions
  py::MtlArray{Float32,1}
  vx::MtlArray{Float32,1}   # velocities
  vy::MtlArray{Float32,1}
  ptypes::MtlArray{Int32,1}   # species index 1..num_types
  gpu_fx::MtlArray{Float32,1}   # forces
  gpu_fy::MtlArray{Float32,1}
  gpu_attr::MtlArray{Float32,1}   # flattened attraction matrix

  #  CPU BUFFERS                    ─
  # Reused every frame, download_positions! writes into these in-place.
  cpu_px::Vector{Float32}
  cpu_py::Vector{Float32}
  # Reused during reset, avoids allocation inside reset_particles!.
  cpu_scratch::Vector{Float32}
  cpu_ptypes::Vector{Int32}    # Int32 scratch for species randomisation (same idea)

  #   Simulation parameters                         ─
  attraction_matrix::Matrix{Float32}
  num_types::Int32
  num_particles::Int32
  dt::Float32
  friction::Float32
  max_radius::Float32
  min_radius::Float32
  force_scale::Float32
  steps_per_frame::Int    # display hint: sim steps between position downloads
  step_count::Int

  rng::MersenneTwister

  #   Cached dispatch geometry                        
  groups::Int    # cld(num_particles, TILE_SIZE) — constant, precomputed once

  #   Dirty flag                               
  # Set to true whenever attraction_matrix changes on the CPU side.
  # model_step! syncs gpu_attr and clears the flag.
  attr_dirty::Bool
end

function create_model(;
  num_particles=5000,
  num_types=6,
  world_size=1.0f0,
  max_radius=0.114f0,
  min_radius=0.025f0,
  friction=0.08f0,
  dt=0.0064f0,
  force_scale=8.0f0,
  attraction_matrix=nothing,
  seed=42,
)
  rng = MersenneTwister(seed)

  mat = if attraction_matrix === nothing
    rand(rng, Float32, num_types, num_types) .* 2.0f0 .- 1.0f0
  else
    Float32.(attraction_matrix)
  end

  px, py, vx, vy, ptypes = _make_particles(rng, num_particles, num_types, world_size)

  ParticleModel(
    world_size,
    # GPU buffers
    px, py, vx, vy, ptypes,
    MtlArray(zeros(Float32, num_particles)),  # gpu_fx
    MtlArray(zeros(Float32, num_particles)),  # gpu_fy
    MtlArray(vec(mat)),                       # gpu_attr  
    # CPU buffers
    Vector{Float32}(undef, num_particles),    # cpu_px
    Vector{Float32}(undef, num_particles),    # cpu_py
    Vector{Float32}(undef, num_particles),    # cpu_scratch
    Vector{Int32}(undef, num_particles),      # cpu_ptypes
    # Params
    mat, # attr
    Int32(num_types), Int32(num_particles),
    dt, friction, max_radius, min_radius, force_scale,
    2,   # steps_per_frame default
    0,   # step_count
    rng,
    # Cached geometry
    cld(num_particles, TILE_SIZE),
    # Dirty flag — false because we just uploaded mat above
    false
  )
end

function _make_particles(rng, n, nt, world_size)
  return (
    MtlArray(rand(rng, Float32, n) .* world_size),
    MtlArray(rand(rng, Float32, n) .* world_size),
    MtlArray(zeros(Float32, n)),
    MtlArray(zeros(Float32, n)),
    MtlArray(Int32[rand(rng, 1:nt) for _ in 1:n]),
  )
end



function _force_kernel!(
  fx::MtlDeviceVector{Float32},
  fy::MtlDeviceVector{Float32},
  px::MtlDeviceVector{Float32},
  py::MtlDeviceVector{Float32},
  ptypes::MtlDeviceVector{Int32},
  attraction::MtlDeviceVector{Float32},
  n::Int32, # num particles
  num_types::Int32,
  max_r::Float32,
  max_r_sq::Float32,   
  half_w::Float32,
  world_sz::Float32,
  fscale::Float32,
  beta::Float32, # min_r / max_r
  inv_beta::Float32,   
  inv_mr::Float32,
  mid::Float32,  # (1 + beta) / 2
  inv_hmb::Float32, # 1 / (mid-beta)

) 
  # Threadgroup shared memory for one tile of particle data
  tile_px = MtlThreadGroupArray(Float32, TILE_SIZE)
  tile_py = MtlThreadGroupArray(Float32, TILE_SIZE)
  tile_pt = MtlThreadGroupArray(Int32, TILE_SIZE)

  i = thread_position_in_grid_1d()
  li = thread_position_in_threadgroup_1d() # index in group


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

    threadgroup_barrier(Metal.MemoryFlagThreadGroup) # all threads wait before using shared memory

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
        a = attraction[(tj-Int32(1))*num_types+ti]
        

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

    threadgroup_barrier(Metal.MemoryFlagThreadGroup) # all threads sync before overwriting shared memory
  end

  if valid_i
    fx[i] = fxi
    fy[i] = fyi
  end
  return nothing
end


function _integrate_kernel!(
  px::MtlDeviceVector{Float32},
  py::MtlDeviceVector{Float32},
  vx::MtlDeviceVector{Float32},
  vy::MtlDeviceVector{Float32},
  fx::MtlDeviceVector{Float32},
  fy::MtlDeviceVector{Float32},
  n::Int32,
  dt::Float32,
  damping::Float32,  # 1 − friction
  world_sz::Float32,
)
  i = thread_position_in_grid_1d()
  i > n && return nothing

  nvx = (vx[i] + fx[i] * dt) * damping
  nvy = (vy[i] + fy[i] * dt) * damping
  nx = px[i] + nvx * dt
  ny = py[i] + nvy * dt

  # Wrapping (doesn't use mod bc it's to expensive)
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

  px[i] = nx
  py[i] = ny
  vx[i] = nvx
  vy[i] = nvy
  return nothing
end


function model_step!(model)
  # These are computed here bc of sliders changing them
  max_r = model.max_radius
  max_r_sq = max_r * max_r                
  beta = model.min_radius / max_r
  inv_beta = 1.0f0 / beta              # precomputed: avoids r/beta divide in kernel
  inv_mr = 1.0f0 / max_r
  mid = (1.0f0 + beta) * 0.5f0
  # mid − beta = (1+β)/2 − β = (1−β)/2,  so inv_hmb = 2/(1−β)
  inv_hmb = 2.0f0 / (1.0f0 - beta)
  damping = 1.0f0 - model.friction

  # Sync attr
  if model.attr_dirty
    copyto!(model.gpu_attr, vec(model.attraction_matrix))
    model.attr_dirty = false
  end

  # Get forces
  n = model.num_particles
  groups = model.groups

  @metal threads = TILE_SIZE groups = groups _force_kernel!(
    model.gpu_fx, model.gpu_fy,
    model.px, model.py, model.ptypes,
    model.gpu_attr,
    n, model.num_types,
    max_r, max_r_sq,
    Float32(model.WORLD_SIZE) * 0.5f0, Float32(model.WORLD_SIZE), model.force_scale,
    beta, inv_beta, inv_mr, mid, inv_hmb,
  )

  # Integrate
  @metal threads = TILE_SIZE groups = groups _integrate_kernel!(
    model.px, model.py,
    model.vx, model.vy,
    model.gpu_fx, model.gpu_fy,
    n, model.dt, damping, Float32(model.WORLD_SIZE),
  )

  model.step_count += 1
  return nothing
end









### DISPLAY HELPERS ###

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
  N   = n * n          # output side length
  out = zeros(Float32, N, N)

  px  = model.cpu_px
  py  = model.cpu_py
  np  = Int(model.num_particles)
  ws  = Float32(model.WORLD_SIZE)
  inv_ws = 1.0f0 / ws

  # ── Coarse pass ───────────────────────────────────────────────────────────
  coarse = zeros(Int32, n, n)
  for k in 1:np
    cx = clamp(floor(Int, px[k] * inv_ws * n) + 1, 1, n)
    cy = clamp(floor(Int, py[k] * inv_ws * n) + 1, 1, n)
    coarse[cy, cx] += Int32(1)
  end

  # ── Fine pass: collect sub-cell counts only for hot coarse cells ──────────
  # fine[fy, fx, cy, cx] counts particles in sub-cell (fx,fy) of coarse cell (cx,cy)
  fine = zeros(Int32, n, n, n, n)
  coarse_cell_size = ws / n
  inv_ccs = 1.0f0 / coarse_cell_size

  # We only need to fill fine for cells above threshold
  hot = any(coarse .> threshold)
  if hot
    for k in 1:np
      cx = clamp(floor(Int, px[k] * inv_ws * n) + 1, 1, n)
      cy = clamp(floor(Int, py[k] * inv_ws * n) + 1, 1, n)
      coarse[cy, cx] > threshold || continue
      # position relative to coarse cell origin
      local_x = px[k] - (cx - 1) * coarse_cell_size
      local_y = py[k] - (cy - 1) * coarse_cell_size
      fx = clamp(floor(Int, local_x * inv_ccs * n) + 1, 1, n)
      fy = clamp(floor(Int, local_y * inv_ccs * n) + 1, 1, n)
      fine[fy, fx, cy, cx] += Int32(1)
    end
  end

  # ── Assemble output ───────────────────────────────────────────────────────
  # Output row 0 = world y-max, so we flip cy when mapping to output rows.
  for cy in 1:n, cx in 1:n
    row0 = (n - cy) * n + 1   # first output row owned by this coarse cell
    col0 = (cx - 1) * n + 1   # first output col

    if coarse[cy, cx] > threshold
      # Refined: write individual sub-cell counts
      for fy in 1:n, fx in 1:n
        out_row = row0 + (n - fy)   # flip fine rows too
        out_col = col0 + (fx - 1)
        out[out_row, out_col] = Float32(fine[fy, fx, cy, cx])
      end
    else
      # Uniform: spread coarse count evenly across all n² sub-cells
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
  s = model.cpu_scratch   #It's a prealocated vector instead of alocating a new buffer each randomize.

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

  # Species — rand! into pre-allocated scratch; no allocation
  rand!(model.rng, model.cpu_ptypes, Int32(1):Int32(nt))
  copyto!(model.ptypes, model.cpu_ptypes)

  model.step_count = 0
  return nothing
end

