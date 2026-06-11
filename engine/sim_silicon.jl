
using Metal
using Random

# export create_model, model_step!, randomize_matrix!, reset_particles!, download_positions!, get_ptypes


# Threadgroup tile width.  
const TILE_SIZE = 128

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
  gpu_cell_ids::MtlArray{Int32,1}      # [n]
  gpu_sorted_order::MtlArray{Int32,1}  # [n]  1-based particle indices sorted by cell
  gpu_cell_start::MtlArray{Int32,1}    # [grid_w*grid_h]  first sorted_order index for cell
  gpu_cell_end::MtlArray{Int32,1}      # [grid_w*grid_h]  one-past-last index for cell
  cpu_cell_ids::Vector{Int32}
  cpu_sorted_order::Vector{Int32}
  cpu_cell_start::Vector{Int32}
  cpu_cell_end::Vector{Int32}
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

  px, py, vx, vy, ptypes = _make_particles(rng, num_particles, num_types, world_size, weights)

  n = num_particles
  gw = max(Int32(1), Int32(floor(world_size / max_radius)))
  gh = gw
  ncells = Int(gw) * Int(gh)

  ParticleModel(
    world_size,
    # GPU buffers
    px, py, vx, vy, ptypes,
    MtlArray(zeros(Float32, n)),   # gpu_fx
    MtlArray(zeros(Float32, n)),   # gpu_fy
    MtlArray(vec(mat)),            # gpu_attr
    # CPU buffers
    Vector{Float32}(undef, n),     # cpu_px
    Vector{Float32}(undef, n),     # cpu_py
    Vector{Float32}(undef, n),     # cpu_scratch
    Vector{Int32}(undef, n),       # cpu_ptypes
    # Params
    mat,
    Int32(num_types), Int32(n),
    dt, friction, max_radius, min_radius, force_scale,
    2,   # steps_per_frame
    0,   # step_count
    rng,
    cld(n, TILE_SIZE),
    false,
    weights,
    # Spatial hash
    gw, gh,
    MtlArray(zeros(Int32, n)),         # gpu_cell_ids
    MtlArray(zeros(Int32, n)),         # gpu_sorted_order
    MtlArray(zeros(Int32, ncells)),    # gpu_cell_start
    MtlArray(zeros(Int32, ncells)),    # gpu_cell_end
    Vector{Int32}(undef, n),           # cpu_cell_ids
    Vector{Int32}(undef, n),           # cpu_sorted_order
    Vector{Int32}(undef, ncells),      # cpu_cell_start
    Vector{Int32}(undef, ncells),      # cpu_cell_end
  )
end

function _make_particles(rng, n, nt, world_size, weights)
  # Build species assignments using weighted sampling via cumulative distribution
  cdf = cumsum(weights)
  cdf[end] = 1.0f0  # guard against float rounding
  species = Vector{Int32}(undef, n)
  @inbounds for i in 1:n
    r = rand(rng, Float32)
    species[i] = Int32(searchsortedfirst(cdf, r))
  end
  return (
    MtlArray(rand(rng, Float32, n) .* world_size),
    MtlArray(rand(rng, Float32, n) .* world_size),
    MtlArray(zeros(Float32, n)),
    MtlArray(zeros(Float32, n)),
    MtlArray(species),
  )
end



function _compute_cells_kernel!(
  cell_ids::MtlDeviceVector{Int32},
  px::MtlDeviceVector{Float32},
  py::MtlDeviceVector{Float32},
  n::Int32,
  inv_max_r::Float32,
  grid_w::Int32,
  grid_h::Int32,
)
  i = thread_position_in_grid_1d()
  i > n && return nothing
  cx = min(unsafe_trunc(Int32, px[i] * inv_max_r), grid_w - Int32(1))
  cy = min(unsafe_trunc(Int32, py[i] * inv_max_r), grid_h - Int32(1))
  cell_ids[i] = cx + cy * grid_w   # 0-based flat cell index
  return nothing
end


function _force_kernel!(
  fx::MtlDeviceVector{Float32},
  fy::MtlDeviceVector{Float32},
  px::MtlDeviceVector{Float32},
  py::MtlDeviceVector{Float32},
  ptypes::MtlDeviceVector{Int32},
  sorted_order::MtlDeviceVector{Int32},
  cell_start::MtlDeviceVector{Int32},
  cell_end::MtlDeviceVector{Int32},
  attraction::MtlDeviceVector{Float32},
  n::Int32,
  num_types::Int32,
  max_r::Float32,
  max_r_sq::Float32,
  half_w::Float32,
  world_sz::Float32,
  fscale::Float32,
  beta::Float32,
  inv_beta::Float32,
  inv_mr::Float32,
  mid::Float32,
  inv_hmb::Float32,
  grid_w::Int32,
  grid_h::Int32,
)
  i = thread_position_in_grid_1d()
  i > n && return nothing

  pxi = px[i]
  pyi = py[i]
  ti  = ptypes[i]
  fxi = 0.0f0
  fyi = 0.0f0

  # Coarse cell of particle i
  cx = min(unsafe_trunc(Int32, pxi / max_r), grid_w - Int32(1))
  cy = min(unsafe_trunc(Int32, pyi / max_r), grid_h - Int32(1))

  for dy in Int32(-1):Int32(1)
    for dx in Int32(-1):Int32(1)
      nx = mod(cx + dx, grid_w)
      ny = mod(cy + dy, grid_h)
      cell = nx + ny * grid_w + Int32(1)   # +1 bc julia is one-based

      kstart = cell_start[cell]
      kend   = cell_end[cell]
      kstart > kend && continue

      for k in kstart:kend
        j = sorted_order[k]

        dx2 = px[j] - pxi
        dy2 = py[j] - pyi

        # space wrapping
        if dx2 >  half_w; dx2 -= world_sz; end
        if dx2 < -half_w; dx2 += world_sz; end
        if dy2 >  half_w; dy2 -= world_sz; end
        if dy2 < -half_w; dy2 += world_sz; end

        # More precompute
        # "blazingly fast"
        dist2 = dx2 * dx2 + dy2 * dy2
        (dist2 < 1.0f-8 || dist2 >= max_r_sq) && continue

        inv_dist = 1.0f0 / sqrt(dist2)
        dist = dist2 * inv_dist

        tj = ptypes[j]
        a  = attraction[(tj - Int32(1)) * num_types + ti]

        # Actual PL Calc
        r = dist * inv_mr
        f = if r < beta
          -(1.0f0 - r * inv_beta)
        else
          a * (1.0f0 - abs(r - mid) * inv_hmb)
        end

        inv_d = fscale * inv_dist
        fxi += f * dx2 * inv_d
        fyi += f * dy2 * inv_d
      end
    end
  end

  fx[i] = fxi
  fy[i] = fyi
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
  # Precompute
  max_r  = model.max_radius
  max_r_sq = max_r * max_r
  beta   = model.min_radius / max_r
  inv_beta = 1.0f0 / beta
  inv_mr = 1.0f0 / max_r
  mid    = (1.0f0 + beta) * 0.5f0
  inv_hmb = 2.0f0 / (1.0f0 - beta)
  damping = 1.0f0 - model.friction

  if model.attr_dirty
    copyto!(model.gpu_attr, vec(model.attraction_matrix))
    model.attr_dirty = false
  end

  n  = model.num_particles
  ws = Float32(model.WORLD_SIZE)
  groups = model.groups

  ### BULD GRID

  # # Recompute grid dims if max_radius changed
  # new_gw = max(Int32(1), Int32(floor(ws / max_r)))
  # new_gh = new_gw
  # if new_gw != model.grid_w || new_gh != model.grid_h
  #   ncells = Int(new_gw) * Int(new_gh)
  #   model.grid_w = new_gw
  #   model.grid_h = new_gh
  #   model.gpu_cell_start = MtlArray(zeros(Int32, ncells))
  #   model.gpu_cell_end   = MtlArray(zeros(Int32, ncells))
  #   model.cpu_cell_start = Vector{Int32}(undef, ncells)
  #   model.cpu_cell_end   = Vector{Int32}(undef, ncells)
  # end

  gw = model.grid_w
  gh = model.grid_h

  # Assign each particle on a cell
  @metal threads=TILE_SIZE groups=groups _compute_cells_kernel!(
    model.gpu_cell_ids, model.px, model.py,
    n, inv_mr, gw, gh,
  )

  # Download sorted and cell ids to the cpu
  # TODO: Why is this not in download_positions!
  copyto!(model.cpu_cell_ids, model.gpu_cell_ids)
  sortperm!(model.cpu_sorted_order, model.cpu_cell_ids)
  copyto!(model.gpu_sorted_order, model.cpu_sorted_order)

  # Build cell_start and cell_end on CPU
  ncells = Int(gw) * Int(gh)
  fill!(model.cpu_cell_start, Int32(0))
  fill!(model.cpu_cell_end,   Int32(0))
  prev_cell = Int32(-1)
  for k in 1:Int(n)
    cell = model.cpu_cell_ids[model.cpu_sorted_order[k]] + Int32(1)  # 1-based
    if cell != prev_cell
      if prev_cell >= Int32(1)
        model.cpu_cell_end[prev_cell] = Int32(k - 1)
      end
      model.cpu_cell_start[cell] = Int32(k)
      prev_cell = cell
    end
  end
  if prev_cell >= Int32(1)
    model.cpu_cell_end[prev_cell] = n
  end
  copyto!(model.gpu_cell_start, model.cpu_cell_start)
  copyto!(model.gpu_cell_end,   model.cpu_cell_end)

  ### FORCE KERNEL
  @metal threads=TILE_SIZE groups=groups _force_kernel!(
    model.gpu_fx, model.gpu_fy,
    model.px, model.py, model.ptypes,
    model.gpu_sorted_order,
    model.gpu_cell_start, model.gpu_cell_end,
    model.gpu_attr,
    n, model.num_types,
    max_r, max_r_sq,
    ws * 0.5f0, ws, model.force_scale,
    beta, inv_beta, inv_mr, mid, inv_hmb,
    gw, gh,
  )

  ### INTEGRATE
  @metal threads=TILE_SIZE groups=groups _integrate_kernel!(
    model.px, model.py,
    model.vx, model.vy,
    model.gpu_fx, model.gpu_fy,
    n, model.dt, damping, ws,
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



function heatmap(model, n::Int, threshold::Int)
  download_positions!(model)
  # n = output grid side length
  # coarse grid = nc×nc where nc = isqrt(n)
  nc  = isqrt(n)
  out = zeros(Float32, n, n)

  px     = model.cpu_px
  py     = model.cpu_py
  np     = Int(model.num_particles)
  ws     = Float32(model.WORLD_SIZE)
  inv_ws = 1.0f0 / ws

  # COARSE PASS
  coarse = zeros(Int32, nc, nc)
  for k in 1:np
    cx = clamp(floor(Int, px[k] * inv_ws * nc) + 1, 1, nc)
    cy = clamp(floor(Int, py[k] * inv_ws * nc) + 1, 1, nc)
    coarse[cx, cy] += Int32(1)
  end

  # FINE PASS — write directly into out
  coarse_cell_size = ws / nc
  inv_ccs = 1.0f0 / coarse_cell_size

  if maximum(coarse) > threshold
    for k in 1:np
      cx = clamp(floor(Int, px[k] * inv_ws * nc) + 1, 1, nc)
      cy = clamp(floor(Int, py[k] * inv_ws * nc) + 1, 1, nc)
      coarse[cx, cy] > threshold || continue
      local_x = px[k] - (cx - 1) * coarse_cell_size
      local_y = py[k] - (cy - 1) * coarse_cell_size
      fx = clamp(floor(Int, local_x * inv_ccs * nc) + 1, 1, nc)
      fy = clamp(floor(Int, local_y * inv_ccs * nc) + 1, 1, nc)
      out[(cx - 1) * nc + fx, (cy - 1) * nc + fy] += 1.0f0
    end
  end

  # FILL SPARSE CELLS
  for cx in 1:nc, cy in 1:nc
    coarse[cx, cy] > threshold && continue
    fill_val = Float32(coarse[cx, cy]) / Float32(nc * nc)
    col0 = (cx - 1) * nc + 1
    row0 = (cy - 1) * nc + 1
    for fx in 1:nc, fy in 1:nc
      out[col0 + (fx - 1), row0 + (fy - 1)] = fill_val
    end
  end

  return out
end


function heatmap_slow(model, n::Int)
  download_positions!(model)
  out    = zeros(Float32, n, n)
  px     = model.cpu_px
  py     = model.cpu_py
  np     = Int(model.num_particles)
  ws     = Float32(model.WORLD_SIZE)
  inv_ws = 1.0f0 / ws
  for k in 1:np
    cx = clamp(floor(Int, px[k] * inv_ws * n) + 1, 1, n)
    cy = clamp(floor(Int, py[k] * inv_ws * n) + 1, 1, n)
    out[cx, cy] += 1.0f0
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

