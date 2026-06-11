export download_positions!, heatmap, heatmap_slow, get_ptypes
include("model.jl")

function download_positions!(model::ParticleModel)
  copyto!(model.cpu_px, model.px)
  copyto!(model.cpu_py, model.py)
  return nothing
end

get_ptypes(model::ParticleModel) = Array(model.ptypes)



function heatmap(model::ParticleModel, n::Int, threshold::Int)
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


function heatmap_slow(model::ParticleModel, n::Int)
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