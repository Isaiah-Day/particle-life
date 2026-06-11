export find_species

include("../physics/model.jl")

const SPECIES_N             = 64   # output grid side (nc = isqrt(64) = 8 coarse cells)
const SPECIES_HM_THRESHOLD  = 14
const SPECIES_VEL_THRESHOLD = 0.1
const SPECIES_DENSITY_FLOOR = Float32(SPECIES_HM_THRESHOLD) / Float32(SPECIES_N) / 5000.0f0 * 4.0f0



function find_species(model::ParticleModel;
                      vel_threshold::Float64 = SPECIES_VEL_THRESHOLD,
                      grid_size::Int         = SPECIES_N,
                      density_threshold::Int = SPECIES_HM_THRESHOLD)

  N  = grid_size      # output grid side (e.g. 64)
  nc = isqrt(N)       # coarse grid side (e.g. 8)

  hm = heatmap(model, N, density_threshold)

  # Fine mask: true for cells inside a subdivided coarse block (non-uniform fine values)
  fine_mask = zeros(Bool, N, N)
  for cx in 1:nc, cy in 1:nc
    col0  = (cx - 1) * nc + 1
    row0  = (cy - 1) * nc + 1
    block = @view hm[col0:col0+nc-1, row0:row0+nc-1]
    all(==(block[1, 1]), block) && continue
    for fx in 1:nc, fy in 1:nc
      fine_mask[col0 + (fx - 1), row0 + (fy - 1)] = true
    end
  end

  # BFS connected-component labelling on N×N flat space
  labels    = zeros(Int32, N, N)
  num_labels = 0

  for x0 in 1:N, y0 in 1:N
    (!fine_mask[x0, y0] || hm[x0, y0] == 0.0f0 || labels[x0, y0] != 0) && continue
    num_labels += 1
    lbl = Int32(num_labels)
    labels[x0, y0] = lbl
    queue = Tuple{Int,Int}[(x0, y0)]
    head  = 1
    while head <= length(queue)
      qx, qy = queue[head]; head += 1
      for (nx, ny) in ((qx-1,qy),(qx+1,qy),(qx,qy-1),(qx,qy+1))
        (nx < 1 || nx > N || ny < 1 || ny > N) && continue
        (!fine_mask[nx, ny] || hm[nx, ny] == 0.0f0 || labels[nx, ny] != 0) && continue
        labels[nx, ny] = lbl
        push!(queue, (nx, ny))
      end
    end
  end

  # Velocity coherence filtering
  cpu_vx = Array(model.vx)
  cpu_vy = Array(model.vy)
  np     = Int(model.num_particles)
  ws     = Float32(model.WORLD_SIZE)
  inv_ws = 1.0f0 / ws

  vx_sum  = zeros(Float64, num_labels)
  vy_sum  = zeros(Float64, num_labels)
  vx2_sum = zeros(Float64, num_labels)
  vy2_sum = zeros(Float64, num_labels)
  counts  = zeros(Int,     num_labels)

  for k in 1:np
    xi  = clamp(floor(Int, model.cpu_px[k] * inv_ws * N) + 1, 1, N)
    yi  = clamp(floor(Int, model.cpu_py[k] * inv_ws * N) + 1, 1, N)
    lbl = labels[xi, yi]
    lbl == 0 && continue
    vxk = Float64(cpu_vx[k]); vyk = Float64(cpu_vy[k])
    vx_sum[lbl]  += vxk;       vy_sum[lbl]  += vyk
    vx2_sum[lbl] += vxk * vxk; vy2_sum[lbl] += vyk * vyk
    counts[lbl]  += 1
  end

  keep = trues(num_labels)
  for lbl in 1:num_labels
    c = counts[lbl]
    if c < 2; keep[lbl] = false; continue; end
    μvx = vx_sum[lbl] / c;  μvy = vy_sum[lbl] / c
    var_vx = vx2_sum[lbl] / c - μvx * μvx
    var_vy = vy2_sum[lbl] / c - μvy * μvy
    keep[lbl] = (var_vx + var_vy) <= vel_threshold
  end

  # Zero out incoherent clusters in-place, return flat N×N label matrix
  for x in 1:N, y in 1:N
    lbl = labels[x, y]
    lbl != 0 && !keep[lbl] && (labels[x, y] = Int32(0))
  end

  return labels   # Int32[N, N]
end


### HELPERS

function cluster_bounds(labels::Matrix{Int32})
  N = size(labels, 1)
  bounds = Dict{Int32, @NamedTuple{x_min::Int, x_max::Int, y_min::Int, y_max::Int, cells::Int}}()
  for x in 1:N, y in 1:N
    lbl = labels[x, y]
    lbl == 0 && continue
    if haskey(bounds, lbl)
      b = bounds[lbl]
      bounds[lbl] = (x_min=min(b.x_min,x), x_max=max(b.x_max,x),
                     y_min=min(b.y_min,y), y_max=max(b.y_max,y),
                     cells=b.cells+1)
    else
      bounds[lbl] = (x_min=x, x_max=x, y_min=y, y_max=y, cells=1)
    end
  end
  return bounds
end
