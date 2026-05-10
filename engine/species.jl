include("sim_silicon.jl")

const SPECIES_N             = 8    # coarse grid size — output is N²×N² = 256×256
const SPECIES_HM_THRESHOLD  = 14    # particles per coarse cell → triggers subdivision
const SPECIES_VEL_THRESHOLD = 0.1
# Minimum normalised density for a fine cell to be considered occupied.
# Fine cells in subdivided coarse blocks hold counts/num_particles, so a cell
# with even 1 particle gets a value of ~1/5000 = 0.0002.  Coarse-filled blocks
# spread their count evenly: a coarse cell with SPECIES_HM_THRESHOLD particles
# gives each fine cell SPECIES_HM_THRESHOLD / (n² * np) ≈ 14/1280000 ≈ 1e-5.
# Setting the floor just above that cuts the sparse background while keeping
# any genuinely dense fine cell.
const SPECIES_DENSITY_FLOOR = Float32(SPECIES_HM_THRESHOLD) / Float32(SPECIES_N^2) / 5000.0f0 * 4.0f0



function find_species(model::ParticleModel;
                      vel_threshold::Float64 = SPECIES_VEL_THRESHOLD)

  n  = SPECIES_N
  N  = n * n      # flat output side length (256 when n=16)

  hm = heatmap(model, n, SPECIES_HM_THRESHOLD)

  # Only label cells that live inside a subdivided (fine) coarse block.
  # Coarse-filled blocks are uniform → heatmap_find_coarse returns false for them.
  # Build a flat N×N mask: true only where the parent coarse cell was subdivided.
  is_fine = heatmap_find_coarse(hm)   # n×n Bool: true = subdivided
  fine_mask = zeros(Bool, N, N)
  for cx in 1:n, cy in 1:n
    is_fine[cx, cy] || continue
    col0 = (cx-1)*n + 1
    row0 = (cy-1)*n + 1
    for fx in 1:n, fy in 1:n
      fine_mask[col0+(fx-1), row0+(fy-1)] = true
    end
  end

  # FIND CLUSTERS 
  # Work in the flat space and translates to [fx,fy,cx,cy] at the end
  labels_2d = zeros(Int32, N, N)   # 0 = empty
  num_labels = 0

  for x0 in 1:N, y0 in 1:N
    if (!fine_mask[x0, y0] || hm[x0, y0] == 0.0f0 || labels_2d[x0, y0] != 0)
      continue
    end

    # Start a new component from this unlabelled non-zero cell
    num_labels += 1
    lbl = Int32(num_labels)
    labels_2d[x0, y0] = lbl

    queue = Tuple{Int,Int}[(x0, y0)] # don't know why i didnt do this recursively but it works lol
    head  = 1

    while head <= length(queue)
      qx, qy = queue[head]
      head   += 1

      # 4 connected neighbours
      for (nx, ny) in ((qx-1, qy), (qx+1, qy), (qx, qy-1), (qx, qy+1))
        (nx < 1 || nx > N || ny < 1 || ny > N) && continue
        (!fine_mask[nx, ny] || hm[nx, ny] == 0.0f0 || labels_2d[nx, ny] != 0) && continue
        labels_2d[nx, ny] = lbl
        push!(queue, (nx, ny))
      end
    end
  end

  # check velocity coherence 
  cpu_vx = Array(model.vx)
  cpu_vy = Array(model.vy)
  np     = Int(model.num_particles)
  ws     = Float32(model.WORLD_SIZE)
  inv_ws = 1.0f0 / ws

  # get velocity sums for each label
  vx_sum  = zeros(Float64, num_labels)
  vy_sum  = zeros(Float64, num_labels)
  vx2_sum = zeros(Float64, num_labels)
  vy2_sum = zeros(Float64, num_labels)
  counts  = zeros(Int,     num_labels)

  for k in 1:np
    # map 4d particle position to 2d 
    xi = clamp(floor(Int, model.cpu_px[k] * inv_ws * N) + 1, 1, N)
    yi = clamp(floor(Int, model.cpu_py[k] * inv_ws * N) + 1, 1, N)

    lbl = labels_2d[xi, yi]
    lbl == 0 && continue   # particle is in an empty / unlabelled region

    vx = Float64(cpu_vx[k])
    vy = Float64(cpu_vy[k])
    vx_sum[lbl]  += vx
    vy_sum[lbl]  += vy
    vx2_sum[lbl] += vx * vx
    vy2_sum[lbl] += vy * vy
    counts[lbl]  += 1
  end

  # find coherent clusters
  keep = trues(num_labels)
  for lbl in 1:num_labels
    c = counts[lbl]
    if c < 2
      keep[lbl] = false   # need at least 2 particles to measure variance
      continue
    end
    μvx = vx_sum[lbl] / c
    μvy = vy_sum[lbl] / c
    # Variance = E[v^2] - E[v]^2
    var_vx = vx2_sum[lbl] / c - μvx * μvx
    var_vy = vy2_sum[lbl] / c - μvy * μvy
    keep[lbl] = (var_vx + var_vy) <= vel_threshold
  end

  # build 4d matrix
  output = zeros(Int32, n, n, n, n)   # [fx, fy, cx, cy]

  for x in 1:N, y in 1:N
    lbl = labels_2d[x, y]
    (lbl == 0 || !keep[lbl]) && continue

    cx = (x - 1) ÷ n + 1;  fx = (x - 1) % n + 1
    cy = (y - 1) ÷ n + 1;  fy = (y - 1) % n + 1
    output[fx, fy, cx, cy] = lbl
  end

  return output
end


### HELPERS
function heatmap_find_coarse(hm::Matrix{Float32})
  N = size(hm, 1)
  n = isqrt(N)

  result = Matrix{Bool}(undef, n, n)

  for cx in 1:n, cy in 1:n
    col0 = (cx - 1) * n + 1
    row0 = (cy - 1) * n + 1
    block = @view hm[col0:col0+n-1, row0:row0+n-1]
    result[cx, cy] = !all(==(block[1, 1]), block)
  end

  return result
end



function cluster_bounds(output::Array{Int32,4}) # returns clusters bounding box given ids
  n = size(output, 3)   # coarse grid side
  bounds = Dict{Int32, @NamedTuple{cx_min::Int, cx_max::Int,
                                    cy_min::Int, cy_max::Int,
                                    cells::Int}}()

  for cx in 1:n, cy in 1:n
    for fx in 1:n, fy in 1:n
      lbl = output[fx, fy, cx, cy]
      lbl == 0 && continue

      if haskey(bounds, lbl)
        b = bounds[lbl]
        bounds[lbl] = (
          cx_min = min(b.cx_min, cx), cx_max = max(b.cx_max, cx),
          cy_min = min(b.cy_min, cy), cy_max = max(b.cy_max, cy),
          cells  = b.cells + 1,
        )
      else
        bounds[lbl] = (cx_min=cx, cx_max=cx, cy_min=cy, cy_max=cy, cells=1)
      end
    end
  end

  return bounds
end
