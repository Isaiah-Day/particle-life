export model_step!
using KernelAbstractions

@kernel function _compute_cells_kernel!(
    cell_ids::AbstractVector{Int32},
    px::AbstractVector{Float32},
    py::AbstractVector{Float32},
    n::Int32,
    inv_max_r::Float32,
    grid_w::Int32,
    grid_h::Int32,
)
    i = @index(Global, Linear)
    if i <= n
        cx = min(unsafe_trunc(Int32, px[i] * inv_max_r), grid_w - Int32(1))
        cy = min(unsafe_trunc(Int32, py[i] * inv_max_r), grid_h - Int32(1))
        cell_ids[i] = cx + cy * grid_w
    end
end

@kernel function _force_kernel!(
    fx::AbstractVector{Float32},
    fy::AbstractVector{Float32},
    px::AbstractVector{Float32},
    py::AbstractVector{Float32},
    ptypes::AbstractVector{Int32},
    sorted_order::AbstractVector{Int32},
    cell_start::AbstractVector{Int32},
    cell_end::AbstractVector{Int32},
    attraction::AbstractVector{Float32},
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
    i = @index(Global, Linear)

    if i <= n
        pxi = px[i]
        pyi = py[i]
        ti = ptypes[i]
        fxi = 0.0f0
        fyi = 0.0f0

        cx = min(unsafe_trunc(Int32, pxi / max_r), grid_w - Int32(1))
        cy = min(unsafe_trunc(Int32, pyi / max_r), grid_h - Int32(1))

        for dy in Int32(-1):Int32(1)
            for dx in Int32(-1):Int32(1)
                nx = mod(cx + dx, grid_w)
                ny = mod(cy + dy, grid_h)
                cell = nx + ny * grid_w + Int32(1)

                kstart = cell_start[cell]
                kend = cell_end[cell]
                kstart > kend && continue

                for k in kstart:kend
                    j = sorted_order[k]

                    dx2 = px[j] - pxi
                    dy2 = py[j] - pyi

                    if dx2 > half_w
                        dx2 -= world_sz
                    end
                    if dx2 < -half_w
                        dx2 += world_sz
                    end
                    if dy2 > half_w
                        dy2 -= world_sz
                    end
                    if dy2 < -half_w
                        dy2 += world_sz
                    end

                    dist2 = dx2 * dx2 + dy2 * dy2
                    (dist2 < 1.0f-8 || dist2 >= max_r_sq) && continue

                    inv_dist = 1.0f0 / sqrt(dist2)
                    dist = dist2 * inv_dist

                    tj = ptypes[j]
                    a = attraction[(tj - Int32(1)) * num_types + ti]

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
    end
end

@kernel function _integrate_kernel!(
    px::AbstractVector{Float32},
    py::AbstractVector{Float32},
    vx::AbstractVector{Float32},
    vy::AbstractVector{Float32},
    fx::AbstractVector{Float32},
    fy::AbstractVector{Float32},
    n::Int32,
    dt::Float32,
    damping::Float32,
    world_sz::Float32,
)
    i = @index(Global, Linear)

    if i <= n
        nvx = (vx[i] + fx[i] * dt) * damping
        nvy = (vy[i] + fy[i] * dt) * damping
        nx = px[i] + nvx * dt
        ny = py[i] + nvy * dt

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
    end
end

const _compute_cells_kernel_inst = _compute_cells_kernel!(backend, TILE_SIZE)
const _force_kernel_inst = _force_kernel!(backend, TILE_SIZE)
const _integrate_kernel_inst = _integrate_kernel!(backend, TILE_SIZE)

function model_step!(model::ParticleModel)
    # Precompute
    max_r = model.max_radius
    max_r_sq = max_r * max_r
    beta = model.min_radius / max_r
    inv_beta = 1.0f0 / beta
    inv_mr = 1.0f0 / max_r
    mid = (1.0f0 + beta) * 0.5f0
    inv_hmb = 2.0f0 / (1.0f0 - beta)
    damping = 1.0f0 - model.friction

    if model.attr_dirty
        copyto!(model.gpu_attr, vec(model.attraction_matrix))
        model.attr_dirty = false
    end

    n = model.num_particles
    ws = Float32(model.WORLD_SIZE)
    ndrange = model.groups * TILE_SIZE

    gw = model.grid_w
    gh = model.grid_h

    _compute_cells_kernel_inst(
        model.gpu_cell_ids, model.px, model.py, n, inv_mr, gw, gh; ndrange=ndrange
    )

    copyto!(model.cpu_cell_ids, model.gpu_cell_ids)
    sortperm!(model.cpu_sorted_order, model.cpu_cell_ids)
    copyto!(model.gpu_sorted_order, model.cpu_sorted_order)

    ncells = Int(gw) * Int(gh)
    fill!(model.cpu_cell_start, Int32(0))
    fill!(model.cpu_cell_end, Int32(0))
    prev_cell = Int32(-1)
    for k in 1:Int(n)
        cell = model.cpu_cell_ids[model.cpu_sorted_order[k]] + Int32(1)
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
    copyto!(model.gpu_cell_end, model.cpu_cell_end)

    _force_kernel_inst(
        model.gpu_fx,
        model.gpu_fy,
        model.px,
        model.py,
        model.ptypes,
        model.gpu_sorted_order,
        model.gpu_cell_start,
        model.gpu_cell_end,
        model.gpu_attr,
        n,
        model.num_types,
        max_r,
        max_r_sq,
        ws * 0.5f0,
        ws,
        model.force_scale,
        beta,
        inv_beta,
        inv_mr,
        mid,
        inv_hmb,
        gw,
        gh;
        ndrange=ndrange,
    )

    _integrate_kernel_inst(
        model.px,
        model.py,
        model.vx,
        model.vy,
        model.gpu_fx,
        model.gpu_fy,
        n,
        model.dt,
        damping,
        ws;
        ndrange=ndrange,
    )

    model.step_count += 1
    return nothing
end