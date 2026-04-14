include("engine/sim_silicon.jl")
using CairoMakie
function sweep(R,T)
# ── Helpers ───────────────────────────────────────────────────────────────────

"""Return the mean speed √(vx²+vy²) over all particles after T steps."""
function avg_speed(attraction_matrix::Matrix{Float32}; steps=T)
    model = create_model(
        num_types        = 2,
        num_particles    = 1000,   # smaller count for speed; change as needed
        attraction_matrix = attraction_matrix,
        seed             = 0,
    )
    for _ in 1:steps
        model_step!(model)
    end
    vx = Array(model.vx)
    vy = Array(model.vy)
    return mean(sqrt.(vx .^ 2 .+ vy .^ 2))
end

# ── Sweep ─────────────────────────────────────────────────────────────────────
println("Sweeping $(R)×$(R) grid, $T steps each…")
speeds = zeros(Float32, R, R)

for xi in 1:R
    print("  row $xi / $R\r")
    flush(stdout)
    for yi in 1:R
        a = 2f0 * xi / R - 1f0   # diagonal entry   (type self-attraction)
        b = 2f0 * yi / R - 1f0   # off-diagonal entry (cross-attraction)
        mat = Float32[a b; b a]
        speeds[xi, yi] = avg_speed(mat)
    end
end
println("\nDone. Mean speed range: $(minimum(speeds)) – $(maximum(speeds))")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(size = (700, 600))
ax  = Axis(fig[1, 1];
    title  = "Average particle speed  (R=$R, T=$T steps)",
    xlabel = "x  →  self-attraction  (diagonal)",
    ylabel = "y  →  cross-attraction (off-diagonal)",
    xticks = (1:R, [i % 5 == 0 ? string(round(2i/R - 1, digits=1)) : "" for i in 1:R]),
    yticks = (1:R, [i % 5 == 0 ? string(round(2i/R - 1, digits=1)) : "" for i in 1:R]),
)

hm = heatmap!(ax, 1:R, 1:R, speeds; colormap = :viridis)
Colorbar(fig[1, 2], hm; label = "mean speed")

outpath = joinpath(@__DIR__, "$T_accel.png")
save(outpath, fig)
println("Saved to: $outpath")
end
