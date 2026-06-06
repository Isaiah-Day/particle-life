# tune_tile_size.jl
# Benchmarks _force_kernel! across TILE_SIZE candidates.
# Run with: julia --project=. tune_tile_size.jl

using Statistics, Printf, Metal

const CANDIDATES    = [32, 64, 128, 256, 512]
const N_WARMUP      = 20
const N_BENCH       = 100
const NUM_PARTICLES = 5000

const SIM_FILE = joinpath(@__DIR__, "engine", "sim_silicon.jl")

function patch_tile_size(src::String, tile::Int)::String
  replace(src, r"^const TILE_SIZE = \d+"m => "const TILE_SIZE = $tile")
end

src_original = read(SIM_FILE, String)
results = Tuple{Int, Float64, Float64}[]

for tile in CANDIDATES
  println("\n=== TILE_SIZE = $tile ===")

  tmp = tempname() * ".jl"
  write(tmp, patch_tile_size(src_original, tile))

  mod = Module(Symbol("Sim_$tile"))

  try
    Core.eval(mod, :(using Metal, Random))
    Base.include(mod, tmp)

    # invokelatest required on Julia 1.12: bindings defined inside
    # Base.include are in a newer world than the calling scope.
    model = Base.invokelatest(getfield(mod, :create_model);
                              num_particles=NUM_PARTICLES)
    step! = Base.invokelatest(getfield, mod, :model_step!)

    print("  warming up ($N_WARMUP steps)... ")
    for _ in 1:N_WARMUP
      Base.invokelatest(step!, model)
    end
    Metal.synchronize()
    println("done")

    times = Vector{Float64}(undef, N_BENCH)
    for i in 1:N_BENCH
      t0 = time_ns()
      Base.invokelatest(step!, model)
      Metal.synchronize()
      times[i] = (time_ns() - t0) * 1e-6
    end

    med = median(times)
    sd  = std(times)
    push!(results, (tile, med, sd))
    @printf("  median %.3f ms  |  std %.3f ms  |  min %.3f ms  |  max %.3f ms\n",
            med, sd, minimum(times), maximum(times))

  catch e
    println("  ERROR: ", e)
  finally
    rm(tmp; force=true)
  end
end

println("\n\n─────────────────────────────────────")
println(" TILE_SIZE benchmark summary")
println(" n = $NUM_PARTICLES particles, $N_BENCH steps each")
println("─────────────────────────────────────")
@printf(" %-12s  %-12s  %-10s\n", "TILE_SIZE", "median (ms)", "std (ms)")
println("─────────────────────────────────────")

let best_tile = 0, best_med = Inf
  for (tile, med, sd) in results
    if med < best_med
      best_med  = med
      best_tile = tile
    end
    @printf(" %-12d  %-12.3f  %-10.3f\n", tile, med, sd)
  end
  println("─────────────────────────────────────")
  println(" Best: TILE_SIZE = $best_tile  ($(round(best_med, digits=3)) ms/step)")
  println("\nTo apply: set `const TILE_SIZE = $best_tile` in engine/sim_silicon.jl")
end
