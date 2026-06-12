### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 2cb162ec-4dae-11f1-bd0c-a340efd45943
begin
    using Pkg: Pkg
    Pkg.activate(".")
end

# ╔═╡ 5d1276ab-d93b-45c2-8032-ce8aa9e28c57
begin
    using GLMakie
    using PlutoUI
    using Random
end

# ╔═╡ fcd9f83e-c6b6-4207-ae35-b28264d6f152
const ParticleLife = include("engine/controller.jl")

# ╔═╡ 0eaa29a4-4258-4a12-9077-fc74a7323a15
mat = rand(Float32, 8, 8) .* 2.0f0 .- 1.0f0
# TYPE_COLORS=[colorant"#e74c3c", colorant"#3498db", colorant"#2ecc71",colorant"#f39c12", colorant"#9b59b6", colorant"#1abc9c",colorant"#e91e63", colorant"#cddc39", colorant"#00bcd4"]

# ╔═╡ 20970d94-1943-4709-b984-f24773af70ec
@bind steps PlutoUI.Slider(1:800)

# ╔═╡ 192d3a38-4b15-4a14-8c02-f828be1c934b
md"Steps [$(steps)]"

# ╔═╡ 344a04af-eae4-4255-b094-ecb90b3aab05
md"### Species Detection"

# ╔═╡ 268d9d84-8208-4a6c-81c0-f49d07f9a99c
@bind vel_threshold PlutoUI.Slider(0:0.01:1)

# ╔═╡ 87113a1d-92b7-40fc-aafe-01c4a8f5f180
md"Vel Threshold [$(vel_threshold)]"

# ╔═╡ 3d53f1c5-2b82-49d9-9a90-175778ce488c
@bind density_threshold PlutoUI.Slider(1:50)

# ╔═╡ a7f1c86f-dada-46a5-8f55-00388747a0ff
md"Density Threshold [$(density_threshold)]"

# ╔═╡ 4076834f-339c-411c-97f8-31ddf219c566
@bind grid_size PlutoUI.Slider([8, 16, 32, 64, 128])

# ╔═╡ 6d02a7d4-6f1c-41da-8ae0-cd22a845e33c
function frame(m)
    set_theme!(theme_black())
    f = Figure()
    ax = Axis(f[1, 1])
    # points = []
    # for i in 1:m.num_particles
    # 	push!(points, Point2f(m.cpu_px[i], m.cpu_py[i]))
    # end

    spec = ParticleLife.find_species(
        m;
        grid_size=grid_size,
        density_threshold=density_threshold,
        vel_threshold=vel_threshold,
    )
    GLMakie.heatmap!(ax, 0 .. m.WORLD_SIZE, 0 .. m.WORLD_SIZE, spec; alpha=1)

    i = [1, 2, 3, 4, 5, 6, 7, 8]
    scatter!(
        ax,
        m.cpu_px,
        m.cpu_py;
        color=map(x -> i[x], ParticleLife.get_ptypes(m)),
        colormap=:curl,
        markersize=3,
    )

    # lines!(ax, Point2i(0,0), Point2i(0,m.WORLD_SIZE))7

    1 + 1

    return f
end

# ╔═╡ e9b20b6a-a1bb-4861-b1e4-e26f03911bac
md"Grid Size [$(grid_size)]"

# ╔═╡ d141665a-0c57-4966-b225-a62828984fce
md"""
0.80

20

16
"""

# ╔═╡ e4c7df1b-aa4d-426b-8ecb-12402a8e7372
begin
    m500 = ParticleLife.create_model(;
        num_types=6, dt=0.001, attraction_matrix=mat, num_particles=10000, world_size=1.0f0
    )
    for i in 1:500
        ParticleLife.model_step!(m500)
    end
end

# ╔═╡ deb309eb-3443-4a97-bf15-f40041ed7f86
begin
    if steps < 500
        m = ParticleLife.create_model(;
            num_types=6,
            dt=0.001,
            attraction_matrix=mat,
            num_particles=10000,
            world_size=1.0f0,
        )
        for i in 1:steps
            ParticleLife.model_step!(m)
        end
    else
        m2 = deepcopy(m500)
        for i in 1:(steps - 500)
            ParticleLife.model_step!(m2)
        end
    end
    # end
    (steps < 500) && ParticleLife.download_positions!(m)
    (steps >= 500) && ParticleLife.download_positions!(m2)

    # end
end

# ╔═╡ 3e861727-8f63-49af-9691-6f4c0287339e
(steps >= 500) && frame(m2)

# ╔═╡ ba0f2c3c-12f1-479b-a388-2155cefc36e6
(steps < 500) && frame(m)

# ╔═╡ Cell order:
# ╠═2cb162ec-4dae-11f1-bd0c-a340efd45943
# ╠═5d1276ab-d93b-45c2-8032-ce8aa9e28c57
# ╠═fcd9f83e-c6b6-4207-ae35-b28264d6f152
# ╠═0eaa29a4-4258-4a12-9077-fc74a7323a15
# ╠═6d02a7d4-6f1c-41da-8ae0-cd22a845e33c
# ╟─192d3a38-4b15-4a14-8c02-f828be1c934b
# ╟─20970d94-1943-4709-b984-f24773af70ec
# ╟─344a04af-eae4-4255-b094-ecb90b3aab05
# ╟─87113a1d-92b7-40fc-aafe-01c4a8f5f180
# ╟─268d9d84-8208-4a6c-81c0-f49d07f9a99c
# ╟─a7f1c86f-dada-46a5-8f55-00388747a0ff
# ╟─3d53f1c5-2b82-49d9-9a90-175778ce488c
# ╟─e9b20b6a-a1bb-4861-b1e4-e26f03911bac
# ╟─4076834f-339c-411c-97f8-31ddf219c566
# ╟─d141665a-0c57-4966-b225-a62828984fce
# ╟─3e861727-8f63-49af-9691-6f4c0287339e
# ╟─ba0f2c3c-12f1-479b-a388-2155cefc36e6
# ╟─deb309eb-3443-4a97-bf15-f40041ed7f86
# ╠═e4c7df1b-aa4d-426b-8ecb-12402a8e7372
