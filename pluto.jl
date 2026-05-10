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

# ╔═╡ b4c4e74b-46cf-43aa-93bb-9944b3aefa09
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 83d01af8-0891-4b94-846b-effffd868005
begin
	using GLMakie
	using PlutoUI
	using Random
end

# ╔═╡ a17ddf22-7734-4894-92dd-43ddefbaea36
include("engine/controller.jl")

# ╔═╡ 5267f881-7727-4fb0-b182-8dfc389f1d1b
mat = rand(Float32, 8, 8) .* 2.0f0 .- 1.0f0

# ╔═╡ 1b7e5ebb-a6ff-476d-a0f6-eef030bdd99b
# TYPE_COLORS=[colorant"#e74c3c", colorant"#3498db", colorant"#2ecc71",colorant"#f39c12", colorant"#9b59b6", colorant"#1abc9c",colorant"#e91e63", colorant"#cddc39", colorant"#00bcd4"]

# ╔═╡ 28145b66-307c-42ec-b60e-bc6b3be4d250
function frame(m)
	set_theme!(theme_black())
	f = Figure()
	ax = Axis(f[1,1])
	# points = []
	# for i in 1:m.num_particles
	# 	push!(points, Point2f(m.cpu_px[i], m.cpu_py[i]))
	# end
	i=[1,2,3,4,5,6,7,8]
	scatter!(m.cpu_px, m.cpu_py, color=map(x->i[x],get_ptypes(m)), colormap=:curl, markersize=3)
	f
end

# ╔═╡ fddee9e1-94f4-486c-81cf-2675fb94fc6b
@bind steps PlutoUI.Slider(1:1000)

# ╔═╡ 1da37a13-d9f9-4137-b01b-97dd75f652cd
begin
	m = create_model(num_types=8, dt=0.001, attraction_matrix=mat, num_particles=8000)
	for i in 1:steps
		model_step!(m)
	end
	download_positions!(m)
end

# ╔═╡ 3d5b7e28-8bd4-4ffc-be4d-c4a68ae6086e
frame(m)

# ╔═╡ Cell order:
# ╠═b4c4e74b-46cf-43aa-93bb-9944b3aefa09
# ╠═83d01af8-0891-4b94-846b-effffd868005
# ╠═a17ddf22-7734-4894-92dd-43ddefbaea36
# ╠═5267f881-7727-4fb0-b182-8dfc389f1d1b
# ╠═1b7e5ebb-a6ff-476d-a0f6-eef030bdd99b
# ╠═1da37a13-d9f9-4137-b01b-97dd75f652cd
# ╠═28145b66-307c-42ec-b60e-bc6b3be4d250
# ╠═3d5b7e28-8bd4-4ffc-be4d-c4a68ae6086e
# ╠═fddee9e1-94f4-486c-81cf-2675fb94fc6b
