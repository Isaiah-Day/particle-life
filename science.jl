include("sim.jl")
include("display.jl")
using .Sim
using Agents
using Makie, GLMakie

function sort(divisions, m)
  sorted = [Vector{Any}() for _ in 1:divisions, _ in 1:divisions]
  extent = spacesize(m)[1]
  for a in allagents(m)
    x = 1+trunc(Int, divisions*a.pos[1]/extent)
    y = 1+trunc(Int, divisions*a.pos[2]/extent)
    push!(sorted[y,x], a)
  end
  sorted
end


function per_frame(model)
  return
end

model = make_model(per_frame=per_frame)

p = sort(4, model)[1,1]

fig, ax, abmobs = with_theme(theme_dark()) do 
  abmplot(model; ac=color_sym, as=8.0, enable_inspection=false, add_controls=true, params=Dict(:viscosity => abmproperties(model).viscosity) )
end

# size = spacesize(model)[1]
# ablines!(ax, [0,size/4,2*size/4, 3*size/4,size],[0,0,0,0,0])
# vlines!(ax, [1,size/4,2*size/4, 3*size/4,size])
# 

display(model)
wait(display(fig))
