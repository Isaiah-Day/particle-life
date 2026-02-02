include("sim.jl")
using .Sim
using Debugger
using Agents
using Makie, GLMakie
using Statistics
GLMakie.activate!(
  vsync=false,
  framerate=30.0,
  float=false,
  # pause_renderloop = false,
  focus_on_show=false,
  decorated=true,
  title="Makie"
)



function getDensity(model)
  s = abmspace(model).spacing
  mo = zeros(Int, trunc(Int,s^2))
  for a in allagents(model)
    index = ceil.(Int,a.pos./50)
    mo[trunc(Int,index.x+index.y*s)] += 1
  end
  return var(mo)
end

function func()
dim = 100
fractal = zeros(Int, (dim,dim))
for i in 1:dim
  @info i
  for j in 1:dim
    model = make_model(; num_colors=2, matrix=[i/dim j/dim;j/dim i/dim], space_size=0.5)
    run!(model,200)
    fractal[i,j] = trunc(getDensity(model))
  end
end
fig = Figure()
ax = Axis(fig[1,1])
print(fractal)
hmap = heatmap!(ax, 1:dim, 1:dim, fractal)
wait(display(fig))
end
func()
