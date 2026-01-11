include("sim.jl")
using .Sim
using Agents
using Makie, GLMakie
GLMakie.activate!(
    vsync = false,
    framerate = 30.0,
    float = false,
    # pause_renderloop = false,
    focus_on_show = false,
    decorated = true,
    title = "Makie"
)
model = make_model(;num_colors=5)
print(model)


fig, ax, abmobs = with_theme(theme_dark()) do 
  abmplot(model; ac=color_sym, as=1.0, enable_inspection=false, add_controls=true, params=Dict(:viscosity => abmproperties(model)[:viscosity]) )
end



wait(display(fig))
