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
model = make_model(;num_colors=3, space_size=0.5)
NUM_COLORS = 3








function display(model)

  fig, ax, abmobs = with_theme(theme_dark()) do 
    abmplot(model; ac=color_sym, as=5.0, enable_inspection=false, params=Dict(:viscosity => abmproperties(model).viscosity))
  end



  m = abmproperties(model).attraction_matrix



  controls = fig[2,2]

  matrix = fig[1,2]

  #for i in 1:length(NUM_COLORS)
  #  Textbox(matrix[i,1], placeholder=string(NUM_COLORS[i]))
  #  Textbox(matrix[1,i], placeholder=string(NUM_COLORS[i]))
  #end
  displaym = [Textbox(matrix[i+1,j+1], placeholder=string(m[i,j]), validator=Float64) for i=1:size(m)[1], j=1:size(m)[1]]
  



  controls[1,1] = content(fig[2,1][1,1])


  Makie.deleterow!(fig.layout, 2)
  wait(Makie.display(fig))
end
display(make_model())

