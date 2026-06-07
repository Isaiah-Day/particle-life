include("engine/controller.jl")
include("engine/display.jl")
import .ParticleLife

ParticleLife.display(ParticleLife.create_model())
