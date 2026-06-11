module ParticleLife

#include("analysis/organisms.jl")
include("physics/cpu.jl")
include("physics/model.jl")
include("physics/step.jl")
include("display/display.jl")
include("analysis/species.jl")

export download_positions!, heatmap, heatmap_slow, get_ptypes
export ParticleModel, create_model, randomize_matrix!, reset_particles!
export model_step!
export find_species


end