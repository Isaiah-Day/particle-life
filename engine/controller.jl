module ParticleLife

#include("sim_cuda.jl")
include("sim_silicon.jl")
#include("display.jl")
include("species.jl")

export create_model, model_step!, randomize_matrix!, reset_particles!, download_positions!, get_ptypes, find_species

end



#
#let mat = Float32[
#  0 1 0 1 0 0 0 0 0;
#  0 0 1 0 1 0 0 0 0;
#  1 0 0 0 0 1 0 0 0;
#  0 0 0 0 1 0 1 0 0;
#  0 0 0 0 0 1 0 1 0;
#  0 0 0 1 0 0 0 0 1;
#  1 0 0 0 0 0 0 1 0;
#  0 1 0 0 0 0 0 0 1;
#  0 0 1 0 0 0 1 0 0;
#]
#ParticleLife.display(ParticleLife.create_model())
#end
