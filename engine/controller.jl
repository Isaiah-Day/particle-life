module ParticleLife

#include("sim_cuda.jl")
include("sim_silicon.jl")
include("display.jl")



export create_model, model_step!, randomize_matrix!, reset_particles!, download_positions!, get_ptypes, display


end


let mat = Float32[
  0 1 0 1 0 0 0 0 0;
  0 0 1 0 1 0 0 0 0;
  1 0 0 0 0 1 0 0 0;
  0 0 0 0 1 0 1 0 0;
  0 0 0 0 0 1 0 1 0;
  0 0 0 1 0 0 0 0 1;
  1 0 0 0 0 0 0 1 0;
  0 1 0 0 0 0 0 0 1;
  0 0 1 0 0 0 1 0 0;
]
  ParticleLife.display(ParticleLife.create_model(num_types=9, attraction_matrix=mat))
end
