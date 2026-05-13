# Particle Life
Currently only avaliable for Apple Silicon. Tested on an M1 with 8 GB ram.
NVIDIA will (hopefully) gain support over the next month.

## Building
Because this project is purely for research purposes, a binary is not included. If someone wants to make one, I'll happily accept it.

*Dependencies*

- The julia language
- OpenGL 3.3 or higher


*Instructions*
- Clone the repo
- Inside the folder, run `julia --project=. -e 'import Pkg; Pkg.instantiate()'`
- Run `julia run.jl`
