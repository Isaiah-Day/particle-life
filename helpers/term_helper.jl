using Term
using .ParticleLife
Base.show(io::IO, ::MIME"text/plain", x::ParticleModel) = termshow(io, x)