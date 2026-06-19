export find_organisms

using Clustering
using NearestNeighbors
function find_organisms(model::ParticleModel; eps::AbstractFloat=0.2, minpts::Int=2)
    organisms = Vector{Int}[]
    for i in 1:(model.num_types)
        par_of_type = get_particles_of_type(   model, i)
        lpar_type = length(par_of_type)
        if lpar_type > 0
            # positions = Matrix{Float32}(undef, 2, lpar_type)
            # positions[1, :] = [model.cpu_px[a] for a in par_of_type]
            # positions[2, :] = [model.cpu_py[a] for a in par_of_type]

            distance_matrix = Matrix{Float32}(undef, lpar_type, lpar_type)
            for a in 1:lpar_type, b in 1:lpar_type
                distance_matrix[a,b] = distance(model, par_of_type[a], par_of_type[b])
            end

            clusters = dbscan(distance_matrix, eps; min_neighbors=minpts).clusters

            for cluster in clusters
                # println(cluster)
                push!(
                    organisms,
                    par_of_type[vcat(cluster.core_indices, cluster.boundary_indices)],
                )
            end
        end
    end
    # println(organisms)
    return organisms
end
### HELPERS

function get_particles_of_type(model::ParticleModel, particle_type::Int)
    cpu_ptypes = Array(model.ptypes)
    return [i for i in 1:length(cpu_ptypes) if (cpu_ptypes[i] == particle_type)]
end
