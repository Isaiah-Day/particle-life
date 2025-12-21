# starts around 0.18-0.35

module ParticleLife
using Agents
using Random
import StatsBase: middle
import StaticArraysCore: SVector, MVector
using FLoops
using DataStructures: OrderedDict
import LinearAlgebra: norm
using Makie, Makie.Observables
using OnlineStats
import DataStructures: DefaultDict
using ThreadPinning
using TimerOutputs

export agent_step!, make_model, color_sym, run_sim

timescale_default = 10.0


# define some module-local vars for tracking fps
last_model_step_time::UInt64 = time_ns()
avg_model_step_duration::Observable{Mean{Float64}} = Observable(Mean(weight=HarmonicWeight(30)))

abstract type ParticleColor end
struct Red <: ParticleColor end
struct Green <: ParticleColor end
struct Yellow <: ParticleColor end
struct Cyan <: ParticleColor end
struct Orange <: ParticleColor end

@agent struct Particle(ContinuousAgent{2, Float64})
    color::ParticleColor
end

function make_model(to=TimerOutput())
    extent::NTuple{2, Float64} = 3 .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=true,
                              # spacing=20,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = agent_step!,
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)),
                            properties=Dict(:attraction_matrix => properties, :time_scale => timescale_default, :hitbox => 0.5, :viscosity => 0.9, :max_distance => 80.0)
                            )

    for c in [Red(), Green(), Orange(), Cyan()]
        for _ in 1:750
            vel =  SVector(0, 0)
            
            add_agent!(model, vel, c)
        end
    end
    
    model
end

par_order = [:green, :red, :orange, :cyan, :yellow]


function color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM)
    # index_lhs = getindex(par_order, color_sym(lhs))
    index_lhs = findfirst(==(color_sym(lhs)), par_order)
    index_rhs = findfirst(==(color_sym(rhs)), par_order)


    return abmproperties(m)[:attraction_matrix][index_rhs, index_lhs]
end


color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Orange) = :orange
color_sym(::Cyan) = :cyan
color_sym(::Yellow) = :yellow

function Base.convert(::Type{Symbol}, a::String)
    return Symbol("Attraction","Matrix")
end



function create_attraction_matrix(n::Int=5)
    return 2 .* rand(n, n) .- 1
end

function add_particle(mat::AbstractMatrix{<:Real})
    n = size(mat, 1)
    @assert n == size(mat, 2) "Matrix must be square"
    new = zeros(eltype(mat), n+1, n+1)
    new[1:n, 1:n] .= mat
    new[n+1, 1:n] .= 2 .* rand(n) .- 1    # interactions FROM new particle to others
    new[1:n, n+1] .= 2 .* rand(n) .- 1    # interactions TO new particle from others
    new[n+1, n+1] = 2 * rand() - 1        # self interaction
    return new
end

function randomize!(mat::AbstractMatrix{Float64})
    mat .= 2 .* rand(size(mat)...) .- 1
    return mat
end


properties = create_attraction_matrix()

function agent_step!(agent, model)
    move_agent!(agent, model, abmproperties(model)[:time_scale])
end

function model_step!(model; to=TimerOutput())
    @time begin
    #   @floop for agent in collect(allagents(model))
    #         # keep agents inside the bounds
    #         for i in 1:2
    #             if agent.pos[i] < 0
    #                 agent.pos[i] = 0.1
    #                 agent.vel[i] = -agent.vel[i]
    #             elseif agent.pos[i] > spacesize(model)[i]
    #                 agent.pos[i] = spacesize(model)[i] - 0.1
    #                 agent.vel[i] = -agent.vel[i]    
    #             end
    #         end
    #   end 

    @timeit to "update_vel! loop" begin
        # about 20% speedup to extract the viscosity var first.
        vis::Float64 = abmproperties(model)[:viscosity]
        @floop for agent in collect(allagents(model))
            update_vel!(agent, model; vis=vis)
        end
    end
    speeds = map(x->norm(x.vel), allagents(model))
    @timeit to "max reduction" begin
        max_vel = maximum(speeds)
    end
    @timeit to "mean reduction" begin
        mean_vel = mean(speeds)
    end
    # model.time_scale = max(0.1/max_vel, 1.)

    # if maximum velocity exceeds grid spacing, reduce time scale
    if max_vel*abmproperties(model)[:time_scale] > getfield(model, :space).spacing || mean_vel*abmproperties(model)[:time_scale] > 30
        abmproperties(model)[:time_scale] /= 1.1
    end
    # if time scale is too low, slowly increase it
    if abmproperties(model)[:time_scale] < timescale_default*0.9
        abmproperties(model)[:time_scale] *= 1.01
    elseif abmproperties(model)[:time_scale] > timescale_default*1.1
        abmproperties(model)[:time_scale] /= 1.01
    end
    @info "Time Scale" abmproperties(model)[:time_scale]

    delta_time = time_ns() - ParticleLife.last_model_step_time
    ParticleLife.last_model_step_time = time_ns()
    fit!(ParticleLife.avg_model_step_duration[], delta_time/1e9)
    notify(ParticleLife.avg_model_step_duration)  # to update the fps label
end
end

function update_vel!(agent::Particle, model::ABM; vis)

    hitbox = abmproperties(model)[:hitbox]


    
    # force = sum(
    #     let g = color_interact(agent.color, other.color, model),
    #         d = sqrt(sum((agent.pos .- other.pos).^2))
    #         (d<abmproperties(model)[:max_distance] ? (0 < d < hitbox ? d/(hitbox-1) : g*(1-abs(1 + hitbox - 2 * d)/(1-hitbox))) .* (agent.pos - other.pos) : zero(SVector{2, Float64}))
    #     end
    #     for other in Agents.nearby_agents(agent, model, 80);
    #     init = SVector(0.,0.),
    # )
    force = zero(SVector{2, Float64})
    for other in Agents.nearby_agents(agent,model, 80)
        # println("AGENT")
        g = color_interact(agent.color, other.color, model)
        d = sqrt(sum((agent.pos .- other.pos).^2))
        if d < abmproperties(model)[:max_distance]
            if 0 < d < hitbox
                force += d/(hitbox-1) .* (agent.pos - other.pos)
                # println("A")
                # println(d/(hitbox-1) .* (agent.pos - other.pos))
            else 
                force += g*(1-abs(1 + hitbox - 2 * d)/(1-hitbox)) .* (agent.pos - other.pos)
                # println("B")
                # println(1-abs(1 + hitbox - 2 * d)/(1-hitbox))
            end
        else
            force += SVector(0.,0.)
            # println("C")
            
        end
        
    end
    # println(force)


    # combine past velocity and current force
    agent.vel = agent.vel * vis + force
    # println(agent)

end
using GLMakie


# Custom agent2string to avoid dimension mismatch in inspector
function Agents.agent2string(model::ABM, pos)
    return "LMAO"
    print("LALALAND")
    ids = nearby_agents(Particle(pos=pos, color=Red()), model, 0.00001)

    print(ids)
    if isempty(ids)
        return "No agents at $(pos)"
    end
    agent_info = []
    for id in ids
        agent = model[id]
        push!(agent_info, "ID: $id, Color: $(color_sym(agent)), Vel: $(agent.vel)")
    end
    return join(agent_info, "\n")
    return nearby_ids(pos, model, r=0.001) |> 
        x -> isempty(x) ? "No agents at $(pos)" : join(["ID: $id, Color: $(color_sym(model[id])), Vel: $(model[id].vel)" for id in x], "\n")

end

# function Agents.agent2string(model::ABM, pos)
#     ids = nearby_ids(model, pos)
#     s = ""

#     for (i, id) in enumerate(ids)
#         if i > 1
#             s *= "\n"
#         end
#         s *= Agents.agent2string(model[id])
#     end

#     return s
# end

function run_sim(; to=TimerOutput())
    model = make_model(to)
    
    params = Dict(
        :timescale => abmproperties(model)[:time_scale],
    )

    fig, ax= abmexploration(model; 
                           params,
                            ac=color_sym, 
                           as=12.0,
                           scatterkwargs=(; :markerspace=>:data))
    
    fig
end

function make_video()
    with_theme(theme_dark()) do
        abmvideo("/tmp/foo.mp4", make_model(), ac=color_sym, as=8.0;
                scatterkwargs=(; :markerspace=>:data))
    end
end

end # module ParticleLife

ParticleLife.run_sim()