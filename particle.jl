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

viscosity = 0.5
time_scale = 1.0


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
                              periodic=false,
                              # spacing=20,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = (args...) -> (@timeit to "agent_step!" agent_step!(args...)),
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)
                            ),
                            # properties=push!(Dict{Symbol, Float64}(
                            #                     lhs_rhs=>rand(range)
                            #                     for (lhs_rhs, range) in properties),
                            #                 :time_scale => 1.0)
                            properties=properties
                            )

    for c in [Red(), Green(), Orange(), Cyan()]
        for _ in 1:750
            vel =  SVector(0., 0.)
            add_agent!(model, vel, c)
        end
    end

    model
end

par_order = [:green, :red, :orange, :cyan, :yellow]


function color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM)
    index_lhs = getindex(par_order, Symbol(string(color_sym(lhs))))
    index_rhs = getindex(par_order, Symbol(string(color_sym(rhs))))
    return abmproperties(m)[rhs, lhs]
end


color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Orange) = :orange
color_sym(::Cyan) = :cyan
color_sym(::Yellow) = :yellow



# properties=OrderedDict(
#     :red_red       => -1:0.1:1,
#     :red_green     => -1:0.1:1,
#     :red_orange    => -1:0.1:1,
#     :red_cyan      => -1:0.1:1,
#     :green_red     => -1:0.1:1,
#     :green_green   => -1:0.1:1,
#     :green_orange  => -1:0.1:1,
#     :green_cyan    => -1:0.1:1,
#     :orange_red    => -1:0.1:1,
#     :orange_green  => -1:0.1:1,
#     :orange_orange => -1:0.1:1,
#     :orange_cyan   => -1:0.1:1,
#     :cyan_red      => -1:0.1:1,
#     :cyan_green    => -1:0.1:1,
#     :cyan_orange   => -1:0.1:1,
#     :cyan_cyan     => -1:0.1:1,
#     :viscosity     =>  0:.01:1,
# )

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

agent_step!(agent, model) = move_agent!(agent, model, time_scale)

function model_step!(model; to=TimerOutput())
    @timeit to "update_vel! loop" begin
        # about 20% speedup to extract the viscosity var first.
        vis::Float64 = viscosity
        @floop for agent in collect(allagents(model))
            update_vel!(agent, model; viscosity=vis)
        end
    end
    @timeit to "max reduction" begin
        max_vel = maximum(map(x->norm(x.vel), allagents(model)))
    end
    @timeit to "mean reduction" begin
        mean_vel = mean(map(x->norm(x.vel), allagents(model)))
    end
    # model.time_scale = max(0.1/max_vel, 1.)
    if max_vel*model.time_scale > getfield(model, :space).spacing || mean_vel*model.time_scale > 30
        model.time_scale /= 1.1
    end
    if model.time_scale < 0.9
        model.time_scale *= 1.01
    elseif model.time_scale > 1.1
        model.time_scale /= 1.01
    end
    @debug model.time_scale

    delta_time = time_ns() - ParticleLife.last_model_step_time
    ParticleLife.last_model_step_time = time_ns()
    fit!(ParticleLife.avg_model_step_duration[], delta_time/1e9)
    notify(ParticleLife.avg_model_step_duration)  # to update the fps label
end

function update_vel!(agent::Particle, model::ABM; viscosity::Union{Nothing, Float64}=nothing)
    force = sum(
        let g = color_interact(agent.color, other.color, model),
            d = euclidean_distance(agent, other, model)
            (0 < d < 80 ? (g / d .* (agent.pos - other.pos)) : zero(SVector{2, Float64}))
        end
        for other in Agents.nearby_agents(agent, model, 80);
        init = zero(SVector{2, Float64}),
    )
    # push away from border
    force += 0.1*(max.(40 .- agent.pos, 0)
                 - max.(40 .- (spacesize(model) - agent.pos), 0))

    # combine past velocity and current force
    agent.vel = agent.vel * (1-viscosity) + force
end
using GLMakie
function run_sim(; to=TimerOutput())
    # this is basically it, but we want to rearrange the layout a bit.
    model = make_model(to)
    fig, ax, abmobs = with_theme(theme_dark()) do
        abmplot(model;
                ac=color_sym, as=8.0,  # agent color and size
                params=ParticleLife.properties,
                scatterkwargs=(; :markerspace=>:data),
                enable_inspection=false)
    end

    # Return the plotted figure without the interactive parameter sliders
    # (the sliders expect `params` in a specific form; skip them here)
    ax.width[] = 1000
    ax.height[] = 1500
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