module Sim
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



export agent_step!, make_model, color_sym, create_attraction_matrix, model_step!, par_order

timescale_default = 0.05
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
    steps_in_hitbox::Integer
end



@kwdef mutable struct Parameters
  attraction_matrix::Matrix{Float64}
  time_scale::Float64 = timescale_default
  hitbox::Float64 = 0.3
  viscosity::Float64 = 0.50
  max_distance::Float64 = 10
end

function make_model(;to=TimerOutput(), num_colors=4, space_size=3, matrix=create_attraction_matrix(num_colors))
    extent::NTuple{2, Float64} = space_size .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=true,
                              spacing=10,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = agent_step!,
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)),
                            properties=(Parameters(attraction_matrix=matrix))
                            )

    for c in par_order[1:num_colors]
      for _ in 1:(300*3)
            vel =  SVector(0, 0)
            
            add_agent!(model, vel, color_sym(c), 0)
        end
    end
    model
end

par_order = [:green, :red, :orange, :cyan, :yellow]


function color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM)
    # index_lhs = getindex(par_order, color_sym(lhs))
    index_lhs = findfirst(==(color_sym(lhs)), par_order)
    index_rhs = findfirst(==(color_sym(rhs)), par_order)

    return abmproperties(m).attraction_matrix[index_rhs, index_lhs]
end


color_sym(p::Particle) = color_sym(p.color)
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Orange) = :orange
color_sym(::Cyan) = :cyan
color_sym(::Yellow) = :yellow

function color_sym(a::Symbol)
  if a == :green
    return Green()
  elseif a == :red
      return Red()
  elseif a == :orange
      return Orange()
  elseif a == :cyan
    return Cyan()
  elseif a == :yellow
    return Yellow()
  end
end




function create_attraction_matrix(n::Int=5)
    return 2 .* rand(n, n) .- 1
end


function agent_step!(agent, model)
  move_agent!(agent, model, abmproperties(model).time_scale)
end

function model_step!(model; to=TimerOutput())

  #@time begin
    @timeit to "update_vel! loop" begin
        # about 20% speedup to extract the viscosity var first.
        vis::Float64 = abmproperties(model).viscosity
        hitbox001 = abmproperties(model).hitbox
        interaction_max = abmproperties(model).max_distance
        Threads.@threads for agent in collect(allagents(model))
            update_vel!(agent, model; vis=vis, hitbox=hitbox001, interaction_max=interaction_max)
        end
    end
    speeds = map(x->norm(x.vel), allagents(model))
  #end
    @timeit to "max reduction" begin
          max_vel = maximum(speeds)
    end
    
    @timeit to "mean reduction" begin
        mean_vel = mean(speeds)
    end

    # if maximum velocity exceeds grid spacing, reduce time scale
    #if max_vel*abmproperties(model).time_scale > getfield(model, :space).spacing || mean_vel*abmproperties(model).time_scale > 30
    #    abmproperties(model).time_scale /= 1.5
    #end
    if mean_vel*abmproperties(model).time_scale > 0.1
        abmproperties(model).time_scale /= 1.5
    end


    # if time scale is too low, slowly increase it
    if abmproperties(model).time_scale < timescale_default*0.9
        abmproperties(model).time_scale *= 1.001
    elseif abmproperties(model).time_scale > timescale_default*1.1
        abmproperties(model).time_scale /= 1.001
    end
    
    #delta_time = time_ns() - Sim.last_model_step_time
    #Sim.last_model_step_time = time_ns()
end

function update_vel!(agent::Particle, model::ABM; vis, hitbox, interaction_max)
    force = zero(SVector{2, Float64})
    for other in Agents.nearby_agents(agent,model, interaction_max)
        a = color_interact(agent.color, other.color, model)
        distance = sqrt(sum((agent.pos .- other.pos).^2))/interaction_max

        if distance < hitbox
          force += (distance/hitbox + 1) .* (agent.pos - other.pos)
        elseif hitbox < distance < 1
          force += (a * (1- abs(2*distance-1-hitbox)/(1-hitbox))) .* (agent.pos - other.pos)
        end
    end

    agent.vel = agent.vel * vis + force * interaction_max
end

function Agents.agent2string(model::ABM, pos) # because inspector is a bitch
    return "LMAO"
end



end
