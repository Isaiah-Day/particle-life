run_window = true

module Sim
using Agents
using Random
#using GLMakie, Makie

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

timescale_default = 100


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

function make_model(;to=TimerOutput(), num_colors=4)
    extent::NTuple{2, Float64} = 3 .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=true,
                              # spacing=20,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = agent_step!,
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)),
                            properties=Dict(:attraction_matrix => create_attraction_matrix(num_colors), :time_scale => timescale_default, :hitbox => 1, :viscosity => 0.5, :max_distance => 160.0)
                            )

    for c in par_order[1:num_colors]
        for _ in 1:750
            vel =  SVector(0, 0)
            
            add_agent!(model, vel, color_sym(c))
        end
    end
    @info abmproperties(model)[:attraction_matrix]
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


# function Base.convert(::Type{Symbol}, a::String)
#     return Symbol("Attraction","Matrix")
# end
# 


function create_attraction_matrix(n::Int=5)
    return 2 .* rand(n, n) .- 1
end


function agent_step!(agent, model)
    move_agent!(agent, model, abmproperties(model)[:time_scale])
end

function model_step!(model; to=TimerOutput())
    @time begin

    @timeit to "update_vel! loop" begin
        # about 20% speedup to extract the viscosity var first.
        vis::Float64 = abmproperties(model)[:viscosity]
        hitbox = abmproperties(model)[:hitbox]
        if hitbox == 1 # prevent div by zero
          hitbox = 1.001
        end
        @floop for agent in collect(allagents(model))
            update_vel!(agent, model; vis=vis, hitbox=hitbox)
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
    delta_time = time_ns() - Sim.last_model_step_time
    Sim.last_model_step_time = time_ns()
    fit!(Sim.avg_model_step_duration[], delta_time/1e9)
    notify(Sim.avg_model_step_duration)  # to update the fps label
end
end

function update_vel!(agent::Particle, model::ABM; vis, hitbox)


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
end

#function display()
# #   model = make_model()
#
# #   fig = Figure()
# #   abmobs = ABMObservable(model)
# #   
#    
#
##    scatter(fig[1,1], 
##        map(p->p.pos[1], a),
##        map(p->p.pos[2], a),
##        color=map(p->color_sym(p.color), a)
##    )
# #   attraction_observables = map(p->Observable{Real}(p),abmproperties(model)[:attraction_matrix])
#
# #   axis = Axis(fig[1, 1])
#
# #   fig[2, 1] = control_grid = GridLayout(; tellwidth=false)
# #   buttons = control_grid[1, 1:4] = map(p->Button(fig, label = p), ["Start", "Stop", "Reset Particles", "Reset All"])
#
# #   on(buttons[1].clicks) do n
# #       @info "Starting Simulation"
# #       global run_sim = true
# #       @async begin
# #           while run_sim == true
# #             Agents.step!(abmobs)
# #             
# #           end
# #       end
# #   end
#
# 
# #   on(buttons[2].clicks) do n
# #       @info "Stopping Simulation"
# #       global run_sim = false
# #   end
#
# #   on(buttons[3].clicks) do n
# #       @info "Resetting Particles"
# #       global model
# #       model = make_model()
# #       global abmobs
# #       abmobs = ABMObservable(model)
# #   end
#
# #   on(buttons[4].clicks) do n
# #       @info "Resetting All"
# #       global model = Sim.make_model()
# #   end
#
#
#
# #       model_step!(model)    
#
#
#
# #   GLMakie.display(fig)
#
# model = make_model()
#
# fig, ax, abmobs = abmplot(model; params=Dict(:viscosity=>abmproperties(model)[:viscosity]))
# control_grid = fig[1,2] = GridLayout()
#
# on (abmobs.model) do m
#
# end
#
#fig
#
#end



end # module Sim

#Sim.display()

# ---- PLOTS APPROACH ----
# run = true
# Sim.model_step!(model)
# a = allagents(model)
# while run == true
#     a = scatter!([p.pos[1] for p in a], [p.pos[2] for p in a]; markersize=abmproperties(model)[:hitbox], markercolor=[Sim.color_sym(p) for p in a])
#     global a = allagents(model)


#     Sim.model_step!(model)

# end
#while run_window == true
#    sleep(0.5)
#end
