using Makie, Agents, TimerOutputs
# import StaticArraysCore: SVector, MVector
using DataStructures: OrderedDict



abstract type ParticleColor end
struct Red <: ParticleColor end
struct Green <: ParticleColor end
struct Yellow <: ParticleColor end
struct Cyan <: ParticleColor end
struct Orange <: ParticleColor end

@agent struct Particle(ContinuousAgent{2, Float64})
    color::ParticleColor
end

color_interact(lhs::ParticleColor,  rhs::ParticleColor, m::ABM) =
    abmproperties(m)[Symbol(string(color_sym(lhs))*'_'*string(color_sym(rhs)))]

color_sym(p::Particle) = color_sym(p.color)
color_sym(::ParticleColor) = :black
color_sym(::Green) = :green
color_sym(::Red) = :red
color_sym(::Orange) = :orange
color_sym(::Cyan) = :cyan
color_sym(::Yellow) = :yellow

properties=OrderedDict(
    :red_red       => -1:0.1:1,
    :red_green     => -1:0.1:1,
    :red_orange    => -1:0.1:1,
    :red_cyan      => -1:0.1:1,
    :green_red     => -1:0.1:1,
    :green_green   => -1:0.1:1,
    :green_orange  => -1:0.1:1,
    :green_cyan    => -1:0.1:1,
    :orange_red    => -1:0.1:1,
    :orange_green  => -1:0.1:1,
    :orange_orange => -1:0.1:1,
    :orange_cyan   => -1:0.1:1,
    :cyan_red      => -1:0.1:1,
    :cyan_green    => -1:0.1:1,
    :cyan_orange   => -1:0.1:1,
    :cyan_cyan     => -1:0.1:1,
    :viscosity     =>  0:.01:1,
)



function initialize_model(to=TimerOutput())
    extent::NTuple{2, Float64} = 3 .* (500.0, 500.0);
    space2d = ContinuousSpace(extent;
                              periodic=false,
                              # spacing=20,
                              );
    model = AgentBasedModel(Particle, space2d;
                            agent_step! = (args...) -> (@timeit to "agent_step!" agent_step!(args...)),
                            model_step! = (args...) -> (@timeit to "model_step!" model_step!(args...; to=to)),
                            properties=push!(Dict{Symbol, Float64}(
                                    lhs_rhs=>rand(range)
                                    for (lhs_rhs, range) in properties),
                                :time_scale => 1.0)
                            )

    for c in [Red(), Green(), Orange(), Cyan()]
        for _ in 1:750
            vel =  SVector(0., 0.)
            add_agent!(model, vel, c)
        end
    end

    model
end

agent_step!(agent, model) = move_agent!(agent, model, 0.5)

function model_step!(model; to=TimerOutput())
  
end

function make_video()
    with_theme(theme_dark()) do
        abmvideo("/tmp/foo.mp4", initialize_model(), ac=color_sym, as=8.0;
                scatterkwargs=(; :markerspace=>:data))
    end
end
make_video()