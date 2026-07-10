using CairoMakie
using ParticleLife

function frame(m::ParticleModel)
    @async begin
        download_positions!(m)
        set_theme!(theme_black())
        f = Figure()
        ax = Axis(f[1, 1])

        scatter!(
            ax,
            m.cpu_px,
            m.cpu_py;
            color=ParticleLife.get_ptypes(m),
            colormap=:curl,
            markersize=3,
        )
        return display(f)
    end
end