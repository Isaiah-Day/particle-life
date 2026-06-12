using GLMakie
using ParticleLife

function frame(m)
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

function frame_org(m)
    set_theme!(theme_black())
    f = Figure()
    ax = Axis(f[1, 1])

    organisms = find_organisms(m; eps=0.2)
    organisms_new = zeros(len(m.cpu_px))

    index = 1
    for list in organisms
        for el in list
            organisms_new[el] = index
        end
        index += 1
    end

    scatter!(
        ax,
        m.cpu_px,
        m.cpu_py;
        color=ParticleLife.get_ptypes(m),
        colormap=:curl,
        markersize=3,
    )
    return f
end
