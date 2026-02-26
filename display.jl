include("sim.jl")
using .ParticleLifeSim

using WGLMakie
using Bonito
using Colors

WGLMakie.activate!()

const TYPE_COLORS = [
    colorant"#e74c3c", colorant"#3498db", colorant"#2ecc71", colorant"#f39c12",
    colorant"#9b59b6", colorant"#1abc9c", colorant"#e91e63", colorant"#cddc39",
]
const MAX_TYPES = length(TYPE_COLORS)   # 8 — fixed display matrix size
const TARGET_DT = 1.0 / 60.0

# ── App ───────────────────────────────────────────────────────────────────────
app = App() do session::Session

    model_ref = Ref(create_model())

    colors_for(pt) = [TYPE_COLORS[mod1(Int(t), MAX_TYPES)] for t in pt]

    function make_display_mat(m)
        nt  = Int(m.num_types)
        out = fill(NaN32, MAX_TYPES, MAX_TYPES)
        out[1:nt, 1:nt] .= m.attraction_matrix
        return out
    end

    # ── Observables ───────────────────────────────────────────────────────────
    download_positions!(model_ref[])
    xs             = Observable(model_ref[].cpu_px)
    ys             = Observable(model_ref[].cpu_py)
    cols           = Observable(colors_for(get_ptypes(model_ref[])))
    mat_obs        = Observable(make_display_mat(model_ref[]))
    is_running     = Observable(false)
    step_obs       = Observable("Step: 0")
    fps_obs        = Observable("FPS: --")
    radius_obs     = Observable(model_ref[].max_radius)
    circle_vis_obs = Observable(false)
    nt_obs         = Observable(Int(model_ref[].num_types))

    # ── Makie Figure ──────────────────────────────────────────────────────────
    set_theme!(theme_dark())
    fig = Figure(size = (1100, 640), backgroundcolor = :black)

    ax_sim = Axis(fig[1, 1];
        title            = "Particle Life", titlecolor = :white,
        backgroundcolor  = :black,          aspect = DataAspect(),
        limits = (0f0, Float32(ParticleLifeSim.WORLD_SIZE),
                  0f0, Float32(ParticleLifeSim.WORLD_SIZE)),
        xgridvisible = false, ygridvisible = false,
        leftspinecolor   = :gray30, rightspinecolor  = :gray30,
        bottomspinecolor = :gray30, topspinecolor    = :gray30,
        xticklabelsvisible = false, yticklabelsvisible = false,
    )

    scatter!(ax_sim, xs, ys; color = cols, markersize = 5, strokewidth = 0)

    # ── View-distance circle ──────────────────────────────────────────────────
    # Visible only while the View Distance slider is being adjusted.
    # circle_vis_obs drives the alpha so no separate `visible` property is needed.
    let circle_pts = 128,
        cx = Float32(ParticleLifeSim.WORLD_SIZE) * 0.5f0,
        cy = Float32(ParticleLifeSim.WORLD_SIZE) * 0.5f0

        circle_xy = @lift begin
            r  = $radius_obs
            ts = LinRange(0f0, 2f0 * Float32(π), circle_pts + 1)
            Point2f.(cx .+ r .* cos.(ts), cy .+ r .* sin.(ts))
        end

        circle_color = @lift($circle_vis_obs ? RGBAf(1.0, 1.0, 1.0, 0.45) :
                                               RGBAf(0.0, 0.0, 0.0, 0.0))

        lines!(ax_sim, circle_xy;
            color = circle_color, linewidth = 1.5, linestyle = :dash)
    end

    text!(ax_sim, 0.01f0, 0.02f0;
        text = fps_obs, color = :white, fontsize = 13, align = (:left, :bottom))

    # ── Attraction matrix heatmap ─────────────────────────────────────────────
    right  = fig[1, 2]
    ax_mat = Axis(right[1, 1];
        title           = "Attraction Matrix  (L+0.1 / R-0.1)",
        titlecolor      = :white, aspect = DataAspect(),
        xgridvisible    = false,  ygridvisible = false,
        backgroundcolor = :black,
        xlabel = "Being seen", ylabel = "Seeing",
        xlabelcolor = :gray70, ylabelcolor = :gray70,
        xticklabelsvisible = false, yticklabelsvisible = false,
        xticksvisible      = false, yticksvisible      = false,
        limits = (0f0, Float32(MAX_TYPES) + 0.5f0,
                  0f0, Float32(MAX_TYPES) + 0.5f0),
    )

    heatmap!(ax_mat, 1:MAX_TYPES, 1:MAX_TYPES, @lift($mat_obs');
        colormap = :RdBu, colorrange = (-1f0, 1f0), nan_color = :black)

    dim = RGBAf(0.25, 0.25, 0.25, 0.4)
    for t in 1:MAX_TYPES
        label_color = @lift(t <= $nt_obs ? TYPE_COLORS[t] : dim)
        text!(ax_mat, Float32(t), 0.22f0; text = "T$t", color = label_color,
            align = (:center, :center), fontsize = 9, font = :bold)
        text!(ax_mat, 0.22f0, Float32(t); text = "T$t", color = label_color,
            align = (:center, :center), fontsize = 9, font = :bold)
    end

    on(events(ax_mat.scene).mousebutton) do event
        event.action == Mouse.press || return
        mp = mouseposition(ax_mat)
        j  = round(Int, mp[1]);  i = round(Int, mp[2])
        nt = nt_obs[]
        if 1 <= i <= nt && 1 <= j <= nt
            m     = model_ref[]
            delta = event.button == Mouse.left ? 0.1f0 : -0.1f0
            m.attraction_matrix[i, j] = clamp(
                m.attraction_matrix[i, j] + delta, -1f0, 1f0)
            m.attr_dirty = true
            mat_obs[]    = make_display_mat(m)
        end
    end

    Colorbar(right[1, 2]; colormap = :RdBu, limits = (-1f0, 1f0),
        label = "attraction", labelcolor = :white,
        tickcolor = :white, ticklabelcolor = :white)

    Legend(right[2, 1:2],
        [MarkerElement(color = TYPE_COLORS[mod1(t, MAX_TYPES)],
                       marker = :circle, markersize = 12) for t in 1:MAX_TYPES],
        ["Type $t" for t in 1:MAX_TYPES];
        orientation = :horizontal, framecolor = :gray30,
        labelcolor = :white, patchcolor = :transparent)

    colsize!(fig.layout, 1, Relative(0.65))

    # ── Sliders ───────────────────────────────────────────────────────────────
    # Bonito.Slider takes any AbstractRange; .value[] holds the element type.
    # We use integer index ranges for radius/dt (Bonito doesn't support
    # set_close_to! on Slider{Float32}) and map index → float in the handlers.
    # The thumb starts at the first element; we assign the default index so the
    # thumb reflects the model's actual starting value.
    radius_vals  = LinRange(0.02f0,   0.125f0, 200)
    dt_vals      = LinRange(0.0005f0, 0.004f0, 200)
    friction_vals = LinRange(0.0f0,   0.5f0,   200)

    radius_default  = argmin(abs.(collect(radius_vals)   .- model_ref[].max_radius))
    dt_default      = argmin(abs.(collect(dt_vals)       .- model_ref[].dt))
    friction_default = argmin(abs.(collect(friction_vals) .- model_ref[].friction))

    sl_radius   = Slider(1:200;       value = radius_default)
    sl_dt       = Slider(1:200;       value = dt_default)
    sl_friction = Slider(1:200;       value = friction_default)
    sl_spf      = Slider(1:32;        value = model_ref[].steps_per_frame)
    sl_types    = Slider(2:MAX_TYPES; value = Int(model_ref[].num_types))

    radius_display   = Observable(string(round(radius_vals[sl_radius.value[]], digits = 3)))
    dt_display       = Observable(string(round(dt_vals[sl_dt.value[]],         digits = 4)))
    friction_display = Observable(string(round(friction_vals[sl_friction.value[]], digits = 3)))
    spf_display      = Observable(string(sl_spf.value[]))
    types_display    = Observable(string(sl_types.value[]))

    # Hide-circle debounce via counter.
    # Each slider move captures the counter value; the spawned task only hides
    # the circle if no subsequent move happened during the 1.5 s sleep.
    # Threads.@spawn (not @async) gives the timer its own OS thread so it
    # isn't starved by the sim loop.
    radius_touch = Ref(0)

    on(sl_radius.value) do idx
        v = radius_vals[clamp(idx, 1, 200)]
        model_ref[].max_radius = v
        radius_display[]       = string(round(v, digits = 3))
        radius_obs[]           = v
        circle_vis_obs[]       = true
        radius_touch[] += 1
        my_touch = radius_touch[]
        Threads.@spawn begin
            sleep(1.5)
            radius_touch[] == my_touch && (circle_vis_obs[] = false)
        end
    end

    on(sl_dt.value) do idx
        v = dt_vals[clamp(idx, 1, 200)]
        model_ref[].dt = v
        dt_display[]   = string(round(v, digits = 4))
    end

    on(sl_friction.value) do idx
        v = friction_vals[clamp(idx, 1, 200)]
        model_ref[].friction = v
        friction_display[]   = string(round(v, digits = 3))
    end

    on(sl_spf.value) do v
        model_ref[].steps_per_frame = v
        spf_display[]               = string(v)
    end

    # ── Model recreation (types slider) ───────────────────────────────────────
    function recreate_model!(nt)
        was = is_running[]; is_running[] = false
        old = model_ref[]

        model_ref[] = create_model(
            num_types = nt,
            seed      = abs(rand(Int)) % typemax(Int32),
        )
        m = model_ref[]
        m.dt              = old.dt
        m.max_radius      = old.max_radius
        m.friction        = old.friction
        m.steps_per_frame = old.steps_per_frame

        download_positions!(m)
        xs[]       = m.cpu_px
        ys[]       = m.cpu_py
        cols[]     = colors_for(get_ptypes(m))
        mat_obs[]  = make_display_mat(m)
        nt_obs[]   = nt
        step_obs[] = "Step: 0"

        is_running[] = was
    end

    on(sl_types.value) do nt
        types_display[] = string(nt)
        recreate_model!(nt)
    end

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_play  = Button("[ Start ]")
    btn_rand  = Button("Randomize Matrix")
    btn_reset = Button("Reset Particles")
    btn_both  = Button("Randomize + Reset")

    on(btn_play) do _
        is_running[]       = !is_running[]
        btn_play.content[] = is_running[] ? "[ Pause ]" : "[ Start ]"
    end

    function refresh_display!()
        m = model_ref[]
        download_positions!(m)
        notify(xs); notify(ys)
        cols[]    = colors_for(get_ptypes(m))
        mat_obs[] = make_display_mat(m)
    end

    on(btn_rand) do _
        randomize_matrix!(model_ref[])
        mat_obs[] = make_display_mat(model_ref[])
    end
    on(btn_reset) do _
        was = is_running[]; is_running[] = false
        reset_particles!(model_ref[]); refresh_display!()
        step_obs[] = "Step: 0"; is_running[] = was
    end
    on(btn_both) do _
        was = is_running[]; is_running[] = false
        randomize_matrix!(model_ref[]); reset_particles!(model_ref[])
        refresh_display!(); step_obs[] = "Step: 0"; is_running[] = was
    end

    # ── Sim loop ──────────────────────────────────────────────────────────────
    Threads.@spawn begin
        fps_smooth = 0.0
        t_last     = time()

        while true
            try
                t0      = time()
                elapsed = t0 - t_last
                t_last  = t0

                if elapsed > 0
                    inst       = 1.0 / elapsed
                    fps_smooth = fps_smooth == 0.0 ? inst : 0.1*inst + 0.9*fps_smooth
                    m          = model_ref[]
                    fps_obs[]  = "FPS: $(round(Int, fps_smooth))  x$(m.steps_per_frame)"
                end

                if is_running[]
                    m = model_ref[]
                    for _ in 1:m.steps_per_frame
                        model_step!(m)
                    end
                    download_positions!(m)
                    notify(xs); notify(ys)
                    step_obs[] = "Step: $(m.step_count)"
                end

                spent = time() - t0
                rem   = TARGET_DT - spent
                rem > 0 && sleep(rem)

            catch e
                @error "Sim loop error" exception = (e, catch_backtrace())
                sleep(0.5)
            end
        end
    end

    # ── DOM layout ────────────────────────────────────────────────────────────
    lbl(s)  = DOM.span(s; style = "color:#aaa")
    val(ob) = DOM.span(ob; style = "color:#888; margin:0 16px 0 6px; min-width:36px; display:inline-block")

    controls = DOM.div(
        DOM.div(
            lbl("Types: "), sl_types, val(types_display);
            style = "display:flex; align-items:center; gap:6px; padding:6px 0;",
        ),
        DOM.div(
            lbl("View Distance: "), sl_radius, val(radius_display),
            lbl("dt: "), sl_dt, val(dt_display),
            lbl("Friction: "), sl_friction, val(friction_display),
            lbl("Steps/frame: "), sl_spf,
            DOM.span(spf_display; style = "color:#888; margin-left:6px; min-width:14px; display:inline-block");
            style = "display:flex; align-items:center; gap:6px; padding:4px 0;",
        ),
        DOM.div(
            btn_play, btn_rand, btn_reset, btn_both,
            DOM.span(step_obs; style = "color:#888; margin-left:16px;");
            style = "display:flex; align-items:center; gap:8px; padding:4px 0;",
        );
        style = "background:#111; padding:8px 12px; font-family:monospace; font-size:13px;",
    )

    return DOM.div(fig, controls;
        style = "display:flex; flex-direction:column; background:#000; width:fit-content;")
end

# ── Server ────────────────────────────────────────────────────────────────────
const HOST = "0.0.0.0"
const PORT = 8080

server = Server(app, HOST, PORT)

println("http://localhost:$(PORT)")
wait(server)
