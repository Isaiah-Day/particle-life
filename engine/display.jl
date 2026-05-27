using WGLMakie
using Bonito
using Colors
using Bonito: Observables

include("species.jl")

WGLMakie.activate!()

# ─────────────────────────────────────────────────────────────────────────────
# DisplayModel
# ─────────────────────────────────────────────────────────────────────────────

mutable struct DisplayModel
  xs::Observable
  ys::Observable
  cols::Observable
  mat_obs::Observable
  is_running::Observable
  step_obs::Observable{String}
  radius_obs::Observable
  circle_vis_obs::Observable
  nt_obs::Observable{Int}
  # Density heatmap
  heatmap_on_obs::Observable
  heatmap_n_obs::Observable
  heatmap_thr_obs::Observable
  heatmap_mat_obs::Observable
  # Species heatmap
  species_on_obs::Observable
  species_hm_mat_obs::Observable   # Matrix{RGBAf}
  # Makeup
  species_weight_obs               # Vector{Observable}
end

function make_display_mat(m, MAX_TYPES)
  nt  = Int(m.num_types)
  out = fill(NaN32, MAX_TYPES, MAX_TYPES)
  out[1:nt, 1:nt] .= m.attraction_matrix
  return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Species cluster heatmap  (CPU, called from sim thread)
# ─────────────────────────────────────────────────────────────────────────────

const CLUSTER_COLORS = [
  colorant"#ff6b6b", colorant"#4ecdc4", colorant"#ffe66d", colorant"#a29bfe",
  colorant"#fd79a8", colorant"#55efc4", colorant"#fdcb6e", colorant"#74b9ff",
  colorant"#e17055", colorant"#00cec9", colorant"#6c5ce7", colorant"#fab1a0",
]

function species_cluster_heatmap(model; vel_threshold=SPECIES_VEL_THRESHOLD)
  labels = find_species(model; vel_threshold=vel_threshold)
  N  = size(labels, 1)
  NC = length(CLUSTER_COLORS)
  img = zeros(RGBAf, N, N)
  for x in 1:N, y in 1:N
    lbl = labels[x, y]
    lbl == 0 && continue
    c = CLUSTER_COLORS[mod1(lbl, NC)]
    img[x, y] = RGBAf(red(c), green(c), blue(c), 0.75f0)
  end
  return img
end

# ─────────────────────────────────────────────────────────────────────────────
# Sim axis + drag-zoom
# ─────────────────────────────────────────────────────────────────────────────

function make_sim_axis(fig, world_size)
  ax = Axis(fig[1, 1];
    backgroundcolor=:black, aspect=DataAspect(),
    xgridvisible=false, ygridvisible=false,
    leftspinevisible=false, rightspinevisible=false,
    bottomspinevisible=false, topspinevisible=false,
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
  )
  xlims!(ax, 0f0, world_size); ylims!(ax, 0f0, world_size)
  ax.limits[] = (nothing, nothing, nothing, nothing)
  deregister_interaction!(ax, :rectanglezoom)
  rect_overlay = Observable(Point2f[])
  lines!(ax, rect_overlay; color=:white, linewidth=1, linestyle=:dash,
         xautolimits=false, yautolimits=false)
  return ax, rect_overlay
end

function setup_drag_zoom!(ax, rect_overlay)
  drag_origin    = Ref{Union{Nothing, Point2f}}(nothing)
  last_mouse_pos = Ref{Point2f}(Point2f(0, 0))
  on(events(ax.scene).mouseposition) do _
    last_mouse_pos[] = mouseposition(ax)
    start = drag_origin[]
    start === nothing && return
    finish = last_mouse_pos[]
    rect_overlay[] = Point2f[
      start, Point2f(finish[1], start[2]),
      finish, Point2f(start[1], finish[2]), start,
    ]
  end
  on(events(ax.scene).mousebutton) do event
    event.button == Mouse.left || return
    if event.action == Mouse.press
      mp = last_mouse_pos[]
      fl = ax.finallimits[]
      in_axis = fl.origin[1] <= mp[1] <= fl.origin[1] + fl.widths[1] &&
                fl.origin[2] <= mp[2] <= fl.origin[2] + fl.widths[2]
      in_axis && (drag_origin[] = mp)
    elseif event.action == Mouse.release
      start = drag_origin[]
      drag_origin[] = nothing
      rect_overlay[] = Point2f[]
      start === nothing && return
      finish = last_mouse_pos[]
      fl = ax.finallimits[]
      min_drag = min(fl.widths[1], fl.widths[2]) * 0.02f0
      (abs(finish[1]-start[1]) < min_drag || abs(finish[2]-start[2]) < min_drag) && return
      x1, x2 = minmax(start[1], finish[1]); y1, y2 = minmax(start[2], finish[2])
      xlims!(ax, x1, x2); ylims!(ax, y1, y2)
      ax.limits[] = (nothing, nothing, nothing, nothing)
    end
  end
end

# ─────────────────────────────────────────────────────────────────────────────
# Double-buffer tick + scatter
# ─────────────────────────────────────────────────────────────────────────────

function setup_double_buffer!(fig, ax, dm,
                               stage_px, stage_py, stage_cols,
                               stage_hm, stage_species_hm,
                               frame_dirty, stage_lock)
  on(events(fig.scene).tick) do _
    if Threads.atomic_cas!(frame_dirty, 1, 0) == 1
      lock(stage_lock) do
        copy!(dm.xs[], stage_px)
        copy!(dm.ys[], stage_py)
        copy!(dm.cols[], stage_cols)
        hm = stage_hm[]
        if hm !== nothing
          dm.heatmap_mat_obs[] = hm
          stage_hm[] = nothing
        end
        shm = stage_species_hm[]
        if shm !== nothing
          dm.species_hm_mat_obs[] = shm
          stage_species_hm[] = nothing
        end
      end
      notify(dm.xs); notify(dm.ys); notify(dm.cols)
    end
  end
  scatter!(ax, dm.xs, dm.ys; color=dm.cols, markersize=5, strokewidth=0,
           xautolimits=false, yautolimits=false)
end

# ─────────────────────────────────────────────────────────────────────────────
# Density heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────

function setup_heatmap_overlay!(ax, dm, world_size)
  heatmap_display = @lift replace($(dm.heatmap_mat_obs), 0.0f0 => NaN32)
  hm_x = Observable(LinRange(0f0, world_size, size(dm.heatmap_mat_obs[], 1) + 1))
  hm_y = Observable(LinRange(0f0, world_size, size(dm.heatmap_mat_obs[], 2) + 1))
  on(dm.heatmap_mat_obs) do m
    r = LinRange(0f0, world_size, size(m, 1) + 1)
    hm_x[] = r; hm_y[] = r
  end
  hm_alpha = @lift($(dm.heatmap_on_obs) ? 0.45f0 : 0.0f0)
  hm = heatmap!(ax, hm_x, hm_y, heatmap_display; colormap=:hot, lowclip=:transparent)
  on(hm_alpha) do a; hm.alpha = a; end
  hm.alpha = hm_alpha[]
end

# ─────────────────────────────────────────────────────────────────────────────
# Species heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────

function setup_species_overlay!(ax, dm, world_size)
  ws = Float32(world_size)
  sp_alpha = @lift($(dm.species_on_obs) ? 1.0f0 : 0.0f0)
  img_plot = image!(ax, 0f0 .. ws, 0f0 .. ws, dm.species_hm_mat_obs;
                    inspectable=false, xautolimits=false, yautolimits=false)
  on(sp_alpha) do a; img_plot.alpha = a; end
  img_plot.alpha = sp_alpha[]
end

# ─────────────────────────────────────────────────────────────────────────────
# View-distance circle overlay
# ─────────────────────────────────────────────────────────────────────────────

function setup_circle_overlay!(ax, dm, world_size)
  cx = Float32(world_size) * 0.5f0; cy = cx
  circle_xy = @lift begin
    r  = $(dm.radius_obs)
    ts = LinRange(0f0, 2f0 * Float32(π), 129)
    Point2f.(cx .+ r .* cos.(ts), cy .+ r .* sin.(ts))
  end
  circle_color = @lift($(dm.circle_vis_obs) ? RGBAf(1,1,1,0.45) : RGBAf(0,0,0,0))
  lines!(ax, circle_xy; color=circle_color, linewidth=1.5, linestyle=:dash)
end

# ─────────────────────────────────────────────────────────────────────────────
# Slider wiring
# ─────────────────────────────────────────────────────────────────────────────

function setup_slider_handlers!(dm, model_ref, MAX_TYPES, MAX_WEIGHT_STEPS,
                                  sl_radius, sl_dt, sl_friction, sl_spf, sl_types, sl_hm_n, sl_hm_thr,
                                  radius_vals, dt_vals, friction_vals,
                                  radius_display, dt_display, friction_display, spf_display, types_display,
                                  hm_n_display, hm_thr_display,
                                  colors_for, make_display_mat_fn)
  radius_touch = Ref(0)
  on(sl_radius.value) do idx
    v = radius_vals[clamp(idx, 1, 200)]
    model_ref[].max_radius = v
    radius_display[] = string(round(v, digits=3))
    dm.radius_obs[] = v; dm.circle_vis_obs[] = true
    radius_touch[] += 1; my_touch = radius_touch[]
    Threads.@spawn begin sleep(1.5); radius_touch[] == my_touch && (dm.circle_vis_obs[] = false); end
  end
  on(sl_dt.value) do idx
    v = dt_vals[clamp(idx, 1, 200)]
    model_ref[].dt = v; dt_display[] = string(round(v, digits=4))
  end
  on(sl_friction.value) do idx
    v = friction_vals[clamp(idx, 1, 200)]
    model_ref[].friction = v; friction_display[] = string(round(v, digits=3))
  end
  on(sl_spf.value) do v
    model_ref[].steps_per_frame = v; spf_display[] = string(v)
  end
  on(sl_hm_n.value) do v
    n = v * v
    dm.heatmap_n_obs[] = n
    hm_n_display[] = string(n)
  end
  on(sl_hm_thr.value) do v; dm.heatmap_thr_obs[] = v; hm_thr_display[] = string(v); end
  on(sl_types.value) do nt
    types_display[] = string(nt)
    was = dm.is_running[]; dm.is_running[] = false
    old = model_ref[]
    model_ref[] = create_model(num_types=nt, seed=abs(rand(Int)) % typemax(Int32))
    m = model_ref[]
    m.dt = old.dt; m.max_radius = old.max_radius
    m.friction = old.friction; m.steps_per_frame = old.steps_per_frame
    eq = MAX_WEIGHT_STEPS ÷ nt
    for t in 1:MAX_TYPES; dm.species_weight_obs[t][] = t <= nt ? eq : 0; end
    download_positions!(m)
    dm.xs[] = m.cpu_px; dm.ys[] = m.cpu_py
    dm.cols[] = colors_for(get_ptypes(m))
    dm.mat_obs[] = make_display_mat_fn(m)
    dm.nt_obs[] = nt; dm.step_obs[] = "Step: 0"
    dm.is_running[] = was
  end
end

# ─────────────────────────────────────────────────────────────────────────────
# Attraction matrix panel
# ─────────────────────────────────────────────────────────────────────────────

function build_matrix_panel(dm, model_ref, TYPE_COLORS, MAX_TYPES, make_display_mat_fn)
  fig_mat = Figure(size=(260, 240), backgroundcolor=RGBAf(0,0,0,0), figure_padding=4)
  ax_mat  = Axis(fig_mat[1, 1];
    title="Attraction  (L +0.1 / R −0.1)", titlecolor=:white, titlesize=11,
    aspect=DataAspect(), xgridvisible=false, ygridvisible=false,
    backgroundcolor=RGBAf(0,0,0,0),
    xlabel="seen", ylabel="seeing",
    xlabelcolor=:gray60, ylabelcolor=:gray60, xlabelsize=10, ylabelsize=10,
    xticklabelsvisible=false, yticklabelsvisible=false,
    xticksvisible=false, yticksvisible=false,
    limits=(0f0, Float32(MAX_TYPES)+0.5f0, 0f0, Float32(MAX_TYPES)+0.5f0),
  )
  heatmap!(ax_mat, 1:MAX_TYPES, 1:MAX_TYPES, @lift($(dm.mat_obs)');
           colormap=:RdBu, colorrange=(-1f0, 1f0), nan_color=:black)
  dim = RGBAf(0.25, 0.25, 0.25, 0.4)
  for t in 1:MAX_TYPES
    lc = @lift(t <= $(dm.nt_obs) ? TYPE_COLORS[t] : dim)
    text!(ax_mat, Float32(t), 0.22f0; text="T$t", color=lc, align=(:center,:center), fontsize=8, font=:bold)
    text!(ax_mat, 0.22f0, Float32(t); text="T$t", color=lc, align=(:center,:center), fontsize=8, font=:bold)
  end
  Colorbar(fig_mat[1, 2]; colormap=:RdBu, limits=(-1f0,1f0),
           tickcolor=:white, ticklabelcolor=:white, ticklabelsize=9, label="", width=10)
  on(events(ax_mat.scene).mousebutton) do event
    event.action == Mouse.press || return
    mp = mouseposition(ax_mat)
    j = round(Int, mp[1]); i = round(Int, mp[2])
    nt = dm.nt_obs[]
    if 1 <= i <= nt && 1 <= j <= nt
      m = model_ref[]
      delta = event.button == Mouse.left ? 0.1f0 : -0.1f0
      m.attraction_matrix[i,j] = clamp(m.attraction_matrix[i,j] + delta, -1f0, 1f0)
      m.attr_dirty = true
      dm.mat_obs[] = make_display_mat_fn(m)
    end
  end
  return fig_mat
end

# ─────────────────────────────────────────────────────────────────────────────
# Weight sliders (Particle Makeup)
# ─────────────────────────────────────────────────────────────────────────────

function build_weight_sliders(dm, weight_pct_obs, TYPE_COLORS, MAX_TYPES, MAX_WEIGHT_STEPS)
  function make_one(t)
    col = "#" * hex(TYPE_COLORS[mod1(t, MAX_TYPES)])
    pct = @lift($weight_pct_obs[t])
    vis = @lift($(dm.nt_obs) >= t ?
      "display:flex; flex-direction:column; align-items:center; gap:1px;" : "display:none;")
    sl  = Slider(0:MAX_WEIGHT_STEPS; value=dm.species_weight_obs[t][])
    on(sl.value) do v; dm.species_weight_obs[t][] = v; end
    DOM.div(
      DOM.span("T$t"; style="color:$col; font-size:10px; font-weight:bold;"),
      sl,
      DOM.span(pct; style="color:#888; font-size:9px; min-width:24px; text-align:center;");
      style=vis,
    )
  end
  DOM.div(
    [make_one(t) for t in 1:MAX_TYPES]...;
    style="display:flex; flex-direction:row; gap:3px; align-items:flex-end; flex-wrap:wrap;",
  )
end

# ─────────────────────────────────────────────────────────────────────────────
# Sim loop
# ─────────────────────────────────────────────────────────────────────────────

function setup_sim_loop!(dm, model_ref, TYPE_COLORS,
                          stage_px, stage_py, stage_cols,
                          stage_hm, stage_species_hm,
                          frame_dirty, stage_lock, colors_for, TARGET_DT)
  Threads.@spawn begin
    t_last = time()
    while true
      try
        t0 = time(); t_last = t0
        if dm.is_running[]
          m = model_ref[]
          for _ in 1:m.steps_per_frame; model_step!(m); end
          download_positions!(m)
          new_cols       = colors_for(get_ptypes(m))
          new_hm         = dm.heatmap_on_obs[] ?
                             heatmap(m, dm.heatmap_n_obs[], dm.heatmap_thr_obs[]) : nothing
          new_species_hm = dm.species_on_obs[] ?
                             species_cluster_heatmap(model_ref[]) : nothing
          lock(stage_lock) do
            copy!(stage_px, m.cpu_px); copy!(stage_py, m.cpu_py)
            copy!(stage_cols, new_cols)
            stage_hm[]         = new_hm
            stage_species_hm[] = new_species_hm
          end
          Threads.atomic_xchg!(frame_dirty, 1)
          dm.step_obs[] = "Step: $(m.step_count)"
        end
        rem = TARGET_DT - (time() - t0)
        rem > 0 && sleep(rem)
      catch e
        @error "Sim loop error" exception=(e, catch_backtrace())
        sleep(0.5)
      end
    end
  end
end

# ─────────────────────────────────────────────────────────────────────────────
# display()
# ─────────────────────────────────────────────────────────────────────────────

function display(model::ParticleModel;
                 PORT=8080, HOST="0.0.0.0",
                 TYPE_COLORS=[colorant"#e74c3c", colorant"#3498db", colorant"#2ecc71",
                               colorant"#f39c12", colorant"#9b59b6", colorant"#1abc9c",
                               colorant"#e91e63", colorant"#cddc39", colorant"#00bcd4"])
  MAX_TYPES        = length(TYPE_COLORS)
  MAX_WEIGHT_STEPS = 100
  TARGET_DT        = 1.0 / 60.0

  app = App() do session::Session
    model_ref  = Ref(model)
    world_size = model_ref[].WORLD_SIZE

    colors_for(pt) = [TYPE_COLORS[mod1(Int(t), MAX_TYPES)] for t in pt]
    make_mat(m)    = make_display_mat(m, MAX_TYPES)
    init_n         = 64

    # ── Build DisplayModel ────────────────────────────────────────────────────
    download_positions!(model_ref[])
    dm = DisplayModel(
      Observable(model_ref[].cpu_px),
      Observable(model_ref[].cpu_py),
      Observable(colors_for(get_ptypes(model_ref[]))),
      Observable(make_mat(model_ref[])),
      Observable(false),
      Observable("Step: 0"),
      Observable(model_ref[].max_radius),
      Observable(false),
      Observable(Int(model_ref[].num_types)),
      # density
      Observable(false),
      Observable(init_n),
      Observable(10),
      Observable(zeros(Float32, init_n, init_n)),
      # species
      Observable(false),
      Observable(zeros(RGBAf, SPECIES_N, SPECIES_N)),
      # makeup
      [Observable(MAX_WEIGHT_STEPS ÷ MAX_TYPES) for _ in 1:MAX_TYPES],
    )

    # ── Figure + sim axis ─────────────────────────────────────────────────────
    set_theme!(theme_dark())
    fig    = Figure(size=(1400, 900), backgroundcolor=:black, figure_padding=0)
    ax_sim, rect_overlay = make_sim_axis(fig, world_size)
    setup_drag_zoom!(ax_sim, rect_overlay)

    # ── Double-buffer stage ───────────────────────────────────────────────────
    np               = Int(model_ref[].num_particles)
    stage_px         = Vector{Float32}(undef, np)
    stage_py         = Vector{Float32}(undef, np)
    stage_cols       = Vector{eltype(TYPE_COLORS)}(undef, np)
    stage_hm         = Ref{Union{Nothing, Matrix{Float32}}}(nothing)
    stage_species_hm = Ref{Union{Nothing, Matrix{RGBAf}}}(nothing)
    frame_dirty      = Threads.Atomic{Int}(0)
    stage_lock       = ReentrantLock()

    setup_double_buffer!(fig, ax_sim, dm,
      stage_px, stage_py, stage_cols, stage_hm, stage_species_hm, frame_dirty, stage_lock)
    setup_heatmap_overlay!(ax_sim, dm, world_size)
    setup_species_overlay!(ax_sim, dm, world_size)
    setup_circle_overlay!(ax_sim, dm, world_size)

    # ── Sliders ───────────────────────────────────────────────────────────────
    radius_vals   = LinRange(0.02f0, 0.125f0, 200)
    dt_vals       = LinRange(0.0005f0, 0.004f0, 200)
    friction_vals = LinRange(0.0f0, 0.5f0, 200)

    sl_radius   = Slider(1:200; value=argmin(abs.(collect(radius_vals)   .- model_ref[].max_radius)))
    sl_dt       = Slider(1:200; value=argmin(abs.(collect(dt_vals)       .- model_ref[].dt)))
    sl_friction = Slider(1:200; value=argmin(abs.(collect(friction_vals) .- model_ref[].friction)))
    sl_spf      = Slider(1:32;  value=model_ref[].steps_per_frame)
    sl_types    = Slider(2:MAX_TYPES; value=Int(model_ref[].num_types))
    sl_hm_n     = Slider(3:24;  value=isqrt(dm.heatmap_n_obs[]))
    sl_hm_thr   = Slider(1:200; value=dm.heatmap_thr_obs[])

    radius_display   = Observable(string(round(radius_vals[sl_radius.value[]], digits=3)))
    dt_display       = Observable(string(round(dt_vals[sl_dt.value[]], digits=4)))
    friction_display = Observable(string(round(friction_vals[sl_friction.value[]], digits=3)))
    spf_display      = Observable(string(sl_spf.value[]))
    types_display    = Observable(string(sl_types.value[]))
    hm_n_display     = Observable(string(dm.heatmap_n_obs[]))
    hm_thr_display   = Observable(string(dm.heatmap_thr_obs[]))

    setup_slider_handlers!(dm, model_ref, MAX_TYPES, MAX_WEIGHT_STEPS,
      sl_radius, sl_dt, sl_friction, sl_spf, sl_types, sl_hm_n, sl_hm_thr,
      radius_vals, dt_vals, friction_vals,
      radius_display, dt_display, friction_display, spf_display, types_display,
      hm_n_display, hm_thr_display, colors_for, make_mat)

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_play          = Button("▶  Start")
    btn_rand          = Button("Randomize Matrix")
    btn_reset         = Button("Reset Particles")
    btn_both          = Button("Randomize + Reset")
    btn_heatmap       = Button("Density Map: OFF")
    btn_species       = Button("Species Map: OFF")
    btn_apply_weights = Button("Apply Makeup")

    # Mutual exclusion: turning on one overlay turns off the other
    on(btn_heatmap) do _
      dm.heatmap_on_obs[] = !dm.heatmap_on_obs[]
      if dm.heatmap_on_obs[]
        dm.species_on_obs[] = false
        btn_species.content[] = "Species Map: OFF"
      end
      btn_heatmap.content[] = dm.heatmap_on_obs[] ? "Density Map: ON" : "Density Map: OFF"
    end
    on(btn_species) do _
      dm.species_on_obs[] = !dm.species_on_obs[]
      if dm.species_on_obs[]
        dm.heatmap_on_obs[] = false
        btn_heatmap.content[] = "Density Map: OFF"
      end
      btn_species.content[] = dm.species_on_obs[] ? "Species Map: ON" : "Species Map: OFF"
    end

    function current_weights()
      nt  = dm.nt_obs[]
      raw = Float32[dm.species_weight_obs[t][] for t in 1:nt]
      s   = sum(raw); s == 0 && return fill(1.0f0/nt, nt)
      raw ./ s
    end

    function refresh_display!()
      m = model_ref[]
      download_positions!(m)
      copy!(dm.xs[], m.cpu_px); copy!(dm.ys[], m.cpu_py)
      dm.cols[] = colors_for(get_ptypes(m))
      notify(dm.xs); notify(dm.ys)
      dm.mat_obs[] = make_mat(m)
    end

    function reset_with_weights!(also_randomize=false)
      was = dm.is_running[]; dm.is_running[] = false
      model_ref[].species_weights = current_weights()
      also_randomize && randomize_matrix!(model_ref[])
      reset_particles!(model_ref[])
      refresh_display!()
      dm.step_obs[] = "Step: 0"; dm.is_running[] = was
    end

    on(btn_rand)          do _; randomize_matrix!(model_ref[]); dm.mat_obs[] = make_mat(model_ref[]); end
    on(btn_reset)         do _; reset_with_weights!(); end
    on(btn_both)          do _; reset_with_weights!(true); end
    on(btn_apply_weights) do _; reset_with_weights!(); end
    on(btn_play) do _
      dm.is_running[] = !dm.is_running[]
      btn_play.content[] = dm.is_running[] ? "⏸  Pause" : "▶  Start"
    end

    # ── Sim loop ──────────────────────────────────────────────────────────────
    setup_sim_loop!(dm, model_ref, TYPE_COLORS,
      stage_px, stage_py, stage_cols, stage_hm, stage_species_hm,
      frame_dirty, stage_lock, colors_for, TARGET_DT)

    # ── Weight percentage observable ──────────────────────────────────────────
    weight_pct_obs = Observable(fill("", MAX_TYPES))
    function update_pct_obs()
      nt  = dm.nt_obs[]
      raw = [Float32(dm.species_weight_obs[t][]) for t in 1:nt]
      s   = sum(raw); out = fill("", MAX_TYPES)
      for t in 1:nt
        out[t] = s == 0 ? "0%" : string(round(Int, 100 * raw[t] / s)) * "%"
      end
      weight_pct_obs[] = out
    end
    update_pct_obs()
    on(dm.nt_obs) do _; update_pct_obs(); end
    for t in 1:MAX_TYPES; on(dm.species_weight_obs[t]) do _; update_pct_obs(); end; end

    # ── DOM helpers ───────────────────────────────────────────────────────────
    lbl(s)       = DOM.span(s; style="color:#aaa; font-size:12px; white-space:nowrap;")
    val(ob)      = DOM.span(ob; style="color:#ccc; font-size:11px; min-width:38px; display:inline-block; text-align:right;")
    row(c...; gap="8px") = DOM.div(c...; style="display:flex; align-items:center; gap:$gap; padding:3px 0;")
    sec_title(s) = DOM.div(s; style="color:#ddd; font-size:11px; font-weight:bold; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;")
    section_style = "padding:8px 0; border-bottom:1px solid #333;"

    # ── Sub-panels ────────────────────────────────────────────────────────────
    fig_mat        = build_matrix_panel(dm, model_ref, TYPE_COLORS, MAX_TYPES, make_mat)
    weight_sliders = build_weight_sliders(dm, weight_pct_obs, TYPE_COLORS, MAX_TYPES, MAX_WEIGHT_STEPS)

    # ── Overlay tabs (Density / Species) ──────────────────────────────────────
    overlay_tab = Observable(:density)   # :density | :species

    tab_btn_style(active_sym) = @lift begin
      active = $overlay_tab == active_sym
      base   = "padding:4px 10px; font-size:11px; cursor:pointer; border:none; border-radius:4px 4px 0 0; "
      active ? base * "background:#333; color:#fff; border-bottom:2px solid #7af;" :
               base * "background:transparent; color:#666;"
    end

    tab_density_btn = DOM.span("Density";  style=tab_btn_style(:density),
                                onclick="")   # wired via Observable below
    tab_species_btn = DOM.span("Species";  style=tab_btn_style(:species), onclick="")

    # Bonito button for tab switching (reuse Button widget for event support)
    tbtn_density = Button("Density")
    tbtn_species = Button("Species")
    on(tbtn_density) do _; overlay_tab[] = :density; end
    on(tbtn_species) do _; overlay_tab[] = :species; end

    tab_density_style = @lift($overlay_tab == :density ?
      "padding:3px 12px; font-size:11px; background:#2a2a2a; color:#fff; border-bottom:2px solid #69aaff;" :
      "padding:3px 12px; font-size:11px; background:transparent; color:#555;")
    tab_species_style = @lift($overlay_tab == :species ?
      "padding:3px 12px; font-size:11px; background:#2a2a2a; color:#fff; border-bottom:2px solid #69aaff;" :
      "padding:3px 12px; font-size:11px; background:transparent; color:#555;")

    density_body_vis = @lift($overlay_tab == :density ? "display:block;" : "display:none;")
    species_body_vis = @lift($overlay_tab == :species ? "display:block;" : "display:none;")

    # Species legend: cluster palette swatches
    NC = length(CLUSTER_COLORS)
    cluster_legend = DOM.div(
      [let col = "#" * hex(CLUSTER_COLORS[i])
        DOM.div(
          DOM.div(; style="width:10px; height:10px; border-radius:2px; background:$col; flex-shrink:0;"),
          DOM.span("Cluster $i"; style="color:#ccc; font-size:10px;");
          style="display:flex; align-items:center; gap:4px; margin:2px 0;",
        )
       end for i in 1:NC]...;
      style="margin-top:4px; columns:2; column-gap:8px;",
    )

    overlay_section = DOM.div(
      sec_title("Overlays"),
      # Tab bar
      DOM.div(
        DOM.div(tbtn_density; style=tab_density_style),
        DOM.div(tbtn_species; style=tab_species_style);
        style="display:flex; gap:2px; border-bottom:1px solid #333; margin-bottom:6px;",
      ),
      # Density tab body
      DOM.div(
        row(btn_heatmap),
        row(lbl("Grid n"),    sl_hm_n,   val(hm_n_display)),
        row(lbl("Threshold"), sl_hm_thr, val(hm_thr_display));
        style=density_body_vis,
      ),
      # Species tab body
      DOM.div(
        row(btn_species),
        DOM.div(
          DOM.span("Coherent clusters from find_species. Each ID → unique color.";
                   style="color:#777; font-size:10px; line-height:1.4;"),
          cluster_legend;
          style="margin-top:4px;",
        );
        style=species_body_vis,
      );
      style=section_style,
    )

    # ── Settings panel ────────────────────────────────────────────────────────
    panel_open = Observable(true)
    toggle_btn = Button("◀ Settings")
    on(toggle_btn) do _
      panel_open[] = !panel_open[]
      toggle_btn.content[] = panel_open[] ? "◀ Settings" : "▶ Settings"
    end

    settings_panel = DOM.div(
      DOM.div(
        DOM.span("⚙ Settings"; style="color:#eee; font-size:13px; font-weight:bold;"),
        toggle_btn;
        style="display:flex; justify-content:space-between; align-items:center; padding:6px 8px; background:#1a1a1a; border-bottom:1px solid #333;",
      ),
      DOM.div(
        DOM.div(
          DOM.div(sec_title("Attraction Matrix"), fig_mat; style=section_style),
          DOM.div(
            sec_title("Simulation"),
            row(lbl("Types"),    sl_types,    val(types_display)),
            row(lbl("Distance"), sl_radius,   val(radius_display)),
            row(lbl("dt"),       sl_dt,       val(dt_display)),
            row(lbl("Friction"), sl_friction, val(friction_display)),
            row(lbl("Steps/fr"), sl_spf,      val(spf_display));
            style=section_style,
          ),
          overlay_section,
          DOM.div(
            sec_title("Particle Makeup"),
            weight_sliders,
            DOM.div(btn_apply_weights; style="margin-top:6px;");
            style="padding:8px 0;",
          );
          style="overflow-y:auto; max-height:calc(100vh - 80px);",
        );
        style=@lift($panel_open ? "display:block;" : "display:none;"),
      );
      style="""
        position:fixed; top:12px; left:12px; z-index:100; width:300px;
        background:rgba(10,10,10,0.92); border:1px solid #444; border-radius:8px;
        font-family:monospace; box-shadow:0 4px 24px rgba(0,0,0,0.7);
        max-height:calc(100vh - 24px); overflow:hidden;
      """,
    )

    # ── Bottom-right controls ──────────────────────────────────────────────────
    br_controls = DOM.div(
      row(btn_play, btn_rand, btn_reset, btn_both; gap="6px"),
      DOM.div(
        DOM.span(dm.step_obs; style="color:#888; font-size:11px; font-family:monospace;");
        style="text-align:right; padding-top:2px;",
      );
      style="""
        position:fixed; bottom:14px; right:16px; z-index:100;
        background:rgba(10,10,10,0.88); border:1px solid #444; border-radius:8px;
        padding:8px 12px; font-family:monospace; box-shadow:0 4px 16px rgba(0,0,0,0.6);
      """,
    )

    return DOM.div(fig, settings_panel, br_controls;
                   style="width:100vw; height:100vh; overflow:hidden; background:#000; position:relative;")
  end

  server = Server(app, HOST, PORT)
  println("http://localhost:$(PORT)")
  wait(server)
end

