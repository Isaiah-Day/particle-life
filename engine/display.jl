using WGLMakie
using Bonito
using Colors
using Bonito: Observables

WGLMakie.activate!()


mutable struct DisplayModel
  xs::Observable
  ys::Observable 
  cols::Observable
  mat_obs::Observable
  is_running ::Observable
  step_obs ::Observable{String}
  #fps_obs ::Observable  
  radius_obs ::Observable
  circle_vis_obs::Observable  
  nt_obs::Observable{Int}   
  heatmap_on_obs::Observable  
  heatmap_n_obs::Observable   
  heatmap_thr_obs::Observable 
  heatmap_mat_obs::Observable 
  species_weight_obs
end
function display(model::ParticleModel; PORT=8080, HOST="0.0.0.0", TYPE_COLORS=[colorant"#e74c3c", colorant"#3498db", colorant"#2ecc71", colorant"#f39c12", colorant"#9b59b6", colorant"#1abc9c", colorant"#e91e63", colorant"#cddc39", colorant"#00bcd4",])
   TARGET_DT = 1.0 / 60.0

  app = App() do session::Session

    model_ref = Ref(model)
    world_size = model_ref[].WORLD_SIZE
      MAX_TYPES=length(TYPE_COLORS)

    colors_for(pt) = [TYPE_COLORS[mod1(Int(t), MAX_TYPES)] for t in pt]
    MAX_WEIGHT_STEPS = 100

    function make_display_mat(m)
      nt = Int(m.num_types)
      out = fill(NaN32, MAX_TYPES, MAX_TYPES)
      out[1:nt, 1:nt] .= m.attraction_matrix
      return out
    end

    # ── Observables ───────────────────────────────────────────────────────────
    download_positions!(model_ref[])
    dm = DisplayModel(
      Observable(model_ref[].cpu_px),#xs   
      Observable(model_ref[].cpu_py),#ys
      Observable(colors_for(get_ptypes(model_ref[]))),#cols
      Observable(make_display_mat(model_ref[])),#mat_obs
      Observable(false),#is_running
      Observable("Step: 0"),#step_obs
      #Observable("FPS: --"),#fps_obs
      Observable(model_ref[].max_radius),#radius_obs
      Observable(false),#circle_vis_obs
      Observable(Int(model_ref[].num_types)),#nt_obs
      Observable(false),#heatmap_on_obs
      Observable(8),#heatmap_n_obs
      Observable(10),#heatmap_thr_obs
      Observable(zeros(Float32, 64, 64)),#heatmap_math_obs

      [Observable(MAX_WEIGHT_STEPS ÷ MAX_TYPES) for _ in 1:MAX_TYPES],
    )
    # ── Figure: full-page, no padding ─────────────────────────────────────────
    set_theme!(theme_dark())
    fig = Figure(size=(1400, 900), backgroundcolor=:black,
                 figure_padding=0)

    ax_sim = Axis(fig[1, 1];
      backgroundcolor=:black, aspect=DataAspect(),
      xgridvisible=false, ygridvisible=false,
      leftspinevisible=false, rightspinevisible=false,
      bottomspinevisible=false, topspinevisible=false,
      xticksvisible=false, yticksvisible=false,
      xticklabelsvisible=false, yticklabelsvisible=false,
    )
    xlims!(ax_sim, 0f0, world_size)
    ylims!(ax_sim, 0f0, world_size)
    ax_sim.limits[] = (nothing, nothing, nothing, nothing)

    deregister_interaction!(ax_sim, :rectanglezoom)

    rect_overlay = Observable(Point2f[])
    lines!(ax_sim, rect_overlay; color=:white, linewidth=1, linestyle=:dash,
           xautolimits=false, yautolimits=false)

    drag_origin    = Ref{Union{Nothing, Point2f}}(nothing)
    last_mouse_pos = Ref{Point2f}(Point2f(0, 0))

    # Always track current axis-space position from raw pointer-move events.
    # We store it ourselves so release/press handlers never call mouseposition()
    # directly (which can return a stale cached value after pointer capture ends).
    on(events(ax_sim.scene).mouseposition) do _
      last_mouse_pos[] = mouseposition(ax_sim)
      start = drag_origin[]
      start === nothing && return
      finish = last_mouse_pos[]
      rect_overlay[] = Point2f[
        start, Point2f(finish[1], start[2]),
        finish, Point2f(start[1], finish[2]), start,
      ]
    end

    on(events(ax_sim.scene).mousebutton) do event
      event.button == Mouse.left || return
      if event.action == Mouse.press
        mp = last_mouse_pos[]
        fl = ax_sim.finallimits[]
        in_axis = fl.origin[1] <= mp[1] <= fl.origin[1] + fl.widths[1] &&
                  fl.origin[2] <= mp[2] <= fl.origin[2] + fl.widths[2]
        in_axis && (drag_origin[] = mp)
      elseif event.action == Mouse.release
        start = drag_origin[]
        drag_origin[] = nothing
        rect_overlay[] = Point2f[]
        start === nothing && return
        finish = last_mouse_pos[]
        fl = ax_sim.finallimits[]
        min_drag = min(fl.widths[1], fl.widths[2]) * 0.02f0
        (abs(finish[1] - start[1]) < min_drag || abs(finish[2] - start[2]) < min_drag) && return
        x1, x2 = minmax(start[1], finish[1])
        y1, y2 = minmax(start[2], finish[2])
        xlims!(ax_sim, x1, x2)
        ylims!(ax_sim, y1, y2)
        ax_sim.limits[] = (nothing, nothing, nothing, nothing)
      end
    end


    # ── Double-buffer ─────────────────────────────────────────────────────────
    np = Int(model_ref[].num_particles)
    stage_px   = Vector{Float32}(undef, np)
    stage_py   = Vector{Float32}(undef, np)
    stage_cols = Vector{eltype(TYPE_COLORS)}(undef, np)
    stage_hm   = Ref{Union{Nothing, Matrix{Float32}}}(nothing)
    frame_dirty = Threads.Atomic{Int}(0)
    stage_lock  = ReentrantLock()

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
        end
        notify(dm.xs); notify(dm.ys); notify(dm.cols)
      end
    end

    scatter!(ax_sim, dm.xs, dm.ys; color=dm.cols, markersize=5, strokewidth=0,
             xautolimits=false, yautolimits=false)

    # ── Density heatmap overlay ───────────────────────────────────────────────
    heatmap_display = @lift begin
      m = $(dm.heatmap_mat_obs)
      replace(m, 0.0f0 => NaN32)
    end
    heatmap_x_obs = Observable(LinRange(0f0, world_size, size(dm.heatmap_mat_obs[], 1) + 1))
    heatmap_y_obs = Observable(LinRange(0f0, world_size, size(dm.heatmap_mat_obs[], 2) + 1))
    on(dm.heatmap_mat_obs) do m
      N = size(m, 1)
      r = LinRange(0f0, world_size, N + 1)
      heatmap_x_obs[] = r
      heatmap_y_obs[] = r
    end
    heatmap_alpha = @lift($(dm.heatmap_on_obs) ? 0.45f0 : 0.0f0)
    hm_overlay = heatmap!(ax_sim, heatmap_x_obs, heatmap_y_obs, heatmap_display;
                          colormap=:hot, lowclip=:transparent)
    on(heatmap_alpha) do a; hm_overlay.alpha = a; end
    hm_overlay.alpha = heatmap_alpha[]

    # ── View-distance circle ──────────────────────────────────────────────────
    let circle_pts = 128, cx = Float32(world_size) * 0.5f0, cy = Float32(world_size) * 0.5f0
      circle_xy = @lift begin
        r = $(dm.radius_obs)
        ts = LinRange(0f0, 2f0 * Float32(π), circle_pts + 1)
        Point2f.(cx .+ r .* cos.(ts), cy .+ r .* sin.(ts))
      end
      circle_color = @lift($(dm.circle_vis_obs) ? RGBAf(1.0,1.0,1.0,0.45) : RGBAf(0,0,0,0))
      lines!(ax_sim, circle_xy; color=circle_color, linewidth=1.5, linestyle=:dash)
    end

 #   text!(ax_sim, 0.01f0, 0.02f0;
 #     text=fps_obs, color=:white, fontsize=13, align=(:left, :bottom))

    # ── Sliders (values only, DOM built later) ────────────────────────────────
    radius_vals   = LinRange(0.02f0, 0.125f0, 200)
    dt_vals       = LinRange(0.0005f0, 0.004f0, 200)
    friction_vals = LinRange(0.0f0, 0.5f0, 200)

    radius_default   = argmin(abs.(collect(radius_vals) .- model_ref[].max_radius))
    dt_default       = argmin(abs.(collect(dt_vals) .- model_ref[].dt))
    friction_default = argmin(abs.(collect(friction_vals) .- model_ref[].friction))

    sl_radius   = Slider(1:200; value=radius_default)
    sl_dt       = Slider(1:200; value=dt_default)
    sl_friction = Slider(1:200; value=friction_default)
    sl_spf      = Slider(1:32;  value=model_ref[].steps_per_frame)
    sl_types    = Slider(2:MAX_TYPES; value=Int(model_ref[].num_types))
    sl_hm_n     = Slider(3:16; value=dm.heatmap_n_obs[])
    sl_hm_thr   = Slider(1:200; value=dm.heatmap_thr_obs[])

    radius_display   = Observable(string(round(radius_vals[sl_radius.value[]], digits=3)))
    dt_display       = Observable(string(round(dt_vals[sl_dt.value[]], digits=4)))
    friction_display = Observable(string(round(friction_vals[sl_friction.value[]], digits=3)))
    spf_display      = Observable(string(sl_spf.value[]))
    types_display    = Observable(string(sl_types.value[]))
    hm_n_display     = Observable(string(dm.heatmap_n_obs[]))
    hm_thr_display   = Observable(string(dm.heatmap_thr_obs[]))


    radius_touch = Ref(0)
    on(sl_radius.value) do idx
      v = radius_vals[clamp(idx, 1, 200)]
      model_ref[].max_radius = v
      radius_display[] = string(round(v, digits=3))
      dm.radius_obs[] = v
      dm.circle_vis_obs[] = true
      radius_touch[] += 1
      my_touch = radius_touch[]
      Threads.@spawn begin
        sleep(1.5)
        radius_touch[] == my_touch && (dm.circle_vis_obs[] = false)
      end
    end
    on(sl_dt.value) do idx
      v = dt_vals[clamp(idx, 1, 200)]
      model_ref[].dt = v
      dt_display[] = string(round(v, digits=4))
    end
    on(sl_friction.value) do idx
      v = friction_vals[clamp(idx, 1, 200)]
      model_ref[].friction = v
      friction_display[] = string(round(v, digits=3))
    end
    on(sl_spf.value) do v
      model_ref[].steps_per_frame = v
      spf_display[] = string(v)
    end
    on(sl_hm_n.value)   do v; dm.heatmap_n_obs[] = v;   hm_n_display[] = string(v);   end
    on(sl_hm_thr.value) do v; dm.heatmap_thr_obs[] = v; hm_thr_display[] = string(v); end

    function recreate_model!(nt)
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
      dm.mat_obs[] = make_display_mat(m)
      dm.nt_obs[] = nt; dm.step_obs[] = "Step: 0"
      dm.is_running[] = was
    end
    on(sl_types.value) do nt; types_display[] = string(nt); recreate_model!(nt); end

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_play       = Button("▶  Start")
    btn_rand       = Button("Randomize Matrix")
    btn_reset      = Button("Reset Particles")
    btn_both       = Button("Randomize + Reset")
    btn_heatmap    = Button("Density Map: OFF")
    btn_apply_weights = Button("Apply Makeup")

    function current_weights()
      nt = dm.nt_obs[]
      raw = Float32[dm.species_weight_obs[t][] for t in 1:nt]
      s = sum(raw); s == 0 && return fill(1.0f0/nt, nt)
      return raw ./ s
    end

    function refresh_display!()
      m = model_ref[]
      download_positions!(m)
      copy!(dm.xs[], m.cpu_px); copy!(dm.ys[], m.cpu_py)
      dm.cols[] = colors_for(get_ptypes(m))
      notify(dm.xs); notify(dm.ys)
      dm.mat_obs[] = make_display_mat(m)
    end

    on(btn_play) do _
      dm.is_running[] = !dm.is_running[]
      btn_play.content[] = dm.is_running[] ? "⏸  Pause" : "▶  Start"
    end
    on(btn_heatmap) do _
      dm.heatmap_on_obs[] = !dm.heatmap_on_obs[]
      btn_heatmap.content[] = dm.heatmap_on_obs[] ? "Density Map: ON" : "Density Map: OFF"
    end
    on(btn_rand) do _
      randomize_matrix!(model_ref[])
      dm.mat_obs[] = make_display_mat(model_ref[])
    end
    on(btn_reset) do _
      was = dm.is_running[]; dm.is_running[] = false
      model_ref[].species_weights = current_weights()
      reset_particles!(model_ref[])
      refresh_display!()
      dm.step_obs[] = "Step: 0"; dm.is_running[] = was
    end
    on(btn_both) do _
      was = dm.is_running[]; dm.is_running[] = false
      model_ref[].species_weights = current_weights()
      randomize_matrix!(model_ref[]); reset_particles!(model_ref[])
      refresh_display!()
      dm.step_obs[] = "Step: 0"; dm.is_running[] = was
    end
    on(btn_apply_weights) do _
      was = dm.is_running[]; dm.is_running[] = false
      model_ref[].species_weights = current_weights()
      reset_particles!(model_ref[])
      refresh_display!()
      dm.step_obs[] = "Step: 0"; dm.is_running[] = was
    end


    # ── Sim loop ──────────────────────────────────────────────────────────────
    Threads.@spawn begin
      fps_smooth = 0.0
      t_last = time()
      while true
        try
          t0 = time()
          elapsed = t0 - t_last; t_last = t0
#          if elapsed > 0
#            inst = 1.0 / elapsed
#            fps_smooth = fps_smooth == 0.0 ? inst : 0.1 * inst + 0.9 * fps_smooth
#            m = model_ref[]
#            fps_obs[] = "FPS: $(round(Int, fps_smooth))  ×$(m.steps_per_frame)"
#          end
          if dm.is_running[]
            m = model_ref[]
            for _ in 1:m.steps_per_frame; model_step!(m); end
            download_positions!(m)
            new_cols = colors_for(get_ptypes(m))
            new_hm = dm.heatmap_on_obs[] ? heatmap(m, dm.heatmap_n_obs[], dm.heatmap_thr_obs[]) : nothing
            lock(stage_lock) do
              copy!(stage_px, m.cpu_px); copy!(stage_py, m.cpu_py)
              copy!(stage_cols, new_cols); stage_hm[] = new_hm
            end
            Threads.atomic_xchg!(frame_dirty, 1)
            dm.step_obs[] = "Step: $(m.step_count)"
          end
          spent = time() - t0
          rem = TARGET_DT - spent
          rem > 0 && sleep(rem)
        catch e
          @error "Sim loop error" exception=(e, catch_backtrace())
          sleep(0.5)
        end
      end
    end

    # ── Species weight observables ────────────────────────────────────────────
    weight_pct_obs = Observable(fill("", MAX_TYPES))
    function update_pct_obs()
      nt = dm.nt_obs[]
      raw = [Float32(dm.species_weight_obs[t][]) for t in 1:nt]
      s = sum(raw); out = fill("", MAX_TYPES)
      for t in 1:nt
        out[t] = s == 0 ? "0%" : string(round(Int, 100 * raw[t] / s)) * "%"
      end
      weight_pct_obs[] = out
    end
    update_pct_obs()
    on(dm.nt_obs) do _; update_pct_obs(); end
    for t in 1:MAX_TYPES; on(dm.species_weight_obs[t]) do _; update_pct_obs(); end; end

    # ── DOM helpers ───────────────────────────────────────────────────────────
    lbl(s) = DOM.span(s; style="color:#aaa; font-size:12px; white-space:nowrap;")
    val(ob) = DOM.span(ob; style="color:#ccc; font-size:11px; min-width:38px; display:inline-block; text-align:right;")

    function row(children...; gap="8px")
      DOM.div(children...;
        style="display:flex; align-items:center; gap:$gap; padding:3px 0;")
    end

    section_style = "padding:8px 0; border-bottom:1px solid #333;"
    sec_title(s) = DOM.div(s; style="color:#ddd; font-size:11px; font-weight:bold; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;")


    # ── Attraction matrix panel (inline Makie axis in DOM) ────────────────────
    fig_mat = Figure(size=(260, 240), backgroundcolor=RGBAf(0,0,0,0),
                     figure_padding=4)
    ax_mat = Axis(fig_mat[1, 1];
      title="Attraction  (L +0.1 / R −0.1)", titlecolor=:white, titlesize=11,
      aspect=DataAspect(), xgridvisible=false, ygridvisible=false,
      backgroundcolor=RGBAf(0,0,0,0),
      xlabel="seen", ylabel="seeing",
      xlabelcolor=:gray60, ylabelcolor=:gray60,
      xlabelsize=10, ylabelsize=10,
      xticklabelsvisible=false, yticklabelsvisible=false,
      xticksvisible=false, yticksvisible=false,
      limits=(0f0, Float32(MAX_TYPES)+0.5f0, 0f0, Float32(MAX_TYPES)+0.5f0),
    )
    heatmap!(ax_mat, 1:MAX_TYPES, 1:MAX_TYPES, @lift($(dm.mat_obs)');
             colormap=:RdBu, colorrange=(-1f0, 1f0), nan_color=:black)
    dim = RGBAf(0.25, 0.25, 0.25, 0.4)
    for t in 1:MAX_TYPES
      lc = @lift(t <= $(dm.nt_obs) ? TYPE_COLORS[t] : dim)
      text!(ax_mat, Float32(t), 0.22f0; text="T$t", color=lc,
            align=(:center,:center), fontsize=8, font=:bold)
      text!(ax_mat, 0.22f0, Float32(t); text="T$t", color=lc,
            align=(:center,:center), fontsize=8, font=:bold)
    end
    Colorbar(fig_mat[1, 2]; colormap=:RdBu, limits=(-1f0,1f0),
             tickcolor=:white, ticklabelcolor=:white, ticklabelsize=9,
             label="", width=10)
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
        dm.mat_obs[] = make_display_mat(m)
      end
    end

    # ── Weight sliders for makeup panel ──────────────────────────────────────
    function make_weight_slider(t)
      col = "#" * hex(TYPE_COLORS[mod1(t, MAX_TYPES)])
      pct_obs_t = @lift($weight_pct_obs[t])
      vis = @lift($(dm.nt_obs) >= t ?
        "display:flex; flex-direction:column; align-items:center; gap:1px;" : "display:none;")
      sl = Slider(0:MAX_WEIGHT_STEPS; value=dm.species_weight_obs[t][])
      on(sl.value) do v; dm.species_weight_obs[t][] = v; end
      DOM.div(
        DOM.span("T$t"; style="color:$col; font-size:10px; font-weight:bold;"),
        sl,
        DOM.span(pct_obs_t; style="color:#888; font-size:9px; min-width:24px; text-align:center;");
        style=vis,
      )
    end
    weight_sliders = DOM.div(
      [make_weight_slider(t) for t in 1:MAX_TYPES]...;
      style="display:flex; flex-direction:row; gap:3px; align-items:flex-end; flex-wrap:wrap;",
    )


    # ── Collapsible settings panel (top-left overlay) ────────────────────────
    panel_open = Observable(true)

    settings_body = DOM.div(
      # ── Matrix section ───────────────────────────────────────────────
      DOM.div(
        sec_title("Attraction Matrix"),
        fig_mat;
        style=section_style,
      ),
      # ── Sim params section ───────────────────────────────────────────
      DOM.div(
        sec_title("Simulation"),
        row(lbl("Types"),    sl_types,   val(types_display)),
        row(lbl("Distance"), sl_radius,  val(radius_display)),
        row(lbl("dt"),       sl_dt,      val(dt_display)),
        row(lbl("Friction"), sl_friction, val(friction_display)),
        row(lbl("Steps/fr"), sl_spf,     val(spf_display));
        style=section_style,
      ),
      # ── Heatmap section ──────────────────────────────────────────────
      DOM.div(
        sec_title("Density Map"),
        row(btn_heatmap),
        row(lbl("Grid n"),    sl_hm_n,   val(hm_n_display)),
        row(lbl("Threshold"), sl_hm_thr, val(hm_thr_display));
        style=section_style,
      ),
      # ── Makeup section ───────────────────────────────────────────────
      DOM.div(
        sec_title("Particle Makeup"),
        weight_sliders,
        DOM.div(btn_apply_weights; style="margin-top:6px;");
        style="padding:8px 0;",
      );
      style="overflow-y:auto; max-height:calc(100vh - 80px);",
    )

    toggle_btn = Button(panel_open[] ? "◀ Settings" : "▶ Settings")
    on(toggle_btn) do _
      panel_open[] = !panel_open[]
      toggle_btn.content[] = panel_open[] ? "◀ Settings" : "▶ Settings"
    end

    panel_vis = @lift($panel_open ?
      "display:block;" : "display:none;")

    settings_panel = DOM.div(
      # Header bar
      DOM.div(
        DOM.span("⚙ Settings"; style="color:#eee; font-size:13px; font-weight:bold;"),
        toggle_btn;
        style="display:flex; justify-content:space-between; align-items:center; padding:6px 8px; background:#1a1a1a; border-bottom:1px solid #333; cursor:pointer;",
      ),
      # Collapsible body
      DOM.div(settings_body; style=panel_vis);
      style="""
        position:fixed; top:12px; left:12px; z-index:100;
        width:300px;
        background:rgba(10,10,10,0.92);
        border:1px solid #444;
        border-radius:8px;
        font-family:monospace;
        box-shadow:0 4px 24px rgba(0,0,0,0.7);
        max-height:calc(100vh - 24px);
        overflow:hidden;
      """,
    )

    # ── Bottom-right sim controls ─────────────────────────────────────────────
    br_controls = DOM.div(
      row(
        btn_play, btn_rand, btn_reset, btn_both;
        gap="6px",
      ),
      DOM.div(
        DOM.span(dm.step_obs; style="color:#888; font-size:11px; font-family:monospace;");
        style="text-align:right; padding-top:2px;",
      );
      style="""
        position:fixed; bottom:14px; right:16px; z-index:100;
        background:rgba(10,10,10,0.88);
        border:1px solid #444;
        border-radius:8px;
        padding:8px 12px;
        font-family:monospace;
        box-shadow:0 4px 16px rgba(0,0,0,0.6);
      """,
    )

    # ── Assemble page ─────────────────────────────────────────────────────────
    return DOM.div(
      fig,
      settings_panel,
      br_controls;
      style="width:100vw; height:100vh; overflow:hidden; background:#000; position:relative;",
    )
  end

  # ── Server ────────────────────────────────────────────────────────────────
  server = Server(app, HOST, PORT)
  println("http://localhost:$(PORT)")
  wait(server)
end

