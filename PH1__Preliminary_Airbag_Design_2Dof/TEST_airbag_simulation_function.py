from airbag_simulation_function import simulate_airbag
import numpy as np
import contextlib
import os

# ---------------------------- output options ---------------------------------
MAKE_ANIMATION          = True   # bag-only impact animation
SAVE_ANIMATION          = True   # ... and export it (.mp4 if ffmpeg, else .gif)

MAKE_COMBINED_ANIMATION = True   # 3x3 plot matrix w/ time cursor + bag, side by side
SAVE_COMBINED_ANIMATION = True   # ... and export it (.mp4 if ffmpeg, else .gif)
SHOW_ANIMATIONS         = True   # pop up an interactive window for each animation

SAVE_PLOT_MATRIX        = True   # export the static 3x3 plot matrix as an image
PLOT_MATRIX_FORMATS     = ("png",)   # add "pdf" / "svg" for vector output
PLOT_MATRIX_DPI         = 150

MAX_FRAMES   = 200   # animation frames (physics runs thousands of adaptive steps)
ANIM_FPS     = 25
ANIM_DPI     = 90    # raise for a crisper file, lower to shrink it
CURVE_MAX_PTS = 4000 # background curves are decimated to this many points in the
                     # combined animation: blit=False redraws all 9 of them every
                     # frame, so full 10k-point curves dominate the render time.
                     # The saved static matrix always uses full resolution.
OUTPUT_DIR   = None  # None -> alongside this script; else an absolute path
# -----------------------------------------------------------------------------

# Environment
g = 9.81
P_amb = 101325
# mu  = 0.4          # [-]   bag-fabric / ground sliding friction coefficient


# Payload 
M_payload = 0.5
u0 = 17.2        # [m/s] initial vertical velocity (design parameter)
ux0 = 5          # [m/s] initial horizontal velocity (design parameter)


# Geometry
shape = 1        # 1 = cylinder, 2 = sphere
D0 = 0.8   # [m] initial airbag height (diameter) 
L0 = 0.1043377808026541       # [m] airbag axial length, not used if shape == 2 (sphere)
d_or = 0.095492405598105    # [m] orifice diameter

# Fabric 
d_fabric = 0.0005238632242683665 # [m] fabric thickness
sigma_fabric = 250e6   # [Pa] effective membrane strength (×0.002 m = 50000 N/m, previously T_fabric)
rho_fabric   = 1100.0    # [kg/m^3] 
# c_abrasion   = 3.0e4     # [J/m] abrasion energy per unit thickness (×0.002 m = 5000 J)


# Gas parameters (air)
P0 = 131781.32180997377     # [Pa] initial airbag inflation pressure
# P0 = 130000
T0 = 284.5695298438421      # [K] initial airbag temperature
R_gas = 2077.0    # [J/(kg K)]
gamma = 1.66


# Time and other settings
dt = 1e-5
# dt = [1e-3, 5e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
# timesteps_and_accelerations_list = []
t_max = 1
make_plots=True
verbose=True


LOG_PATH = r"C:\TEMP\full_log.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

with open(LOG_PATH, "w") as f:
    with contextlib.redirect_stdout(f):
        res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                               sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                               dt, t_max, make_plots, verbose, a_allow=500, ux0=ux0)

"""
for dt_test in dt:
    res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g, dt_test, t_max, make_plots, verbose, a_allow=500, ux0=ux0)
    print("Calculating for", dt_test, "s")
    timesteps_and_accelerations_list.append([dt_test, res['a_peak']])
print(timesteps_and_accelerations_list)

"""

print("done — check C:\\TEMP\\full_log.txt")

print(f"{'Fabric mass':<14}: {res['m_bag']:.5f} kg")
print(f"{'Gas mass':<14}: {res['m_gas0']:.5f} kg")
print(f"{'Total mass':<14}: {res['m_total']:.5f} kg")
print(f"Peak accel  : {res['a_peak']:.5f} m/s^2  ({res['a_peak']/9.81:.2f} g)")
print(f"{'u_residual':<14}: {res['u_residual']:.5f} m/s")
print(f"Rebounded    : {res['rebounded']}")
print(f"Bottomed    : {res['bottomed']}")
print(f"Burst    : {res['burst']}")
print(f"Survived    : {res['survived']}")

# =============================================================================
# Post-processing: static plot matrix + animations
# =============================================================================
# simulate_airbag() (airbag_simulation_function.py) does NOT return any
# per-step history — only final/peak scalars. Since that file must not be
# modified, the stepping algorithm is replayed here, standalone, using only the
# same building blocks it already uses (airbag_geometry_from_h, thermo_step),
# purely to reconstruct the per-step histories needed for the figures below.
# This duplicate MUST be kept in sync by hand with airbag_simulation_function.py's
# loop if that algorithm ever changes.
#
# The bag silhouette drawn each frame is a schematic reconstruction from the
# scalar crush state h (via airbag_geometry_from_h), not a physical deformation
# mesh — same for the sphere's "buried circle" treatment.
# =============================================================================

NEED_REPLAY = MAKE_ANIMATION or MAKE_COMBINED_ANIMATION or SAVE_PLOT_MATRIX

if NEED_REPLAY:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    from airbag_geometry import airbag_geometry_from_h
    from airbag_thermo import thermo_step

    OUT_DIR = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUT_DIR, exist_ok=True)

    a_allow_plot = 500

    def _replay_for_animation(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                               sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                               dt, t_max, a_allow, ux0):
        """Standalone replica of airbag_simulation_function.simulate_airbag's loop,
        kept only to record the same nine histories that simulate_airbag plots."""
        R0 = D0 / 2.0
        k_shape = 1.0 if shape == 1 else 0.5
        T_fabric = sigma_fabric * d_fabric
        P_burst = P_amb + T_fabric / (k_shape * R0)

        theta = np.arctan2(ux0, u0)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)

        t = 0.0
        h = 0.0
        y_pen = 0.0
        x = 0.0
        u_x, u_y = ux0, u0
        burst = bottomed = rebounded = False
        fail_t = None

        geom0 = airbag_geometry_from_h(h, D0, L0, shape=shape)
        A_prev, V_prev = geom0["A_t"], geom0["V_t"]
        m_gas = P0 * V_prev / (R_gas * T0)
        T_gas, P_bag, rho_prev = T0, P0, m_gas / V_prev

        # same ordering/seeding as simulate_airbag: t=0 sample, accelerations NaN
        t_hist, h_hist, x_hist = [t], [h], [x]
        y_pen_hist, P_hist, a_hist = [y_pen], [P_bag], [np.nan]
        u_hist, un_hist = [u_y], [u_x * sin_theta + u_y * cos_theta]
        ares_hist, m_hist, V_hist = [np.nan], [m_gas], [V_prev]

        while t < t_max:
            DT_FINE, DT_COARSE = 1e-5, 2e-4
            A_TRIGGER_fine, A_TRIGGER_coarse = 0.03 * a_allow, 0.4 * a_allow
            a_prev = a_hist[-1] if not np.isnan(a_hist[-1]) else 0.0
            if abs(a_prev) > A_TRIGGER_fine:
                dt = DT_FINE
            elif abs(a_prev) < A_TRIGGER_coarse:
                dt = DT_COARSE

            F_n = max((P_bag - P_amb) * A_prev, 0.0)
            F_x = -F_n * sin_theta
            F_y = -F_n * cos_theta
            a_y = F_y / M_payload + g
            a_x = F_x / M_payload
            a_res = np.sqrt(a_x ** 2 + a_y ** 2)

            u_x_new = u_x + a_x * dt
            u_y_new = u_y + a_y * dt
            u_n_new = u_x_new * sin_theta + u_y_new * cos_theta

            h_new = min(max(h + u_n_new * dt, 0.0), D0)
            y_pen_new = y_pen + u_y_new * dt
            x_new = x + u_x_new * dt

            geom = airbag_geometry_from_h(h_new, D0, L0, shape=shape)
            A_new, V_new = max(geom["A_t"], 0.0), geom["V_t"]

            thermo = thermo_step(m_prev=m_gas, V_t=V_new, gamma=gamma, P_amb=P_amb, dt=dt,
                                  A_or=(np.pi / 4) * d_or ** 2, P_prev=P_bag, rho_prev=rho_prev,
                                  T_prev=T_gas, R_gas=R_gas, verbose=False)
            P_new = max(thermo["P_bag"], P_amb)

            t_next = t + dt
            t_hist.append(t_next); h_hist.append(h_new); x_hist.append(x_new)
            y_pen_hist.append(y_pen_new); P_hist.append(P_new); a_hist.append(a_y)
            u_hist.append(u_y_new); un_hist.append(u_n_new); ares_hist.append(a_res)
            m_hist.append(thermo["m_t"]); V_hist.append(V_new)

            if P_new > P_burst:
                burst = True
                fail_t = t_next
                break
            if h_new >= D0:
                bottomed = True
                fail_t = t_next
                break
            if y_pen_new <= 0.0:
                rebounded = True
                fail_t = t_next
                break

            t = t_next
            h, y_pen, u_x, u_y, x = h_new, y_pen_new, u_x_new, u_y_new, x_new
            A_prev, V_prev = A_new, V_new
            P_bag, rho_prev, T_gas, m_gas = P_new, thermo["rho_t"], thermo["T_t"], thermo["m_t"]

        arr = np.asarray
        return dict(t_hist=arr(t_hist), h_hist=arr(h_hist), x_hist=arr(x_hist),
                    y_pen_hist=arr(y_pen_hist), P_hist=arr(P_hist), a_hist=arr(a_hist),
                    u_hist=arr(u_hist), un_hist=arr(un_hist), ares_hist=arr(ares_hist),
                    m_hist=arr(m_hist), V_hist=arr(V_hist),
                    P_burst=P_burst, burst=burst, bottomed=bottomed, rebounded=rebounded,
                    fail_t=fail_t)

    replay = _replay_for_animation(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                                    sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                                    dt, t_max, a_allow=a_allow_plot, ux0=ux0)

    # Sanity check: the replay must agree with the real simulate_airbag() call above,
    # since it's meant to reproduce the exact same run for post-processing only.
    if (replay["burst"], replay["bottomed"], replay["rebounded"]) != (res["burst"], res["bottomed"], res["rebounded"]):
        print("[post-processing] WARNING: replay outcome does not match simulate_airbag() result "
              "— the standalone replay driver may have drifted out of sync with "
              "airbag_simulation_function.py's algorithm.")

    t_hist = replay["t_hist"]
    h_hist, x_hist, y_pen_hist = replay["h_hist"], replay["x_hist"], replay["y_pen_hist"]
    P_hist, P_burst = replay["P_hist"], replay["P_burst"]
    n_steps = len(h_hist)
    last_idx = n_steps - 1

    is_failure = replay["burst"] or res.get("abraded", False)

    # ---- shared frame selection -------------------------------------------
    frame_idx = np.linspace(0, last_idx, min(n_steps, MAX_FRAMES)).astype(int)
    frame_idx = np.unique(frame_idx)          # monotonic & includes last_idx exactly
    if is_failure:
        frame_idx = np.concatenate([frame_idx, np.full(30, last_idx, dtype=int)])  # hold on last frame

    # ---- shared helpers ----------------------------------------------------
    def _pressure_color(P, P_burst):
        ratio = float(np.clip(P / P_burst, 0.0, 1.0))
        if ratio < 0.5:
            f = ratio / 0.5
            return (f, 1.0, 0.0)          # green -> yellow
        else:
            f = (ratio - 0.5) / 0.5
            return (1.0, 1.0 - f, 0.0)    # yellow -> red

    def _stadium_outline(L_t, D_t, n_arc=60):
        """Single closed outline of the stadium cross-section: right semicircular
        cap, straight top, left semicircular cap, straight bottom. Drawn as ONE
        polygon so no construction lines appear inside the bag."""
        r = D_t / 2.0
        th_right = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
        th_left = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)
        right = np.column_stack((L_t / 2 + r * np.cos(th_right), r + r * np.sin(th_right)))
        left = np.column_stack((-L_t / 2 + r * np.cos(th_left), r + r * np.sin(th_left)))
        return np.vstack((right, left))

    # Drawing window: wide enough for the payload's horizontal travel and for the
    # bag at full crush (half-width -> (L_t + D_t)/2 with L_t = pi*D0/2), no wider.
    # Excess x-padding is expensive here: the bag axes uses aspect='equal', so a
    # wider x-range letterboxes the drawing inside its tall 3-row cell.
    x_span = max(np.ptp(x_hist), D0)
    x_pad = 0.35 * D0
    XLIM = (-0.5 * np.pi * D0 / 2 - x_pad, x_span + x_pad)
    YLIM = (-0.05 * D0, 1.15 * D0)

    def _save_anim(anim, basename, fps=ANIM_FPS, dpi=ANIM_DPI):
        """Save BEFORE plt.show(): show() runs the animation to exhaustion
        (repeat=False), after which anim.save() writes a blank/partial file.
        Writes an absolute path so the file never lands in whatever the
        interpreter's cwd happens to be (the workspace root, under VS Code)."""
        mp4_path = os.path.join(OUT_DIR, basename + ".mp4")
        gif_path = os.path.join(OUT_DIR, basename + ".gif")
        try:
            anim.save(mp4_path, fps=fps, dpi=dpi)
            print(f"Saved animation -> {mp4_path}")
        except Exception as e:
            # No ffmpeg on this machine -> fall back to the always-available
            # Pillow GIF writer so an animation is still produced.
            print(f"[save] mp4 failed ({type(e).__name__}: {e}); falling back to GIF")
            anim.save(gif_path, writer=mpl_animation.PillowWriter(fps=fps), dpi=dpi)
            print(f"Saved animation -> {gif_path}")

    # ---- the nine series, in the same order simulate_airbag() plots them ----
    PLOT_SPECS = [
        (replay["h_hist"],     "h [m]",         "Crush Along n"),
        (replay["y_pen_hist"], "y_pen [m]",     "Vertical Penetration"),
        (replay["u_hist"],     "u_y [m/s]",     "Vertical Velocity"),
        (replay["un_hist"],    "u_n [m/s]",     "Velocity Along n"),
        (replay["a_hist"],     "a_y [m/s^2]",   "Vertical Acceleration"),
        (replay["ares_hist"],  "a_res [m/s^2]", "Resultant Acceleration"),
        (replay["P_hist"],     "P_bag [Pa]",    "Internal Pressure"),
        (replay["m_hist"],     "m [kg]",        "Gas Mass"),
        (replay["V_hist"],     "V [m^3]",       "Airbag Volume"),
    ]

    def _decorate_matrix(axes, label_fs=9, title_fs=10):
        """Reference lines / formatting shared by the static matrix and the
        animated left panel (mirrors simulate_airbag's make_plots block)."""
        for ax, (_, ylabel, title) in zip(axes, PLOT_SPECS):
            ax.set_ylabel(ylabel, fontsize=label_fs)
            ax.set_title(title, fontsize=title_fs)
            ax.grid(True)
            ax.tick_params(labelsize=label_fs - 1)
        axes[0].axhline(D0, color='r', ls='--', lw=1, label=f'D0 = {D0:.3f} m')
        axes[0].legend(fontsize=label_fs - 1)
        axes[4].axhline(-a_allow_plot, color='r', ls='--', lw=1)
        axes[5].axhline(a_allow_plot, color='r', ls='--', lw=1,
                        label=f'a_allow = {a_allow_plot:.0f}')
        axes[5].legend(fontsize=label_fs - 1)
        axes[6].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        axes[6].yaxis.get_offset_text().set_fontsize(label_fs - 1)
        for ax in axes[6:9]:
            ax.set_xlabel("Time [s]", fontsize=label_fs)


# =============================================================================
# (a) static 3x3 plot matrix
# =============================================================================
if SAVE_PLOT_MATRIX:
    fig_m, axs_m = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ax, (data, _, _) in zip(axs_m.flat, PLOT_SPECS):
        ax.plot(t_hist, data)
    _decorate_matrix(list(axs_m.flat))
    fig_m.suptitle("Airbag Simulation (2-DOF oblique)", fontsize=13)
    fig_m.tight_layout()
    for ext in PLOT_MATRIX_FORMATS:
        out = os.path.join(OUT_DIR, f"airbag_plot_matrix.{ext}")
        fig_m.savefig(out, dpi=PLOT_MATRIX_DPI, bbox_inches="tight")
        print(f"Saved plot matrix -> {out}")
    plt.close(fig_m)


# =============================================================================
# (b) bag-only impact animation
# =============================================================================
if MAKE_ANIMATION:
    fig, ax = plt.subplots(figsize=(7, 6))

    def _draw_frame(frame_number):
        i = int(frame_idx[frame_number])
        ax.clear()
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(f"t = {t_hist[i]:.4f} s")

        color = _pressure_color(P_hist[i], P_burst)

        if shape == 1:
            # cylinder: side-view "stadium" cross-section, flattening as h grows
            ax.axhline(0, color='saddlebrown', lw=2, zorder=1)
            D_t = max(D0 - h_hist[i], 1e-4)
            L_t = airbag_geometry_from_h(h_hist[i], D0, L0, shape=shape)["L_t"]
            ax.add_patch(plt.Polygon(_stadium_outline(L_t, D_t), closed=True,
                                      facecolor=color, edgecolor='k', zorder=2))
            payload_y = D_t
        else:
            # sphere: fixed circle, ground line rises into it as h grows ("buried" look)
            R0 = D0 / 2.0
            ax.add_patch(plt.Circle((0, R0), R0, facecolor=color, edgecolor='k', zorder=1))
            ax.axhspan(YLIM[0], h_hist[i], color='saddlebrown', zorder=2)
            ax.axhline(h_hist[i], color='k', lw=1, zorder=3)
            payload_y = D0

        ax.add_patch(plt.Rectangle((x_hist[i] - 0.03 * D0, payload_y),
                                    0.06 * D0, 0.06 * D0, facecolor='navy', zorder=4))

        if is_failure and i == last_idx:
            reason = "BURST" if replay["burst"] else "ABRADED"
            ax.text(0.5, 0.92, f"*** {reason} at t = {replay['fail_t']:.4f} s ***",
                    transform=ax.transAxes, ha='center', color='red', fontsize=11,
                    fontweight='bold')
        return []

    anim = mpl_animation.FuncAnimation(fig, _draw_frame, frames=len(frame_idx),
                                        interval=1000 // ANIM_FPS, blit=False, repeat=False)
    if SAVE_ANIMATION:
        _save_anim(anim, "airbag_impact_animation")
    if SHOW_ANIMATIONS:
        plt.show()
    plt.close(fig)


# =============================================================================
# (c) combined animation: 3x3 plot matrix with a time cursor (left)
#                         + bag compression (right)
# =============================================================================
if MAKE_COMBINED_ANIMATION:
    fig_c = plt.figure(figsize=(17, 7))
    gs = fig_c.add_gridspec(3, 4, width_ratios=[1, 1, 1, 2.3],
                            wspace=0.34, hspace=0.45,
                            left=0.05, right=0.98, top=0.88, bottom=0.09)

    # --- left: the nine panels, drawn ONCE; only the cursor moves per frame ---
    p_axes = []
    for r in range(3):
        for c in range(3):
            p_axes.append(fig_c.add_subplot(gs[r, c], sharex=p_axes[0] if p_axes else None))

    dstep = max(1, len(t_hist) // CURVE_MAX_PTS)   # display-only decimation
    cursors, markers = [], []
    for ax, (data, _, _) in zip(p_axes, PLOT_SPECS):
        # decimated background curve, but the cursor/marker below still index the
        # FULL history, so the readout stays exact
        ax.plot(t_hist[::dstep], data[::dstep], lw=1.2, color='C0')
        cursors.append(ax.axvline(t_hist[0], color='crimson', lw=1.2, zorder=5))
        markers.append(ax.plot([t_hist[0]], [data[0]], 'o', color='crimson',
                               ms=5, zorder=6)[0])
    _decorate_matrix(p_axes, label_fs=8, title_fs=9)

    # --- right: the bag, set up ONCE; artists are mutated per frame ----------
    ax_bag = fig_c.add_subplot(gs[:, 3])
    ax_bag.set_xlim(*XLIM)
    ax_bag.set_ylim(*YLIM)
    ax_bag.set_aspect('equal')
    ax_bag.set_xlabel("x [m]")
    ax_bag.set_ylabel("height [m]")

    c0 = _pressure_color(P_hist[0], P_burst)
    if shape == 1:
        ax_bag.axhline(0, color='saddlebrown', lw=2, zorder=1)
        D_t0 = max(D0 - h_hist[0], 1e-4)
        L_t0 = airbag_geometry_from_h(h_hist[0], D0, L0, shape=shape)["L_t"]
        bag_patch = plt.Polygon(_stadium_outline(L_t0, D_t0), closed=True,
                                facecolor=c0, edgecolor='k', zorder=2)
        ax_bag.add_patch(bag_patch)
        ground_span = ground_line = None
        payload_y0 = D_t0
    else:
        R0 = D0 / 2.0
        bag_patch = plt.Circle((0, R0), R0, facecolor=c0, edgecolor='k', zorder=1)
        ax_bag.add_patch(bag_patch)
        ground_span = ax_bag.axhspan(YLIM[0], h_hist[0], color='saddlebrown', zorder=2)
        ground_line = ax_bag.axhline(h_hist[0], color='k', lw=1, zorder=3)
        payload_y0 = D0

    payload = plt.Rectangle((x_hist[0] - 0.03 * D0, payload_y0),
                            0.06 * D0, 0.06 * D0, facecolor='navy', zorder=4)
    ax_bag.add_patch(payload)

    fail_txt = ax_bag.text(0.5, 0.92, "", transform=ax_bag.transAxes, ha='center',
                           color='red', fontsize=12, fontweight='bold', zorder=7)

    def _draw_combined(frame_number):
        i = int(frame_idx[frame_number])
        t_i = t_hist[i]

        for cur, mk, (data, _, _) in zip(cursors, markers, PLOT_SPECS):
            cur.set_xdata([t_i, t_i])
            mk.set_data([t_i], [data[i]])   # NaN samples simply render nothing

        color = _pressure_color(P_hist[i], P_burst)
        if shape == 1:
            D_t = max(D0 - h_hist[i], 1e-4)
            L_t = airbag_geometry_from_h(h_hist[i], D0, L0, shape=shape)["L_t"]
            bag_patch.set_xy(_stadium_outline(L_t, D_t))
            payload_y = D_t
        else:
            # axhspan lives in a blended (axes-x, data-y) transform -> x stays 0..1
            ground_span.set_xy([[0.0, YLIM[0]], [0.0, h_hist[i]],
                                [1.0, h_hist[i]], [1.0, YLIM[0]]])
            ground_line.set_ydata([h_hist[i], h_hist[i]])
            payload_y = D0
        bag_patch.set_facecolor(color)
        payload.set_xy((x_hist[i] - 0.03 * D0, payload_y))

        ax_bag.set_title(f"t = {t_i:.4f} s     h = {h_hist[i]:.3f} m     "
                         f"P = {P_hist[i]/1e3:.1f} kPa", fontsize=11)

        if is_failure and i == last_idx:
            reason = "BURST" if replay["burst"] else "ABRADED"
            fail_txt.set_text(f"*** {reason} at t = {replay['fail_t']:.4f} s ***")
        return []

    fig_c.suptitle("Airbag Simulation (2-DOF oblique) — time cursor + bag compression",
                   fontsize=13)

    anim_c = mpl_animation.FuncAnimation(fig_c, _draw_combined, frames=len(frame_idx),
                                          interval=1000 // ANIM_FPS, blit=False, repeat=False)
    if SAVE_COMBINED_ANIMATION:
        _save_anim(anim_c, "airbag_combined_animation")
    if SHOW_ANIMATIONS:
        plt.show()
    plt.close(fig_c)
