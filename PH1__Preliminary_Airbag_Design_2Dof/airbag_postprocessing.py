import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation

from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step
from differential_evolution_wrapper import A_ALLOW, SF

PLOT_MATRIX_DPI = 150
MAX_FRAMES = 200          # animation frames
ANIM_FPS = 25
ANIM_DPI = 90
CURVE_MAX_PTS = 4000      # curve decimation in the combined animation (display only)
N_ROWS, N_COLS = 4, 4


def replay_histories(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                     sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                     dt, t_max, a_allow, ux0, phi_vent=0.0):
    """Re-runs simulate_airbag's loop to record per-step histories (simulate_airbag
    only returns scalars). Keep in sync with airbag_simulation_function.py by hand."""
    R0 = D0 / 2.0
    k_shape = 1.0 if shape == 1 else 0.5
    T_fabric = sigma_fabric * d_fabric
    P_burst = P_amb + T_fabric / (k_shape * R0)
    A_or = (np.pi / 4) * d_or ** 2
    S_surface = (np.pi * D0 * L0 + 0.5 * np.pi * D0 ** 2) if shape == 1 else np.pi * D0 ** 2

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

    t_hist, h_hist, x_hist = [t], [h], [x]
    y_pen_hist, P_hist, a_hist = [y_pen], [P_bag], [np.nan]
    u_hist, un_hist = [u_y], [u_x * sin_theta + u_y * cos_theta]
    ares_hist, m_hist, V_hist = [np.nan], [m_gas], [V_prev]
    ux_hist, ax_hist, T_hist = [u_x], [np.nan], [T_gas]
    Fx_hist, Fy_hist, Fn_hist = [np.nan], [np.nan], [np.nan]

    while t < t_max:
        DT_FINE, DT_COARSE = 1e-5, 2e-4
        A_TRIGGER_fine, A_TRIGGER_coarse = 0.03 * a_allow, 0.01 * a_allow
        load_prev = Fn_hist[-1] / M_payload if not np.isnan(Fn_hist[-1]) else 0.0
        if load_prev > A_TRIGGER_fine:
            dt = DT_FINE
        elif load_prev < A_TRIGGER_coarse:
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
                             A_or=A_or + phi_vent * max(S_surface - A_new, 0.0),
                             P_prev=P_bag, rho_prev=rho_prev,
                             T_prev=T_gas, R_gas=R_gas, verbose=False)
        P_new = max(thermo["P_bag"], P_amb)

        t_next = t + dt
        t_hist.append(t_next); h_hist.append(h_new); x_hist.append(x_new)
        y_pen_hist.append(y_pen_new); P_hist.append(P_new); a_hist.append(a_y)
        u_hist.append(u_y_new); un_hist.append(u_n_new); ares_hist.append(a_res)
        m_hist.append(thermo["m_t"]); V_hist.append(V_new)
        ux_hist.append(u_x_new); ax_hist.append(a_x); T_hist.append(thermo["T_t"])
        Fx_hist.append(F_x); Fy_hist.append(M_payload * a_y); Fn_hist.append(F_n)

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
                ux_hist=arr(ux_hist), ax_hist=arr(ax_hist), T_hist=arr(T_hist),
                Fx_hist=arr(Fx_hist), Fy_hist=arr(Fy_hist), Fn_hist=arr(Fn_hist),
                P_burst=P_burst, burst=burst, bottomed=bottomed, rebounded=rebounded,
                fail_t=fail_t)


def _pressure_color(P, P_burst):
    ratio = float(np.clip(P / P_burst, 0.0, 1.0))
    if ratio < 0.5:
        f = ratio / 0.5
        return (f, 1.0, 0.0)          # green -> yellow
    else:
        f = (ratio - 0.5) / 0.5
        return (1.0, 1.0 - f, 0.0)    # yellow -> red


def _stadium_outline(L_t, D_t, n_arc=60):
    """Closed outline of the cylinder's stadium cross-section, as one polygon."""
    r = D_t / 2.0
    th_right = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
    th_left = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)
    right = np.column_stack((L_t / 2 + r * np.cos(th_right), r + r * np.sin(th_right)))
    left = np.column_stack((-L_t / 2 + r * np.cos(th_left), r + r * np.sin(th_left)))
    return np.vstack((right, left))


def _save_anim(anim, out_dir, basename, fps=ANIM_FPS, dpi=ANIM_DPI):
    """Writes .mp4 and .gif. Must run before plt.show(), which exhausts the animation."""
    writers = (
        ("mp4", lambda: mpl_animation.FFMpegWriter(fps=fps)),
        ("gif", lambda: mpl_animation.PillowWriter(fps=fps)),
    )
    for ext, make_writer in writers:
        path = os.path.join(out_dir, f"{basename}.{ext}")
        try:
            anim.save(path, writer=make_writer(), dpi=dpi)
            print(f"Saved animation -> {path}")
        except Exception as e:
            print(f"[save] {ext} failed ({type(e).__name__}: {e})")


def _plot_specs(replay, M_payload):
    """(data, ylabel, title) for the 4x4 matrix: rows x / y / n / gas."""
    return [
        (replay["x_hist"],     "x [m]",           "Horizontal Displacement"),
        (replay["ux_hist"],    "u_x [m/s]",       "Horizontal Velocity"),
        (replay["ax_hist"],    "a_x [m/s^2]",     "Horizontal Acceleration"),
        (replay["Fx_hist"],    "F_x [N]",         "Horizontal Force"),

        (replay["y_pen_hist"], "y_pen [m]",       "Vertical Penetration"),
        (replay["u_hist"],     "u_y [m/s]",       "Vertical Velocity"),
        (replay["a_hist"],     "a_y [m/s^2]",     "Vertical Acceleration"),
        (replay["Fy_hist"],    "F_y [N]",         "Net Vertical Force (incl. g)"),

        (replay["h_hist"],     "h [m]",           "Displacement Along n"),
        (replay["un_hist"],    "u_n [m/s]",       "Velocity Along n"),
        (replay["Fn_hist"] / M_payload, "F_n/M [m/s^2]", "Bag Load Along n (F_n/M)"),
        (replay["Fn_hist"],    "F_n [N]",         "Pressure Force Along n"),

        (replay["m_hist"],     "m [kg]",          "Gas Mass"),
        (replay["V_hist"],     "V [m^3]",         "Airbag Volume"),
        (replay["P_hist"],     "P_bag [Pa]",      "Internal Pressure"),
        (replay["T_hist"],     "T [K]",           "Gas Temperature"),
    ]


def _decorate_matrix(axes, specs, D0, a_lim, label_fs=9, title_fs=10):
    by_title = {}
    for ax, (_, ylabel, title) in zip(axes, specs):
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.set_title(title, fontsize=title_fs)
        ax.grid(True)
        ax.tick_params(labelsize=label_fs - 1)
        by_title[title] = ax
    ax_h = by_title["Displacement Along n"]
    ax_h.axhline(D0, color='tab:green', ls='--', lw=1.2, label=f'D0 = {D0:.3f} m')
    ax_h.legend(fontsize=label_fs - 1)

    # F_n/M is the constrained quantity; a_x and a_y get the line on their peak's side
    data_by_title = {title: data for data, _, title in specs}
    for title in ("Horizontal Acceleration", "Vertical Acceleration",
                  "Bag Load Along n (F_n/M)"):
        data = data_by_title[title]
        if title.startswith("Bag Load"):
            lim, sign = a_lim, ""
        elif data[np.nanargmax(np.abs(data))] <= 0:
            lim, sign = -a_lim, "-"
        else:
            lim, sign = a_lim, ""
        by_title[title].axhline(lim, color='r', ls='--', lw=1,
                                label=f'{sign}A_LIM = {lim:.1f} m/s^2  (= {sign or "+"}{A_ALLOW:.0f}/{SF})')
        by_title[title].legend(fontsize=label_fs - 1)
    ax_P = by_title["Internal Pressure"]
    ax_P.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    ax_P.yaxis.get_offset_text().set_fontsize(label_fs - 1)
    for ax in axes[-N_COLS:]:
        ax.set_xlabel("Time [s]", fontsize=label_fs)


def run_postprocessing(inputs, res, feasible, show_popups=True, save_plot_matrix=True,
                       plot_matrix_formats=("png", "pdf"), make_animation=True,
                       save_animation=False, make_combined_animation=False,
                       save_combined_animation=False, output_dir=None):
    """Plot matrix and animations for one simulate_airbag run.
    inputs: the kwargs passed to simulate_airbag (without make_plots/verbose)."""
    out_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    D0, L0, shape = inputs["D0"], inputs["L0"], inputs["shape"]
    a_lim = inputs["a_allow"]

    replay = replay_histories(**inputs)
    if (replay["burst"], replay["bottomed"], replay["rebounded"]) != (res["burst"], res["bottomed"], res["rebounded"]):
        print("[post-processing] WARNING: replay outcome differs from simulate_airbag() "
              "-- replay_histories() is out of sync with airbag_simulation_function.py")

    t_hist = replay["t_hist"]
    h_hist, x_hist = replay["h_hist"], replay["x_hist"]
    P_hist, P_burst = replay["P_hist"], replay["P_burst"]
    n_steps = len(h_hist)
    last_idx = n_steps - 1

    is_failure = replay["burst"] or replay["rebounded"] or not feasible
    if is_failure:
        if replay["burst"]:
            fail_reason = "BURST"
        elif replay["rebounded"]:
            fail_reason = "REBOUND"
        elif not res["bottomed"]:
            fail_reason = "TIMEOUT (never bottomed)"
        else:
            fail_reason = "INFEASIBLE (a_peak_n or u_residual_n over limit)"
        fail_overlay_t = replay["fail_t"] if replay["fail_t"] is not None else replay["t_hist"][-1]

    frame_idx = np.linspace(0, last_idx, min(n_steps, MAX_FRAMES)).astype(int)
    frame_idx = np.unique(frame_idx)
    if is_failure:
        frame_idx = np.concatenate([frame_idx, np.full(30, last_idx, dtype=int)])  # hold last frame

    # wide enough for the payload's travel and the fully crushed bag (aspect='equal')
    x_span = max(np.ptp(x_hist), D0)
    x_pad = 0.35 * D0
    XLIM = (-0.5 * np.pi * D0 / 2 - x_pad, x_span + x_pad)
    YLIM = (-0.05 * D0, 1.15 * D0)

    specs = _plot_specs(replay, inputs["M_payload"])

    # ---- static 4x4 plot matrix ----
    if save_plot_matrix or show_popups:
        fig_m, axs_m = plt.subplots(N_ROWS, N_COLS, figsize=(16, 9), sharex=True,
                                    layout="constrained")
        for ax, (data, _, _) in zip(axs_m.flat, specs):
            ax.plot(t_hist, data)
        _decorate_matrix(list(axs_m.flat), specs, D0, a_lim)
        fig_m.suptitle("Airbag Simulation (2-DOF oblique)", fontsize=13)
        if save_plot_matrix:
            for ext in plot_matrix_formats:
                out = os.path.join(out_dir, f"airbag_plot_matrix.{ext}")
                fig_m.savefig(out, dpi=PLOT_MATRIX_DPI, bbox_inches="tight")
                print(f"Saved plot matrix -> {out}")
        if not show_popups:
            plt.close(fig_m)

    # ---- bag-only animation ----
    if make_animation:
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
                ax.axhline(0, color='saddlebrown', lw=2, zorder=1)
                D_t = max(D0 - h_hist[i], 1e-4)
                L_t = airbag_geometry_from_h(h_hist[i], D0, L0, shape=shape)["L_t"]
                ax.add_patch(plt.Polygon(_stadium_outline(L_t, D_t), closed=True,
                                         facecolor=color, edgecolor='k', zorder=2))
                payload_y = D_t
            else:
                # sphere: fixed circle, ground rises into it as h grows
                R0 = D0 / 2.0
                ax.add_patch(plt.Circle((0, R0), R0, facecolor=color, edgecolor='k', zorder=1))
                ax.axhspan(YLIM[0], h_hist[i], color='saddlebrown', zorder=2)
                ax.axhline(h_hist[i], color='k', lw=1, zorder=3)
                payload_y = D0

            ax.add_patch(plt.Rectangle((x_hist[i] - 0.03 * D0, payload_y),
                                       0.06 * D0, 0.06 * D0, facecolor='navy', zorder=4))

            if is_failure and i == last_idx:
                ax.text(0.5, 0.92, f"*** {fail_reason} at t = {fail_overlay_t:.4f} s ***",
                        transform=ax.transAxes, ha='center', color='red', fontsize=11,
                        fontweight='bold')
            return []

        anim = mpl_animation.FuncAnimation(fig, _draw_frame, frames=len(frame_idx),
                                           interval=1000 // ANIM_FPS, blit=False, repeat=False)
        if save_animation:
            _save_anim(anim, out_dir, "airbag_impact_animation")
        if not show_popups:
            plt.close(fig)

    # ---- combined animation: plot matrix with time cursor + bag ----
    if make_combined_animation:
        fig_c = plt.figure(figsize=(22, 9.5))
        gs = fig_c.add_gridspec(N_ROWS, N_COLS + 1, width_ratios=[1] * N_COLS + [3.4],
                                wspace=0.30, hspace=0.55,
                                left=0.035, right=0.99, top=0.90, bottom=0.07)

        p_axes = []
        for r in range(N_ROWS):
            for c in range(N_COLS):
                p_axes.append(fig_c.add_subplot(gs[r, c], sharex=p_axes[0] if p_axes else None))

        dstep = max(1, len(t_hist) // CURVE_MAX_PTS)
        cursors, markers = [], []
        for ax, (data, _, _) in zip(p_axes, specs):
            ax.plot(t_hist[::dstep], data[::dstep], lw=1.2, color='C0')
            cursors.append(ax.axvline(t_hist[0], color='crimson', lw=1.2, zorder=5))
            markers.append(ax.plot([t_hist[0]], [data[0]], 'o', color='crimson',
                                   ms=5, zorder=6)[0])
        _decorate_matrix(p_axes, specs, D0, a_lim, label_fs=8, title_fs=9)

        ax_bag = fig_c.add_subplot(gs[:, N_COLS])
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

            for cur, mk, (data, _, _) in zip(cursors, markers, specs):
                cur.set_xdata([t_i, t_i])
                mk.set_data([t_i], [data[i]])

            color = _pressure_color(P_hist[i], P_burst)
            if shape == 1:
                D_t = max(D0 - h_hist[i], 1e-4)
                L_t = airbag_geometry_from_h(h_hist[i], D0, L0, shape=shape)["L_t"]
                bag_patch.set_xy(_stadium_outline(L_t, D_t))
                payload_y = D_t
            else:
                # axhspan uses axes coordinates in x
                ground_span.set_xy([[0.0, YLIM[0]], [0.0, h_hist[i]],
                                    [1.0, h_hist[i]], [1.0, YLIM[0]]])
                ground_line.set_ydata([h_hist[i], h_hist[i]])
                payload_y = D0
            bag_patch.set_facecolor(color)
            payload.set_xy((x_hist[i] - 0.03 * D0, payload_y))

            ax_bag.set_title(f"t = {t_i:.4f} s     h = {h_hist[i]:.3f} m     "
                             f"P = {P_hist[i]/1e3:.1f} kPa", fontsize=11)

            if is_failure and i == last_idx:
                fail_txt.set_text(f"*** {fail_reason} at t = {fail_overlay_t:.4f} s ***")
            return []

        fig_c.suptitle("Airbag Simulation (2-DOF oblique) — time cursor + bag compression",
                       fontsize=13)

        anim_c = mpl_animation.FuncAnimation(fig_c, _draw_combined, frames=len(frame_idx),
                                             interval=1000 // ANIM_FPS, blit=False, repeat=False)
        if save_combined_animation:
            _save_anim(anim_c, out_dir, "airbag_combined_animation")
        if not show_popups:
            plt.close(fig_c)

    if show_popups:
        plt.show()
