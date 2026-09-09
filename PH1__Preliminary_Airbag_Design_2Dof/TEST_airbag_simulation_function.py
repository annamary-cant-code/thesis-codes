from airbag_simulation_function import simulate_airbag
import numpy as np
import contextlib
import os

MAKE_ANIMATION = True   # set True to render the 2D impact animation
SAVE_ANIMATION = True   # set True (with MAKE_ANIMATION) to also export an .mp4/.gif

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


if MAKE_ANIMATION:
    # ------------------------------------------------------------------------
    # simulate_airbag() (airbag_simulation_function.py) does NOT return any
    # per-step history — only final/peak scalars (checked: res.keys() above
    # has no "history", no h_hist/x_hist/etc). Since that file must not be
    # modified, the stepping algorithm is replayed here, standalone, using
    # only the same building blocks it already uses (airbag_geometry_from_h,
    # thermo_step) purely to reconstruct h/x/y_pen/P histories for drawing
    # animation frames. This duplicate MUST be kept in sync by hand with
    # airbag_simulation_function.py's loop if that algorithm ever changes.
    #
    # The bag silhouette drawn each frame is a schematic reconstruction from
    # the scalar crush state h (via airbag_geometry_from_h), not a physical
    # deformation mesh — same for the sphere's "buried circle" treatment.
    # ------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    from airbag_geometry import airbag_geometry_from_h
    from airbag_thermo import thermo_step

    def _replay_for_animation(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                               sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                               dt, t_max, a_allow, ux0):
        """Standalone replica of airbag_simulation_function.simulate_airbag's
        loop, kept only long enough to record t/h/x/y_pen/P per step."""
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

        t_hist, h_hist, x_hist, y_pen_hist, P_hist, a_hist = [t], [h], [x], [y_pen], [P_bag], [np.nan]

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

        return dict(t_hist=np.array(t_hist), h_hist=np.array(h_hist), x_hist=np.array(x_hist),
                    y_pen_hist=np.array(y_pen_hist), P_hist=np.array(P_hist),
                    P_burst=P_burst, burst=burst, bottomed=bottomed, rebounded=rebounded,
                    fail_t=fail_t)

    replay = _replay_for_animation(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,
                                    sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g,
                                    dt, t_max, a_allow=500, ux0=ux0)

    # Sanity check: the replay must agree with the real simulate_airbag() call above,
    # since it's meant to reproduce the exact same run for animation purposes only.
    if (replay["burst"], replay["bottomed"], replay["rebounded"]) != (res["burst"], res["bottomed"], res["rebounded"]):
        print("[MAKE_ANIMATION] WARNING: replay outcome does not match simulate_airbag() result "
              "— the standalone replay driver may have drifted out of sync with "
              "airbag_simulation_function.py's algorithm.")

    h_hist, x_hist, y_pen_hist = replay["h_hist"], replay["x_hist"], replay["y_pen_hist"]
    P_hist, P_burst = replay["P_hist"], replay["P_burst"]
    n_steps = len(h_hist)
    last_idx = n_steps - 1

    # Subsample to a manageable number of animation frames (the physics loop
    # can take thousands of adaptive-dt steps; we don't need to draw all of them).
    MAX_FRAMES = 200
    frame_idx = np.linspace(0, last_idx, min(n_steps, MAX_FRAMES)).astype(int)
    frame_idx = np.unique(frame_idx)  # keep monotonic & include last_idx exactly

    failure = replay["burst"]  # (res["abraded"] is always False in this 2-DOF model, checked for completeness)
    is_failure = replay["burst"] or res.get("abraded", False)
    if is_failure:
        frame_idx = np.concatenate([frame_idx, np.full(30, last_idx, dtype=int)])  # hold on last frame

    def _pressure_color(P, P_burst):
        ratio = float(np.clip(P / P_burst, 0.0, 1.0))
        if ratio < 0.5:
            f = ratio / 0.5
            return (f, 1.0, 0.0)      # green -> yellow
        else:
            f = (ratio - 0.5) / 0.5
            return (1.0, 1.0 - f, 0.0)  # yellow -> red

    fig, ax = plt.subplots(figsize=(7, 6))

    x_span = max(np.ptp(x_hist), D0)
    xlim = (-0.7 * x_span - 0.5 * D0, 0.7 * x_span + 0.5 * D0)
    ylim = (-0.05 * D0, 1.15 * D0)

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

    def _draw_frame(frame_number):
        i = int(frame_idx[frame_number])
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(f"t = {replay['t_hist'][i]:.4f} s")

        color = _pressure_color(P_hist[i], P_burst)

        if shape == 1:
            # cylinder: side-view "stadium" cross-section, flattening as h grows
            ax.axhline(0, color='saddlebrown', lw=2, zorder=1)
            D_t = max(D0 - h_hist[i], 1e-4)
            geom = airbag_geometry_from_h(h_hist[i], D0, L0, shape=shape)
            L_t = geom["L_t"]
            ax.add_patch(plt.Polygon(_stadium_outline(L_t, D_t), closed=True,
                                      facecolor=color, edgecolor='k', zorder=2))
            payload_y = D_t
        else:
            # sphere: fixed circle, ground line rises into it as h grows ("buried" look)
            R0 = D0 / 2.0
            ax.add_patch(plt.Circle((0, R0), R0, facecolor=color, edgecolor='k', zorder=1))
            ax.axhspan(ylim[0], h_hist[i], color='saddlebrown', zorder=2)
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
                                        interval=40, blit=False, repeat=False)
    plt.show()

    if SAVE_ANIMATION:
        try:
            anim.save("airbag_impact_animation.mp4", fps=25)
            print("Saved animation to airbag_impact_animation.mp4")
        except Exception as e:
            print(f"[SAVE_ANIMATION] could not save animation "
                  f"(ffmpeg/imagemagick likely not installed): {e}")
