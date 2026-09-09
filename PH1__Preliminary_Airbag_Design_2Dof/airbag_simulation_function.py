import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step

# NOTE: system_dynamics_step (airbag_dynamics.py) is scalar/1-DOF and cannot express
# the oblique force resolution below (F_n along the fixed n-axis, gravity in y only),
# so the mechanical update is inlined here instead. horizontal_dynamics_step (friction)
# is intentionally not called — see D_abrasion note below.



def simulate_airbag(
        # geometry / structural
        D0, L0, d_or, d_fabric, P0, shape,
        # impact scenario
        M_payload, u0,
        # fabric material constants
        sigma_fabric, rho_fabric,
        # gas / environment
        T0, R_gas, gamma, P_amb, g,
        # control
        dt=1e-4, t_max=1, make_plots=False, verbose=False, a_allow=500,
        # oblique-impact (crude 2-DOF) — 0.0 keeps this a pure vertical (1-DOF) run
        ux0=0.0):

    # AIRBAG MASS calculation
    A_or = (np.pi / 4.0) * d_or**2
    R0 = D0 / 2.0
    k_shape = 1.0 if shape == 1 else 0.5

    # thickness drives pressure-bursting and abrasion-bursting criteria:
    T_fabric = sigma_fabric * d_fabric              # strength per width -> burst
    P_burst  = P_amb + T_fabric / (k_shape * R0)

    # airbag mass on the fixed (undeformed) fabric area
    if shape == 1:                                  # cylinder: side + 2 caps
        S_surface = np.pi * D0 * L0 + 0.5 * np.pi * D0**2
    else:                                           # sphere
        S_surface = np.pi * D0**2
    m_bag = rho_fabric * S_surface * d_fabric


    # ---- fixed oblique-impact axis n, aligned with the initial resultant velocity ----
    # computed ONCE, outside the loop: recomputing it every step would make the
    # tangential velocity identically zero and degenerate the model back to 1-DOF.
    theta = np.arctan2(ux0, u0)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)


    # ---- initial conditions ----
    t = 0.0
    h = 0.0             # crush along n -> feeds geometry/pressure
    y_pen = 0.0          # vertical penetration -> governs ground contact/separation
    x = 0.0              # horizontal global position
    u_x = ux0
    u_y = u0
    D_abrasion = 0.0     # inert in this 2-DOF model (no friction source); kept for API compatibility
    burst = False
    abraded = False
    rebounded = False
    u_rebound = 0.0
    bottomed = False
    u_residual = 0.0    # residual (undissipated) velocity when full stroke is reached
    fail_t = None
    fail_P = None


    # Initialise geometry at t=0 (needed to get A0, V0)
    geom0 = airbag_geometry_from_h(h, D0, L0, shape=shape)
    A_prev = geom0["A_t"]
    V_prev = geom0["V_t"]

    # Initialise gas state at t=0
    m_gas = P0 * V_prev / (R_gas * T0)
    m_gas0 = m_gas
    m_total = m_bag + m_gas0
    T_gas = T0
    P_bag = P0  # carry pressure explicitly for dynamics
    rho_prev = m_gas / V_prev

    # History storage
    (t_hist, V_hist, P_hist, m_hist, a_hist, u_hist, h_hist, x_hist, ux_hist, D_hist,
     y_pen_hist, un_hist, ax_hist, ares_hist) = ([] for _ in range(14))

    # Store initial point
    u_n0 = u_x * sin_theta + u_y * cos_theta
    t_hist.append(t)
    V_hist.append(V_prev)
    P_hist.append(P_bag)
    m_hist.append(m_gas)
    a_hist.append(np.nan)  # unknown at t=0 until first dynamics update
    u_hist.append(u_y)
    h_hist.append(h)
    x_hist.append(x)
    ux_hist.append(u_x)
    D_hist.append(D_abrasion)
    y_pen_hist.append(y_pen)
    un_hist.append(u_n0)
    ax_hist.append(np.nan)
    ares_hist.append(np.nan)



    while t < t_max:

        DT_FINE   = 1e-5
        DT_COARSE = 2e-4
        A_TRIGGER_fine = 0.03 * a_allow   # enter fine mode
        A_TRIGGER_coarse = 0.4 * a_allow  # exit fine mode (hysteresis)
        a_prev = a_hist[-1] if len(a_hist) > 0 and not np.isnan(a_hist[-1]) else 0.0

        if abs(a_prev) > A_TRIGGER_fine:
            dt = DT_FINE
        elif abs(a_prev) < A_TRIGGER_coarse:
            dt = DT_COARSE
        # else: dt unchanged from previous iteration (hysteresis band)

        #GAS_DEPLETED_FRAC = 0.02   # gas essentially gone
        #PRESSURE_NEAR_AMBIENT = 1.05  # P_bag within 5% of P_amb
        #is_settled = (m_gas <= GAS_DEPLETED_FRAC * m_gas0) or (P_bag <= PRESSURE_NEAR_AMBIENT * P_amb)
        #if abs(a_prev) > A_TRIGGER_fine:
        #    dt = DT_FINE
        #elif is_settled:
        #    dt = DT_COARSE
        ## else: dt unchanged — still mid-event, hold whatever regime we're in


        # =========================
        # 1) System dynamics analysis
        #    Normal force uses (P_bag_prev, A_prev), consistent with the h entering
        #    this step, then is resolved onto the fixed n-axis back to global (x,y).
        #    u_n is re-projected from the (possibly diverging) global u_x,u_y using
        #    the FIXED theta — never recomputed from the current velocity direction.
        # =========================
        F_n = max((P_bag - P_amb) * A_prev, 0.0)   # normal force along n, opposing motion
        F_x = -F_n * sin_theta
        F_y = -F_n * cos_theta

        a_y = F_y / M_payload + g     # gravity acts only in global y (down-positive)
        a_x = F_x / M_payload
        a_res = np.sqrt(a_x**2 + a_y**2)

        u_x_new = u_x + a_x * dt
        u_y_new = u_y + a_y * dt
        u_n_new = u_x_new * sin_theta + u_y_new * cos_theta   # re-projected onto fixed n

        h_new = h + u_n_new * dt
        h_new = min(max(h_new, 0.0), D0)          # bottoming-out clamp, along n only
        y_pen_new = y_pen + u_y_new * dt          # vertical penetration, NOT clamped
        x_new = x + u_x_new * dt


        # =========================
        # 2) Airbag deformation assumption (geometry)
        #    compute new footprint area and new volume from updated crush h
        # =========================
        geom = airbag_geometry_from_h(h_new, D0, L0, shape=shape)
        A_new = geom["A_t"]     # will be fed to loop in the next step to recalculate acceleration, as A_prev
        A_new = max(A_new, 0.0)
        V_new = geom["V_t"]

        # =========================
        # 3) Gas thermodynamic analysis (+ orifice flow inside thermo_step)
        #    compute new rho, T, P, m based on new volume
        # =========================
        thermo = thermo_step(
            m_prev=m_gas,
            V_t=V_new,

            gamma=gamma,
            P_amb=P_amb,

            dt=dt,
            A_or=(np.pi/4) * d_or**2,

            P_prev=P_bag,
            rho_prev=rho_prev,
            T_prev=T_gas,

            R_gas=R_gas,
            verbose=verbose
        )

        rho_t = thermo["rho_t"]
        P_new = max(thermo["P_bag"], P_amb)
        m_new = thermo["m_t"]
        dm = thermo["dm"]
        T_new = thermo["T_t"]

        # =========================
        # 4) History
        # =========================
        if verbose:
            print(f"t={t:.7f}, h={h:.4f}, y_pen={y_pen:.4f}, V={V_new:.4e}, P={P_new:.1f}, "
                  f"A={A_new:.5f}, a_y={a_y:.3f}, a_x={a_x:.3f}, m={m_new:.4f}")

        t_next = t + dt

        # Save histories at the end of the step
        t_hist.append(t_next)
        V_hist.append(V_new)
        P_hist.append(P_new)
        m_hist.append(m_new)
        a_hist.append(a_y)
        u_hist.append(u_y_new)
        h_hist.append(h_new)
        x_hist.append(x_new)
        ux_hist.append(u_x_new)
        D_hist.append(D_abrasion)          # no friction in this model -> stays at 0
        y_pen_hist.append(y_pen_new)
        un_hist.append(u_n_new)
        ax_hist.append(a_x)
        ares_hist.append(a_res)

        # =========================
        # 5) Failure / termination checks — report whichever trips first
        # =========================
        if P_new > P_burst:
            burst = True
            fail_t, fail_P = t_next, P_new
            if verbose:
                print("\n*** AIRBAG BURST (hoop over-pressure) ***")
                print(f"    P = {P_new:.0f} Pa exceeded P_burst = {P_burst:.0f} Pa "
                    f"at t = {t_next:.4f} s.")
                print("    Simulation stopped — design point FAILED.\n")
            break

        if h_new >= D0:
            bottomed = True
            u_residual = u_y_new         # velocity AT the moment of hitting full stroke
            fail_t = t_next
            if verbose:
                print(f"\n*** FULL STROKE (h reached D0={D0:.3f} m) at t={t_next:.4f} s, "
                    f"u_residual={u_residual:.3f} m/s ***\n")
            break

        if y_pen_new <= 0.0:
            rebounded = True
            u_rebound = u_y_new
            fail_t = t_next
            if verbose:
                print("\n*** AIRBAG REBOUND ***")
                print(f"    y_pen = {y_pen_new:.4f} m <= 0 at t = {t_next:.4f} s — design REBOUNDED.")
                print("    Simulation stopped — design point FAILED.\n")
            break



        # =========================
        # 6) Advance "previous" states for next loop
        # =========================
        t = t_next
        h = h_new           # crush along n, feeds geometry/pressure next step
        y_pen = y_pen_new   # vertical penetration, feeds termination check next step
        u_x = u_x_new
        u_y = u_y_new
        x = x_new

        A_prev = A_new      # consequence of dynamic step
        V_prev = V_new      # consequence of dynamic step

        P_bag = P_new       # uptades gas-state variables found outside loop, which are then fed to the loop in the dynamic and thermo steps
        rho_prev = rho_t
        T_gas = T_new
        m_gas = m_new




    # Convert lists to arrays
    t_hist = np.array(t_hist)
    V_hist = np.array(V_hist)
    P_hist = np.array(P_hist)
    m_hist = np.array(m_hist)
    a_hist = np.array(a_hist)
    u_hist = np.array(u_hist)
    h_hist = np.array(h_hist)
    x_hist = np.array(x_hist)
    ux_hist = np.array(ux_hist)
    D_hist = np.array(D_hist)
    y_pen_hist = np.array(y_pen_hist)
    un_hist = np.array(un_hist)
    ax_hist = np.array(ax_hist)
    ares_hist = np.array(ares_hist)


    a_peak = float(np.nanmax(np.abs(a_hist)))              # nanmax skips the t=0 NaN — max ABS VERTICAL accel
    a_peak_resultant = float(np.nanmax(np.abs(ares_hist)))
    survived = not (burst or rebounded or abraded)
    P_peak     = float(np.nanmax(P_hist))

    if make_plots:
        fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

        plots = [
            (h_hist,     "h [m]",          "Crush Along n"),
            (y_pen_hist, "y_pen [m]",      "Vertical Penetration"),
            (u_hist,     "u_y [m/s]",      "Vertical Velocity"),
            (un_hist,    "u_n [m/s]",      "Velocity Along n"),
            (a_hist,     "a_y [m/s^2]",    "Vertical Acceleration"),
            (ares_hist,  "a_res [m/s^2]",  "Resultant Acceleration"),
            (P_hist,     "P_bag [Pa]",     "Internal Pressure"),
            (m_hist,     "m [kg]",         "Gas Mass"),
            (V_hist,     "V [m^3]",        "Airbag Volume"),
        ]

        for ax, (data, ylabel, title) in zip(axs.flat, plots):
            ax.plot(t_hist, data)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True)

        # stroke limit on the crush-along-n plot
        axs.flat[0].axhline(D0, color='r', ls='--', lw=1, label=f'D0 = {D0:.3f} m')
        axs.flat[0].legend(fontsize=8)

        # allowable acceleration on the vertical- and resultant-acceleration plots
        A_ALLOW_PLOT = a_allow
        axs.flat[4].axhline(-A_ALLOW_PLOT, color='r', ls='--', lw=1)
        axs.flat[5].axhline(A_ALLOW_PLOT, color='r', ls='--', lw=1, label=f'a_allow = {a_allow:.0f}')
        axs.flat[5].legend(fontsize=8)

        axs.flat[6].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        axs.flat[6].yaxis.get_offset_text().set_fontsize(9)

        for ax in axs[-1, :]:
            ax.set_xlabel("Time [s]")

        fig.suptitle("Airbag Simulation (2-DOF oblique)", fontsize=13)
        fig.tight_layout()
        plt.show()


    return {
        "m_total": m_total,        # objective = fabric + initial gas
        "m_bag": m_bag,            # fabric only
        "m_gas0": m_gas0,          # initial gas charge
        "a_peak": a_peak,          # constraint (vs allowable acceleration) — max ABS vertical accel
        "a_peak_resultant": a_peak_resultant,  # max ABS resultant (x,y) accel
        "theta": theta,            # fixed impact angle [rad], atan2(ux0, u0)
        "survived": survived,
        "burst": burst,
        "rebounded": rebounded,
        "u_rebound": u_rebound,
        "bottomed": bottomed,
        "u_residual": u_residual,
        "fail_t": fail_t,
        "P_burst": P_burst,
        "P_peak": P_peak,
        "D_abrasion": D_abrasion,  # inert in this 2-DOF model (no friction source); reported for API compatibility
    }



