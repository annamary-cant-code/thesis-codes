import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from airbag_dynamics import system_dynamics_step
from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step



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
        dt=1e-4, t_max=1, make_plots=False, verbose=False, a_allow=500):
    
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


    # ---- initial conditions ----
    t = 0.0
    h = 0.0
    u = u0
    burst = False
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
    t_hist, V_hist, P_hist, m_hist, a_hist, u_hist, h_hist, x_hist, ux_hist, D_hist = ([] for _ in range(10))

    # Store initial point
    t_hist.append(t)
    V_hist.append(V_prev)
    P_hist.append(P_bag)
    m_hist.append(m_gas)
    a_hist.append(np.nan)  # unknown at t=0 until first dynamics update
    u_hist.append(u)
    h_hist.append(h)



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
        #    uses (P_bag_prev, A_prev) to compute a, then updates u,h
        # =========================
        a, u_new, h_new = system_dynamics_step(
            h_prev=h,
            u_prev=u,
            P_bag=P_bag,     # P0 in the first step
            A_t=A_prev,      # 0 in the first step (calculated)
            M=M_payload,
            g=g,
            P_atm=P_amb,
            dt=dt
        )


        # =========================
        # 2) Airbag deformation assumption (geometry)
        #    compute new footprint area and new volume from updated displacement
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
            print(f"t={t:.7f}, h={h:.4f}, V={V_new:.4e}, P={P_new:.1f}, A={A_new:.5f}, a={a:.3f}, m={m_new:.4f}")
        
        t_next = t + dt

        # Save histories at the end of the step
        t_hist.append(t_next)
        V_hist.append(V_new)
        P_hist.append(P_new)
        m_hist.append(m_new)
        a_hist.append(a)
        u_hist.append(u_new)
        h_hist.append(h_new)

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
            u_residual = u_new         # velocity AT the moment of hitting full stroke
            fail_t = t_next
            if verbose:
                print(f"\n*** FULL STROKE (h reached D0={D0:.3f} m) at t={t_next:.4f} s, "
                    f"u_residual={u_residual:.3f} m/s ***\n")
            break       

        if h_new < 0.0:
            rebounded = True
            u_rebound = u_new
            fail_t = t_next
            if verbose:
                print("\n*** AIRBAG REBOUND ***")
                print(f"    h = {h_new:.4f} m < 0 at t = {t_next:.4f} s — design REBOUNDED.")
                print("    Simulation stopped — design point FAILED.\n")
            break



        # =========================
        # 6) Advance "previous" states for next loop
        # =========================
        t = t_next
        h = h_new           # calculated from dynamic deformation at the beginning of the loop step
        u = u_new           # calculated from dynamic deformation at the beginning of the loop step

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


    a_peak = float(np.nanmax(np.abs(a_hist)))   # nanmax skips the t=0 NaN
    survived = not (burst or rebounded)
    P_peak     = float(np.nanmax(P_hist))

    if make_plots:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

        plots = [
            (h_hist,  "h [m]",          "Payload Displacement"),
            (u_hist,  "u [m/s]",        "Payload Velocity"),
            (a_hist,  "a [m/s^2]",      "Payload Acceleration"),
            (P_hist,  "P_bag [Pa]",     "Internal Pressure"),
            (m_hist,  "m [kg]",         "Gas Mass"),
            (V_hist,  "V [m^3]",        "Airbag Volume"),
        ]

        for ax, (data, ylabel, title) in zip(axs.flat, plots):
            ax.plot(t_hist, data)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True)
        
        # stroke limit on the displacement plot
        axs.flat[0].axhline(D0, color='r', ls='--', lw=1, label=f'D0 = {D0:.3f} m')
        axs.flat[0].legend(fontsize=8)

        # allowable acceleration on the acceleration plot (both directions, since a_peak uses abs)
        A_ALLOW_PLOT = a_allow
        axs.flat[2].axhline(-A_ALLOW_PLOT, color='r', ls='--', lw=1)
        axs.flat[2].legend(fontsize=8)

        axs.flat[3].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        axs.flat[3].yaxis.get_offset_text().set_fontsize(9)

        for ax in axs[-1, :]:
            ax.set_xlabel("Time [s]")

        fig.suptitle("Airbag Simulation", fontsize=13)
        fig.tight_layout()
        plt.show()
        

    return {
        "m_total": m_total,        # objective = fabric + initial gas
        "m_bag": m_bag,            # fabric only
        "m_gas0": m_gas0,          # initial gas charge
        "a_peak": a_peak,          # constraint (vs allowable acceleration)
        "survived": survived,
        "burst": burst,
        "rebounded": rebounded,
        "u_rebound": u_rebound,
        "bottomed": bottomed,
        "u_residual": u_residual,
        "fail_t": fail_t,
        "P_burst": P_burst,
        "P_peak": P_peak,
    }

    

