import matplotlib.pyplot as plt
import numpy as np
from airbag_dynamics import system_dynamics_step, horizontal_dynamics_step
from airbag_geometry import airbag_geometry_from_h
from airbag_thermo import thermo_step
import matplotlib.ticker as ticker


def simulate_airbag(
        # geometry / structural
        D0, L0, d_or, d_fabric, P0, shape,
        # impact scenario
        M_payload, u0, ux0, mu,
        # fabric material constants
        sigma_fabric, rho_fabric, c_abrasion,
        # gas / environment
        T0, R_gas, gamma, P_amb, g,
        # control
        dt=1e-3, t_max=0.5, make_plots=False, verbose=False):
    
    # AIRBAG MASS calculation
    A_or = (np.pi / 4.0) * d_or**2
    R0 = D0 / 2.0
    k_shape = 1.0 if shape == 1 else 0.5

    # thickness drives pressure-bursting and abrasion-bursting criteria:
    T_fabric = sigma_fabric * d_fabric              # strength per width -> burst
    P_burst  = P_amb + T_fabric / (k_shape * R0)
    W_abrasion_max = c_abrasion * d_fabric          # the friction/abrasion link

    # airbag mass on the fixed (undeformed) fabric area
    if shape == 1:                                  # cylinder: side + 2 caps
        S_surface = np.pi * D0 * L0 + 0.5 * np.pi * D0**2
    else:                                           # sphere
        S_surface = np.pi * D0**2
    m_bag = rho_fabric * S_surface * d_fabric


    # ---- initial conditions ----
    t = 0.0
    h = 0.0
    x = 0.0
    u = u0
    u_x = ux0
    D_abrasion = 0.0
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
    t_hist, V_hist, P_hist, m_hist, a_hist, u_hist, h_hist, x_hist, ux_hist, D_hist = ([] for _ in range(10))

    # Store initial point
    t_hist.append(t)
    V_hist.append(V_prev)
    P_hist.append(P_bag)
    m_hist.append(m_gas)
    a_hist.append(np.nan)  # unknown at t=0 until first dynamics update
    u_hist.append(u)
    h_hist.append(h)
    x_hist.append(x)
    ux_hist.append(u_x)
    D_hist.append(D_abrasion)



    while t < t_max:

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
        # 1b) Horizontal dynamics: Coulomb friction at the contact patch
        # =========================
        horiz = horizontal_dynamics_step(
            x_prev=x,
            ux_prev=u_x,
            P_bag=P_bag,
            A_t=A_prev,
            mu=mu,
            M=M_payload,
            P_atm=P_amb,
            dt=dt,
        )
        a_x      = horiz["a_x"]
        u_x_new  = horiz["ux_t"]
        x_new    = horiz["x_t"]
        D_abrasion += horiz["dW_fric"]


        # =========================
        # 2) Airbag deformation assumption (geometry)
        #    compute new footprint area and new volume from updated displacement
        # =========================
        geom = airbag_geometry_from_h(h_new, D0, L0, shape=shape)
        A_new = geom["A_t"]     # will be fed to loop in the next step to recalculate acceleration, as A_prev
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
        # 4) Termination check 
        # =========================
        if verbose:
            print(f"t={t:.4f}, h={h:.3f}, V={V_new:.4e}, P={P_new:.1f}, A={A_new:.3f}, a={a:.2f}, m={m_new:.3f}")
        
        t_next = t + dt

        # Save histories at the end of the step
        t_hist.append(t_next)
        V_hist.append(V_new)
        P_hist.append(P_new)
        m_hist.append(m_new)
        a_hist.append(a)
        u_hist.append(u_new)
        h_hist.append(h_new)
        x_hist.append(x_new)
        ux_hist.append(u_x_new)
        D_hist.append(D_abrasion)

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

        if D_abrasion > W_abrasion_max:
            abraded = True
            fail_t = t_next
            if verbose:
                print("\n*** AIRBAG ABRASION FAILURE ***")
                print(f"    Cumulative friction energy {D_abrasion:.0f} J exceeded "
                    f"W_abrasion_max = {W_abrasion_max:.0f} J at t = {t_next:.4f} s.")
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


        if m_new <= 0:
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
        x   = x_new
        u_x = u_x_new




    # Convert lists to arrays
    t_hist = np.array(t_hist)
    V_hist = np.array(V_hist)
    P_hist = np.array(P_hist)
    m_hist = np.array(m_hist)
    a_hist = np.array(a_hist)
    u_hist = np.array(u_hist)
    h_hist = np.array(h_hist)
    x_hist  = np.array(x_hist)
    ux_hist = np.array(ux_hist)
    D_hist  = np.array(D_hist)


    a_peak = float(np.nanmax(np.abs(a_hist)))   # nanmax skips the t=0 NaN
    survived = not (burst or abraded)
    P_peak     = float(np.nanmax(P_hist))
    W_abrasion = float(D_hist[-1])              # cumulative friction energy at end / failure

    if make_plots:
        fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

        plots = [
            (V_hist,  "V [m^3]",        "Airbag Volume"),
            (P_hist,  "P_bag [Pa]",     "Internal Pressure"),
            (m_hist,  "m [kg]",         "Gas Mass"),
            (a_hist,  "a [m/s^2]",      "Payload Acceleration"),
            (u_hist,  "u [m/s]",        "Payload Speed"),
            (h_hist,  "h [m]",          "Payload Displacement"),
            (x_hist,  "x [m]",          "Horizontal Displacement"),
            (ux_hist, "u_x [m/s]",      "Horizontal Velocity"),
            (D_hist,  "D_abrasion [J]", "Cumulative Abrasion Energy"),
        ]

        for ax, (data, ylabel, title) in zip(axs.flat, plots):
            ax.plot(t_hist, data)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True)

        # pressure gets its scientific-notation treatment, same as before
        axs.flat[1].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        axs.flat[1].yaxis.get_offset_text().set_fontsize(9)

        # abrasion limit line stays on its own subplot
        axs.flat[8].axhline(W_abrasion_max, color='r', ls='--', lw=1, label='Abrasion limit')
        axs.flat[8].legend(fontsize=8)

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
        "abraded": abraded, 
        "rebounded": rebounded,
        "u_rebound": u_rebound,
        "bottomed": bottomed,
        "u_residual": u_residual,
        "fail_t": fail_t,
        "P_burst": P_burst, "W_abrasion_max": W_abrasion_max,
        "P_peak": P_peak, "W_abrasion": W_abrasion,
    }

    

