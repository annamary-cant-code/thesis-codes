from airbag_venting import delta_m_vented

def thermo_step(m_prev, V_t, gamma, P_amb, dt, A_or,
               P_prev, rho_prev, T_prev,
               R_gas,
               max_inner_iters=50, verbose=False):
    """
    Thermodynamics with inner venting loop driven by ADIABATIC pressure-density relation.

    Outer-step inputs (from previous outer loop):
      - P_prev  : pressure at previous outer loop step [Pa]
      - rho_prev: density at previous outer loop step   [kg/m^3]

    Current outer-step geometry (post-dynamics / post-compression):
      - V_t : current volume after compression [m^3]

    Inner loop:
      1) rho_t = m / V_t
      2) P = P_prev * (rho_t / rho_prev)^gamma
      3) if P > P_open: vent -> m := m - Δm, repeat
         else: stop

    Notes:
      - T_prev is used for venting mass flow (orifice) evaluation.
      - R_gas is only used inside delta_m_vented / orifice flow computation (via sqrt(1/(R T)) terms).
    """


    #print(f"m_prev={m_prev:.3e}, V_t={V_t:.3e}, rho_prev={rho_prev:.3e}, P_prev={P_prev:.3e}, gamma={gamma}")
    # numerical safety
    V_eff = max(V_t, 1e-12)                 # post-compression
    rho_prev_eff = max(rho_prev, 1e-12)     # pre-compression

    # Initialise state for inner loop
    m = float(m_prev)                       # pre-compression
    dm_total = 0.0

    # Initial post-compression density (no venting yet)
    rho_t = m / V_eff                       # post-compression

    P = P_prev * (rho_t / rho_prev_eff) ** gamma
    T = T_prev * (rho_t / rho_prev_eff) ** (gamma - 1.0)

    # diagnostics
    # if not hasattr(thermo_step, "_call_count"):
    #     thermo_step._call_count = 0
    # if thermo_step._call_count < 300:
    #     print(f"[CALL {thermo_step._call_count}] rho_prev={rho_prev_eff:.6e}, "
    #           f"rho_t={rho_t:.6e}, ratio={rho_t/rho_prev_eff:.6f}, "
    #           f"P_computed={P:.1f}")
    # thermo_step._call_count += 1

    # sub-iteration for loop
    dt_inner = dt / max_inner_iters

    # Inner venting loop
    for _ in range(max_inner_iters):
        if P <= P_amb:
            break

        # Vent mass during this time step based on current internal pressure
        dm = delta_m_vented(
            P_bag=P,
            P_amb=P_amb,
            T_bag=T,
            A_or=A_or,
            R_gas=R_gas,
            gamma=gamma,
            dt=dt_inner
        )

        # Prevent unphysical mass removal
        dm = min(dm, m)
        m_new = m - dm
        dm_total += dm

        # Recompute density and pressure after venting. Temperature stays the same during venting
        rho_new = m_new / V_eff
        P_new = m_new * R_gas * T / V_eff
        T_new = T * (rho_new / rho_t) ** (gamma - 1.0)
        
        
        # Update for next iteration
        m = m_new
        rho_t = rho_new
        P = P_new
        T = T_new

        # If all gas has vented, stop
        if m <= 0.0:
            m = 0.0
            rho_t = 0.0
            P = 0.0
            if verbose: 
              print("all mass has vented")
            break

    return {
        "rho_t": rho_t,      # density after compression + all venting [kg/m^3]
        "P_bag": P,          # final pressure after inner loop [Pa]
        "T_t": T,            # final temperature after inner loop [K]
        "m_t": m,            # mass after venting [kg]
        "dm": dm_total       # total mass vented during this outer step [kg]
    }
