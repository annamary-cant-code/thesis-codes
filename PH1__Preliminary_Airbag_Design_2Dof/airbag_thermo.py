from airbag_venting import delta_m_vented

def thermo_step(m_prev, V_t, gamma, P_amb, dt, A_or,
               P_prev, rho_prev, T_prev,
               R_gas,
               max_inner_iters=50, verbose=False):
    """Adiabatic compression to V_t, then venting through A_or in max_inner_iters sub-steps."""

    V_eff = max(V_t, 1e-12)
    rho_prev_eff = max(rho_prev, 1e-12)

    m = float(m_prev)
    dm_total = 0.0

    # adiabatic compression, before venting
    rho_t = m / V_eff

    P = P_prev * (rho_t / rho_prev_eff) ** gamma
    T = T_prev * (rho_t / rho_prev_eff) ** (gamma - 1.0)

    dt_inner = dt / max_inner_iters

    for _ in range(max_inner_iters):
        if P <= P_amb:
            break

        dm = delta_m_vented(
            P_bag=P,
            P_amb=P_amb,
            T_bag=T,
            A_or=A_or,
            R_gas=R_gas,
            gamma=gamma,
            dt=dt_inner
        )

        dm = min(dm, m)
        m_new = m - dm
        dm_total += dm

        rho_new = m_new / V_eff
        P_new = m_new * R_gas * T / V_eff
        T_new = T * (rho_new / rho_t) ** (gamma - 1.0)

        m = m_new
        rho_t = rho_new
        P = P_new
        T = T_new

        if m <= 0.0:
            m = 0.0
            rho_t = 0.0
            P = 0.0
            if verbose:
                print("all mass has vented")
            break

    return {
        "rho_t": rho_t,      # [kg/m^3]
        "P_bag": P,          # [Pa]
        "T_t": T,            # [K]
        "m_t": m,            # [kg]
        "dm": dm_total       # vented this step [kg]
    }
