"""Compiled (numba) twin of simulate_airbag, for the optimiser's search phase only.

Same physics and the same explicit scheme as airbag_simulation_function.py, airbag_thermo.py,
airbag_venting.py and airbag_geometry.py, step for step. Differences:
  * no history arrays: only the running peaks and final values the objective needs are kept;
  * no plots / verbose output;
  * venting sub-iterations depend on the time step: max_inner_iters_fine during the fine
    (dt = 1e-5 s) phase, max_inner_iters during the coarse (dt = 2e-4 s) phase.
    With fine = coarse it reproduces simulate_airbag(max_inner_iters=...) to round-off.

The original simulate_airbag stays the reference: it is used for the full-fidelity
re-simulation of the winner, for plots and by the other scripts.
If you change the physics there, change it here too.
"""
import math

import numpy as np
from numba import njit

DT_FINE   = 1e-5      # [s] must match simulate_airbag
DT_COARSE = 2e-4      # [s]


@njit(cache=True)
def _geometry(h_t, D0, L0, shape):
    """airbag_geometry_from_h, returning only (A_t, V_t)."""
    if shape == 1:
        D_t_eff = max(D0 - h_t, 1e-12)
        L_t = 0.5 * np.pi * h_t
        A_t = L_t * L0
        V_t = L0 * D_t_eff * ((np.pi / 4) * D_t_eff + L_t)
    else:
        r_c = math.sqrt(max(h_t * (D0 - h_t), 0.0))
        A_t = np.pi * r_c**2
        V_sphere = (np.pi / 6.0) * D0**3
        V_cap = (np.pi / 3.0) * h_t**2 * (1.5 * D0 - h_t)
        V_t = V_sphere - V_cap
    return A_t, V_t


@njit(cache=True)
def _orifice_mdot(P_up, P_down, T_up, A_vent, R_gas, gamma, choke_factor):
    """airbag_venting.orifice_mdot; choke_factor depends only on gamma, so it is passed in."""
    if P_up <= P_down:
        return 0.0
    lmbd = P_down / P_up
    l = min(max(lmbd, 0.0), 1.0)
    C_D = (-3.8399*l**6 + 9.4363*l**5 - 7.2326*l**4 + 1.6972*l**3
           - 0.2908*l**2 - 0.0130*l + 0.8426)
    C_D = max(C_D, 0.0)
    common_term = math.sqrt(1.0 / (R_gas * T_up))
    if lmbd >= 0.528:      # subsonic
        pr = P_up / P_down
        expo = (gamma - 1.0) / gamma
        prx = pr ** expo
        term1 = math.sqrt((2.0 * gamma) / (gamma - 1.0) * prx)
        term2 = math.sqrt(max(prx - 1.0, 0.0))
        mdot = C_D * A_vent * P_down * common_term * term1 * term2
    else:                  # choked
        mdot = C_D * A_vent * P_up * common_term * choke_factor
    return max(mdot, 0.0)


@njit(cache=True)
def _thermo_step(m_prev, V_t, gamma, P_amb, dt, A_vent, P_prev, rho_prev, T_prev, R_gas,
                 n_inner, choke_factor):
    """airbag_thermo.thermo_step: adiabatic compression, then n_inner venting sub-steps."""
    V_eff = max(V_t, 1e-12)
    rho_prev_eff = max(rho_prev, 1e-12)
    m = m_prev
    rho_t = m / V_eff
    P = P_prev * (rho_t / rho_prev_eff) ** gamma
    T = T_prev * (rho_t / rho_prev_eff) ** (gamma - 1.0)
    dt_inner = dt / n_inner
    for _ in range(n_inner):
        if P <= P_amb:
            break
        dm = _orifice_mdot(P, P_amb, T, A_vent, R_gas, gamma, choke_factor) * dt_inner
        dm = min(dm, m)
        m_new = m - dm
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
            break
    return rho_t, P, T, m


@njit(cache=True)
def _run(D0, L0, A_or, P0, shape, M_payload, uy0, ux0, T0, R_gas, gamma, P_amb, g,
         P_burst, S_surface, phi_vent, a_allow, dt, t_max, n_inner_coarse, n_inner_fine):
    """Time loop of simulate_airbag. Returns peaks, final state and the stopping event."""
    theta = math.atan2(ux0, uy0)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    choke_factor = math.sqrt(gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))

    t = 0.0
    h = 0.0
    y_pen = 0.0
    x = 0.0
    ux = ux0
    uy = uy0
    burst = False
    rebounded = False
    bottomed = False
    u_rebound_y = 0.0
    u_residual_y = 0.0
    u_residual_n = 0.0
    u_residual_x = 0.0
    t_stop = -1.0                        # -1 -> no stopping event (ran to t_max)

    A_prev, V_prev = _geometry(0.0, D0, L0, shape)
    m_gas = P0 * V_prev / (R_gas * T0)
    m_gas0 = m_gas
    T_gas = T0
    P_bag = P0
    rho_prev = m_gas / V_prev

    # running peaks instead of history arrays (the t = 0 accelerations are nan -> skipped)
    un_last = ux * sin_theta + uy * cos_theta
    x_last = 0.0
    P_peak = P_bag
    h_max = 0.0
    a_peak_y = -np.inf
    a_peak_n = -np.inf
    a_peak_res = -np.inf
    load_prev = 0.0

    A_TRIGGER_fine = 0.03 * a_allow
    A_TRIGGER_coarse = 0.01 * a_allow

    while t < t_max:
        if load_prev > A_TRIGGER_fine:
            dt = DT_FINE
        elif load_prev < A_TRIGGER_coarse:
            dt = DT_COARSE
        n_inner = n_inner_fine if dt < DT_COARSE else n_inner_coarse

        Fn = max((P_bag - P_amb) * A_prev, 0.0)
        Fx = -Fn * sin_theta
        Fy = -Fn * cos_theta
        ay = Fy / M_payload + g
        ax = Fx / M_payload
        ares = math.sqrt(ax**2 + ay**2)
        load_n = Fn / M_payload

        ux_new = ux + ax * dt
        uy_new = uy + ay * dt
        un_new = ux_new * sin_theta + uy_new * cos_theta
        h_new = h + un_new * dt
        h_new = min(max(h_new, 0.0), D0)
        y_pen_new = y_pen + uy_new * dt
        x_new = x + ux_new * dt

        A_new, V_new = _geometry(h_new, D0, L0, shape)
        A_new = max(A_new, 0.0)
        A_vent = A_or + phi_vent * max(S_surface - A_new, 0.0)

        rho_t, P_th, T_new, m_new = _thermo_step(m_gas, V_new, gamma, P_amb, dt, A_vent,
                                                 P_bag, rho_prev, T_gas, R_gas, n_inner, choke_factor)
        P_new = max(P_th, P_amb)
        t_next = t + dt

        # "history": peaks and last values only
        P_peak = max(P_peak, P_new)
        h_max = max(h_max, h_new)
        a_peak_y = max(a_peak_y, abs(ay))
        a_peak_n = max(a_peak_n, load_n)
        a_peak_res = max(a_peak_res, abs(ares))
        un_last = un_new
        x_last = x_new
        load_prev = load_n

        if P_new > P_burst:
            burst = True
            t_stop = t_next
            break
        if h_new >= D0:
            bottomed = True
            u_residual_y = uy_new
            u_residual_n = un_new
            u_residual_x = ux_new
            t_stop = t_next
            break
        if y_pen_new <= 0.0:
            rebounded = True
            u_rebound_y = uy_new
            t_stop = t_next
            break

        t = t_next
        h = h_new
        y_pen = y_pen_new
        ux = ux_new
        uy = uy_new
        x = x_new
        A_prev = A_new
        V_prev = V_new
        P_bag = P_new
        rho_prev = rho_t
        T_gas = T_new
        m_gas = m_new

    return (m_gas0, theta, a_peak_y, a_peak_n, a_peak_res, P_peak, h_max, x_last, un_last,
            burst, bottomed, rebounded, u_rebound_y, u_residual_y, u_residual_n, u_residual_x, t_stop)


def simulate_airbag_fast(
        D0, L0, d_or, P0, shape,
        M_payload, uy0,
        N_ult_Ncm, areal_weight, t_ply, n_ply,
        T0, R_gas, gamma, P_amb, g,
        eta_seam=0.70, sf_burst=1.50,
        dt=1e-4, t_max=1, a_allow=500,
        ux0=0.0, phi_vent=0.0,
        max_inner_iters=50,          # venting sub-iterations in the coarse phase
        max_inner_iters_fine=None):  # ... in the fine phase (None -> same as max_inner_iters)
    """Drop-in for simulate_airbag(make_plots=False, verbose=False): same scalar outputs."""
    if max_inner_iters_fine is None:
        max_inner_iters_fine = max_inner_iters

    # sizing, exactly as in simulate_airbag
    A_or = (np.pi / 4.0) * d_or**2
    R0 = D0 / 2.0
    k_shape = 1.0 if shape == 1 else 0.5
    d_fabric    = n_ply * t_ply
    N_ult       = 100.0 * n_ply * N_ult_Ncm
    N_allow     = eta_seam * N_ult / sf_burst
    P_burst     = P_amb + N_allow / (k_shape * R0)
    P_burst_ult = P_amb + N_ult / (k_shape * R0)
    if shape == 1:
        S_surface = np.pi * D0 * L0 + 0.5 * np.pi * D0**2
    else:
        S_surface = np.pi * D0**2
    m_bag = n_ply * areal_weight * S_surface

    (m_gas0, theta, a_peak_y, a_peak_n, a_peak_res, P_peak, h_max, x_final, u_final_n,
     burst, bottomed, rebounded, u_rebound_y, u_residual_y, u_residual_n, u_residual_x,
     t_stop) = _run(float(D0), float(L0), float(A_or), float(P0), int(shape), float(M_payload),
                    float(uy0), float(ux0), float(T0), float(R_gas), float(gamma), float(P_amb),
                    float(g), float(P_burst), float(S_surface), float(phi_vent), float(a_allow),
                    float(dt), float(t_max), int(max_inner_iters), int(max_inner_iters_fine))

    nan = float("nan")
    return {
        "m_total": m_bag + m_gas0,
        "m_bag": m_bag,
        "m_gas0": m_gas0,
        "S_surface": S_surface,
        "A_vent0": A_or + phi_vent * S_surface,
        "a_peak_y": a_peak_y if np.isfinite(a_peak_y) else nan,
        "a_peak_res": a_peak_res if np.isfinite(a_peak_res) else nan,
        "a_peak_n": a_peak_n if np.isfinite(a_peak_n) else nan,
        "theta": theta,
        "survived": not (burst or rebounded),
        "burst": bool(burst),
        "rebounded": bool(rebounded),
        "u_rebound_y": u_rebound_y,
        "bottomed": bool(bottomed),
        "u_residual_y": u_residual_y,
        "u_residual_n": u_residual_n,
        "u_residual_x": u_residual_x,
        "t_stop": None if t_stop < 0.0 else t_stop,
        "h_max": h_max,
        "x_final": x_final,
        "P_burst": P_burst,
        "P_burst_ult": P_burst_ult,
        "N_allow": N_allow,
        "d_fabric": d_fabric,
        "n_ply": n_ply,
        "P_peak": P_peak,
        "settled": not (burst or bottomed or rebounded),
        "u_final_n": u_final_n,
        "D_abrasion": 0.0,
    }
