import numpy as np
def system_dynamics_step(h_prev, u_prev, P_bag, A_t, M, g, P_atm, dt):
    # Acceleration (M = M_payload)
    a = g - ((P_bag - P_atm) * A_t) / M

    # Velocity update
    u = u_prev + a * dt                             # u_t = u_{t-1} + Δu

    # Displacement update
    h = h_prev + u * dt # + 0.5 * a * dt**2           # h_t = h_{t-1} + u_t*Δt + 0.5*a*Δt^2

    return a, u, h






################################################################################################################
""" x_prev, ux_prev : float   horizontal position [m] and velocity [m/s] at step start
    P_bag           : float   bag absolute pressure [Pa]
    A_t             : float   contact-patch area [m^2]
    mu              : float   sliding friction coefficient [-]
    M               : float   payload mass [kg]
    P_atm           : float   ambient pressure [Pa]
    dt              : float   timestep [s]

    Returns:
        a_x      : effective horizontal acceleration this step [m/s^2]
        ux_t     : updated horizontal velocity [m/s]
        x_t      : updated horizontal position [m]
        N        : normal load at the patch [N]
        F_fric   : kinetic friction magnitude [N]
        dW_fric  : frictional energy dissipated this step [J] """


def horizontal_dynamics_step(x_prev, ux_prev, P_bag, A_t, mu, M, P_atm, dt):
    N = max(P_bag - P_atm, 0.0) * A_t          # normal load at patch
    F_fric = mu * N                            # kinetic friction magnitude

    # Update velocity without letting friction reverse the motion
    dv = (F_fric / M) * dt
    if abs(ux_prev) <= dv:
        ux_t = 0.0                             # friction brings it to rest this step
    else:
        ux_t = ux_prev - np.sign(ux_prev) * dv

    x_t = x_prev + ux_prev * dt                # forward Euler, matches the scheme

    # Effective acceleration consistent with the actual velocity change
    # (correct even on the step where the clamp fires)
    a_x = (ux_t - ux_prev) / dt

    # Abrasion dosage: frictional energy dissipated this step
    dW_fric = F_fric * abs(ux_prev) * dt

    return {
        "a_x": a_x,
        "ux_t": ux_t,
        "x_t": x_t,
        "N": N,
        "F_fric": F_fric,
        "dW_fric": dW_fric,
    }