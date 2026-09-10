import numpy as np

# Not used by simulate_airbag (its 2-DOF dynamics are inlined there); kept for reference.


def system_dynamics_step(h_prev, u_prev, P_bag, A_t, M, g, P_atm, dt):
    a = g - ((P_bag - P_atm) * A_t) / M
    u = u_prev + a * dt
    h = h_prev + u * dt
    return a, u, h


def horizontal_dynamics_step(x_prev, ux_prev, P_bag, A_t, mu, M, P_atm, dt):
    """Kinetic friction at the contact patch. Returns a_x, ux_t, x_t, N, F_fric, dW_fric."""
    N = max(P_bag - P_atm, 0.0) * A_t
    F_fric = mu * N

    dv = (F_fric / M) * dt
    if abs(ux_prev) <= dv:
        ux_t = 0.0                             # friction stops the motion, never reverses it
    else:
        ux_t = ux_prev - np.sign(ux_prev) * dv

    x_t = x_prev + ux_prev * dt

    a_x = (ux_t - ux_prev) / dt

    dW_fric = F_fric * abs(ux_prev) * dt       # energy dissipated this step

    return {
        "a_x": a_x,
        "ux_t": ux_t,
        "x_t": x_t,
        "N": N,
        "F_fric": F_fric,
        "dW_fric": dW_fric,
    }
