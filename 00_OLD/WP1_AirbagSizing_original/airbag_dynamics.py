import numpy as np
def system_dynamics_step(h_prev, u_prev, P_bag, A_t, M, g, P_atm, dt):

    # Acceleration
    a = g - ((P_bag - P_atm) * A_t) / M

    # Velocity update
    u = u_prev + a * dt                             # u_t = u_{t-1} + Δu

    # Displacement update
    h = h_prev + u_prev * dt + 0.5 * a * dt**2           # h_t = h_{t-1} + u_t*Δt + 0.5*a*Δt^2

    return a, u, h
