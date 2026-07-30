import numpy as np

def fun_tc(m, g, Vx_i, c, gamma, t_top, Gamma, Hd):
    """Time at which horizontal and vertical velocities are equal (Vx == Vy)."""
    numerator   = m * (g * t_top - Gamma * Hd + Vx_i * (1 + (Hd - gamma * g * t_top) ** 2))
    denominator = m * g + c * Vx_i * (g * t_top - Gamma * Hd)
    return numerator / denominator