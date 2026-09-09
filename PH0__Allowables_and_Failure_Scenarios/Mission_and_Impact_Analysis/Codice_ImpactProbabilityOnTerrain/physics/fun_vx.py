import numpy as np

def fun_vx(t, tc, m, Vx_i, c, gamma, Hd, g, Gd):
    """Horizontal velocity at time t.
    Two regimes: before crossing time tc and after.
    """
    Vx = np.zeros_like(t, dtype=float)

    below = t < tc
    above = ~below

    Vx[below] = (m * Vx_i[below]) / (m + Vx_i[below] * c[below] * t[below])
    Vx[above] = Vx_i[above] * np.exp(Gd[above]) / np.cosh(g * gamma[above] * t[above] + Hd[above])

    return Vx