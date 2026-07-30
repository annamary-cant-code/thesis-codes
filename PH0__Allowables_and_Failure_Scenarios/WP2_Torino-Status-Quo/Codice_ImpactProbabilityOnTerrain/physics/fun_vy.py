import numpy as np

def fun_vy(t, Vy_i, gamma, Gamma, Hu, Hd, g, t_top):
    """Vertical velocity at time t.
    Two regimes: ascending (t < t_top) and descending (t >= t_top).
    All inputs are (guess,) arrays except t which is (guess,) too.
    """
    Vy = np.zeros_like(t, dtype=float)

    below = t < t_top
    above = ~below

    Vy[below] = Gamma[below] * np.tan(g * gamma[below] * t[below] + Hu[below])
    Vy[above] = Gamma[above] * np.tanh(g * gamma[above] * t[above] + Hd[above])

    # Special case: t == 0 and Vy_i == Gamma
    special = (t == 0) & (Vy_i == Gamma)
    Vy[special] = Gamma[special]

    return Vy