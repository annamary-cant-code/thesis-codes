import numpy as np
from physics.fun_xv_x  import fun_xv_x
from physics.fun_tc    import fun_tc
from physics.fun_vy    import fun_vy
from physics.fun_vx    import fun_vx
from physics.fun_yu    import fun_yu
from physics.fun_tdrop import fun_tdrop

def compute_impact_point_ballistic_descent(
        altezza, Cd_mu, Cd_sigma, densita, g, m, S,
        Vx_i_mu, Vx_i_sigma, Vy_i_mu, Vy_i_sigma, guess):
    """
    Vectorised port of MATLAB computeImpactPointBallisticDescent.
    All outputs are (guess,) arrays.

    NOTE: Cd is drawn once and shared across all simulations,
    matching the MATLAB behaviour: normrnd(Cd_mu, Cd_sigma)*ones(guess,1)
    """

    # --- Random inputs ---
    Cd    = np.random.normal(Cd_mu, Cd_sigma)              # scalar — shared across all sims
    c     = 0.5 * S * densita * Cd * np.ones(guess)        # (guess,)
    Vx_i  = np.random.normal(Vx_i_mu, Vx_i_sigma, guess)  # (guess,)
    Vy_i  = np.random.normal(Vy_i_mu, Vy_i_sigma, guess)  # (guess,)

    y = altezza.copy()   # (guess,) — altitude at separation

    # --- Derived constants ---
    Gamma = np.sqrt(m * g / c)
    gamma = 1.0 / Gamma

    # --- Clip negative Vy_i to zero (payload already descending) ---
    Vy_i = np.where(Vy_i < 0, 0.0, Vy_i)


    # --- intermediate constants - precomputed terms that appear repeatedly in the analytical trajectory equations
    Hu = np.arctan(Vy_i * gamma)
    Hd = np.arctanh(Vy_i * gamma)
    Gu = -0.5 * np.log(1 + (Vy_i * gamma) ** 2)
    Gd = -0.5 * np.log(1 - (Vy_i * gamma) ** 2)

    # --- Phase 1: upward ---
    t_top_hat = -(Gamma / g) * np.arctan(gamma * Vy_i)
    x1        = fun_xv_x(t_top_hat, Vx_i, m, c)
    y_top     = fun_yu(m, c, g, gamma, t_top_hat, Hu, Gu)

    # --- Phase 2: descent to Vx == Vy crossing ---
    tc_hat = fun_tc(m, g, Vx_i, c, gamma, t_top_hat, Gamma, Hd)

    # --- Total impact time ---
    t_drop_hat = fun_tdrop(m, c, y - y_top, Gd, Hd, Gamma, g)
    t_im       = t_top_hat + t_drop_hat

    # --- Horizontal velocity at peak ---
    Vx_top = m * Vx_i / (m + Vx_i * c * t_top_hat)

    # --- x2: peak to Vx==Vy crossing ---
    minore = np.minimum(t_im, tc_hat)
    x2     = m / c * np.log(1 + c * Vx_top * (minore - t_top_hat) / m)

    # --- Velocities at crossing ---
    Vx_c = m * Vx_i / (m + Vx_i * tc_hat)
    Vy_c = Gamma * np.tanh(g * gamma * (tc_hat - t_top_hat) + Hd)

    # --- Phase 3: crossing to impact ---
    Hc = np.arctan(gamma * Vy_c)
    Gc = np.log(np.cosh(Hc))
    x3 = ((Vx_c * np.exp(Gc) * Gamma) / g) * (
         np.arctan(np.sinh(g * gamma * (t_im - tc_hat) + Hc))
         - np.arcsin(gamma * Vy_c))

    x = x1 + x2 + x3

    # --- Velocity arrays at impact (single time point = t_im per sim) ---
    Vx_impact = fun_vx(t_im, tc_hat, m, Vx_i, c, gamma, Hd, g, Gd)
    Vy_impact = fun_vy(t_im, Vy_i, gamma, Gamma, Hu, Hd, g, t_top_hat)

    return x, t_im, Vx_impact, Vy_impact