import numpy as np
from physics.compute_impact_point import compute_impact_point_ballistic_descent

def impact_point_ballistic_descent(
        N1, N2, altezza_mu, altezza_sigma, massa, superficie,
        mu_vento, sigma_vento, mu_drone, sigma_drone, theta,
        Cd_mu, Cd_sigma, Vx_i_mu, Vx_i_sigma, Vy_i_mu, Vy_i_sigma,
        center_x, center_y, guess, g, densita, x_limits, y_limits):
    """
    Vectorised port of MATLAB impact_point_ballistic_descent.
    Returns pdf (N1 x N2), impact_angle_matrix, Vx_impact_matrix, Vy_impact_matrix.
    """

    # --- Rotation matrix for drone heading ---
    theta_rad = theta / 180.0 * np.pi
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    matr_1 = np.array([[cos_t, -sin_t],
                        [sin_t,  cos_t]])   # (2,2)

    # --- Random wind and drone heading ---
    phi_vals = np.random.normal(mu_drone  / 180.0 * np.pi,
                                sigma_drone / 180.0 * np.pi, guess)   # (guess,)
    w_vals   = np.random.normal(mu_vento, sigma_vento, guess)          # (guess,)

    matr_2_list = np.column_stack([np.cos(phi_vals),
                                   np.sin(phi_vals)])                  # (guess, 2)

    # --- Random altitude at failure ---
    altezza_vals = np.random.normal(altezza_mu, altezza_sigma, guess)  # (guess,)

    # --- Physics: impact range and time ---
    x, t_im, Vx_impact, Vy_impact = compute_impact_point_ballistic_descent(
        altezza_vals, Cd_mu, Cd_sigma, densita, g, massa, superficie,
        Vx_i_mu, Vx_i_sigma, Vy_i_mu, Vy_i_sigma, guess)

    # --- Build impact positions in local frame then rotate ---
    x_expanded  = np.column_stack([x, np.zeros(guess)])     # (guess, 2)
    second_term = (w_vals * t_im)[:, None] * matr_2_list    # (guess, 2)  wind drift
    p           = x_expanded @ matr_1.T + second_term       # (guess, 2)

    # --- Translate to real-world coordinates ---
    p[:, 0] += center_x
    p[:, 1] += center_y

    # --- 2D histogram -> PDF ---
    pdf, x_edges, y_edges = np.histogram2d(
        p[:, 0], p[:, 1],
        bins=[N1, N2],
        range=[x_limits, y_limits],
        density=False)
    pdf = pdf / guess   # normalise to probability

    # --- Per-cell average impact angle and velocities ---
    impact_velocities = np.column_stack([Vx_impact, Vy_impact])  # (guess, 2)
    impact_angles     = np.arctan2(impact_velocities[:, 1],
                                   impact_velocities[:, 0])       # (guess,)

    # Bin each impact point to a cell
    x_idx = np.searchsorted(x_edges, p[:, 0], side='right') - 1  # (guess,)
    y_idx = np.searchsorted(y_edges, p[:, 1], side='right') - 1  # (guess,)

    # Valid mask: inside grid
    valid = (x_idx >= 0) & (x_idx < N1) & (y_idx >= 0) & (y_idx < N2)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    angles_v  = impact_angles[valid]
    Vx_v      = impact_velocities[valid, 0]
    Vy_v      = impact_velocities[valid, 1]

    # Flat indices for accumulation (matches MATLAB sub2ind)
    flat_idx = np.ravel_multi_index((x_idx, y_idx), (N1, N2))

    # Accumulate sums and counts using bincount
    angle_sum  = np.bincount(flat_idx, weights=angles_v, minlength=N1*N2)
    angle_cnt  = np.bincount(flat_idx,                   minlength=N1*N2)
    Vx_sum     = np.bincount(flat_idx, weights=Vx_v,     minlength=N1*N2)
    Vy_sum     = np.bincount(flat_idx, weights=Vy_v,     minlength=N1*N2)

    # Average per cell (avoid divide by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        impact_angle_matrix = np.where(angle_cnt > 0, angle_sum / angle_cnt, 0.0)
        Vx_impact_matrix    = np.where(angle_cnt > 0, Vx_sum   / angle_cnt, 0.0)
        Vy_impact_matrix    = np.where(angle_cnt > 0, Vy_sum   / angle_cnt, 0.0)

    # Reshape to (N1, N2)
    impact_angle_matrix = impact_angle_matrix.reshape(N1, N2)   #  angle of the velocity vector at ground impact
    Vx_impact_matrix    = Vx_impact_matrix.reshape(N1, N2)
    Vy_impact_matrix    = Vy_impact_matrix.reshape(N1, N2)

    return pdf, impact_angle_matrix, Vx_impact_matrix, Vy_impact_matrix, Vx_impact, Vy_impact, t_im

""" 
for cell (i,j): 

impact_angle_matrix[i,j]  = average impact angle  of all payloads that hit cell (i,j)
Vx_impact_matrix[i,j]     = average horizontal velocity of all payloads that hit cell (i,j)
Vy_impact_matrix[i,j]     = average vertical velocity of all payloads that hit cell (i,j)"

"""
