import numpy as np


def airbag_geometry_from_h(h_t, D0, L0, shape=1):
    """Deformed geometry for crush h_t: height D_t, footprint length L_t,
    footprint area A_t, volume V_t, cross-section S_t (cylinder only)."""

    D_t = D0 - h_t
    D_t_eff = max(D_t, 1e-12)

    if shape == 1:
        # cylinder: stadium cross-section extruded over L0
        L_t = 0.5 * np.pi * h_t                                  # inextensible membrane
        A_t = L_t * L0
        V_t = L0 * D_t_eff * ((np.pi / 4) * D_t_eff + L_t)
        S_t = (np.pi / 4) * D_t_eff**2 + D_t_eff * L_t

    elif shape == 2:
        # sphere flattened by a spherical cap of height h_t
        R0 = D0 / 2.0
        r_c = np.sqrt(max(h_t * (D0 - h_t), 0.0))                # footprint radius
        A_t = np.pi * r_c**2
        V_sphere = (np.pi / 6.0) * D0**3
        V_cap = (np.pi / 3.0) * h_t**2 * (1.5 * D0 - h_t)
        V_t = V_sphere - V_cap
        L_t = 2 * r_c
        S_t = np.nan

    else:
        raise ValueError(f"Unknown shape={shape!r}; use 1 (cylinder) or 2 (sphere).")

    return {
        "D_t": D_t,
        "L_t": L_t,
        "A_t": A_t,
        "V_t": V_t,
        "S_t": S_t
    }
