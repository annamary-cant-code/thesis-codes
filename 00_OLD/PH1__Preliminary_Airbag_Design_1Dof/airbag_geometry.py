# Outputs:
#   D_t  : height of deformed airbag
#   L_t  : airbag footprint length
#   A_t  : footprint contact area
#   V_t  : airbag internal volume
#   S_t  : cross-sectional area

import numpy as np

def airbag_geometry_from_h(h_t, D0, L0, shape=1):
    # Current vertical height of the deformed bag (both shapes)
    # if h_t <= 0:
    #     h_t = 0 # evita forze di trazione

    D_t = D0 - h_t
    D_t_eff = max(D_t, 1e-12)   # numerical safety against zero/negative

    
    if shape == 1:
        # ---- Cylinder: stadium cross-section extruded over L0 ----
        L_t = 0.5 * np.pi * h_t                                  # footprint length (membrane assumption)
        A_t = L_t * L0                                           # rectangular footprint
        V_t = L0 * D_t_eff * ((np.pi / 4) * D_t_eff + L_t)       # = L0 * S_t
        S_t = (np.pi / 4) * D_t_eff**2 + D_t_eff * L_t           # stadium cross-section

    elif shape == 2:
        # ---- Sphere: flattened by a spherical cap of height h_t ----
        R0 = D0 / 2.0
        r_c = np.sqrt(max(h_t * (D0 - h_t), 0.0))                # cap base radius
        A_t = np.pi * r_c**2                                     # circular footprint
        V_sphere = (np.pi / 6.0) * D0**3                         # full sphere
        V_cap = (np.pi / 3.0) * h_t**2 * (1.5 * D0 - h_t)        # removed cap
        V_t = V_sphere - V_cap
        L_t = 2 * r_c                                            # stadium footprint length, unused
        S_t = np.nan                                             # stadium cross-section, unused

    else:
        raise ValueError(f"Unknown shape={shape!r}; use 1 (cylinder) or 2 (sphere).")

    return {
        "D_t": D_t,
        "L_t": L_t,
        "A_t": A_t,
        "V_t": V_t,
        "S_t": S_t
    }