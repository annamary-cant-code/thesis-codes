# Outputs:
#   D_t  : height of deformed airbag
#   L_t  : airbag footprint length
#   A_t  : footprint contact area
#   V_t  : airbag internal volume
#   S_t  : cross-sectional area


import numpy as np
def airbag_geometry_from_h(h_t, D0, L0):
    
    # Current height of deformed bag
    D_t = D0 - h_t

    # Prevent negative/zero height (numerical safety)
    D_t_eff = max(D_t, 1e-12)

    # Footpring length (geometric assumption, see eq. 6 in Zhou et al. 2019)
    L_t = 0.5 * np.pi * h_t

    # Footprint area (rectangular)
    A_t = L_t * L0

    # Volume
    V_t = L0 * D_t_eff * ( (np.pi / 4) * D_t_eff + L_t )

    # Cross-sectional area
    S_t = (np.pi / 4) * D_t_eff**2 + D_t_eff * L_t

    return {
        "D_t": D_t,
        "L_t": L_t,
        "A_t": A_t,
        "V_t": V_t,
        "S_t": S_t
    }
