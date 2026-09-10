import numpy as np


def Cd_from_lambda(lmbd):
    """Discharge coefficient as a function of the pressure ratio P_down / P_up."""
    return (-3.8399*lmbd**6
            + 9.4363*lmbd**5
            - 7.2326*lmbd**4
            + 1.6972*lmbd**3
            - 0.2908*lmbd**2
            - 0.0130*lmbd
            + 0.8426)


def orifice_mdot(P_up, P_down, T_up, A_or, R_gas, gamma):
    """Compressible mass flow through an orifice [kg/s], subsonic or choked."""
    if P_up <= P_down:
        return 0.0

    lmbd = P_down / P_up

    lmbd_clip = float(np.clip(lmbd, 0.0, 1.0))
    C_D = Cd_from_lambda(lmbd_clip)
    C_D = max(C_D, 0.0)

    common_term = np.sqrt(1.0 / (R_gas * T_up))

    if lmbd >= 0.528:  # subsonic

        pr = P_up / P_down
        expo = (gamma - 1.0) / gamma

        term1 = np.sqrt((2.0 * gamma) / (gamma - 1.0) * (pr ** expo))
        term2_inside = (pr ** expo) - 1.0
        term2 = np.sqrt(max(term2_inside, 0.0))

        mdot = C_D * A_or * P_down * common_term * term1 * term2
        return float(max(mdot, 0.0))

    else:  # choked
        choke_factor = np.sqrt(gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
        mdot = C_D * A_or * P_up * common_term * choke_factor
        return float(max(mdot, 0.0))


def delta_m_vented(P_bag, P_amb, T_bag, A_or, R_gas, gamma, dt):
    mdot = orifice_mdot(P_up=P_bag, P_down=P_amb, T_up=T_bag,
                        A_or=A_or, R_gas=R_gas, gamma=gamma)
    return mdot * dt  # [kg]
