from airbag_simulation_function_rev02 import simulate_airbag
import numpy as np
import contextlib

# Environment
g = 9.81
P_amb = 101325
# mu  = 0.4          # [-]   bag-fabric / ground sliding friction coefficient


# Payload 
M_payload = 0.5
u0 = 17.2        # [m/s] initial vertical velocity (design parameter)
ux0 = 5          # [m/s] initial horizontal velocity (design parameter)


# Geometry
shape = 1        # 1 = cylinder, 2 = sphere
D0 = 0.8   # [m] initial airbag height (diameter) 
L0 = 0.1043377808026541       # [m] airbag axial length, not used if shape == 2 (sphere)
d_or = 0.095492405598105    # [m] orifice diameter

# Fabric 
d_fabric = 0.0005238632242683665 # [m] fabric thickness
sigma_fabric = 250e6   # [Pa] effective membrane strength (×0.002 m = 50000 N/m, previously T_fabric)
rho_fabric   = 1100.0    # [kg/m^3] 
# c_abrasion   = 3.0e4     # [J/m] abrasion energy per unit thickness (×0.002 m = 5000 J)


# Gas parameters (air)
P0 = 131781.32180997377     # [Pa] initial airbag inflation pressure
# P0 = 130000
T0 = 284.5695298438421      # [K] initial airbag temperature
R_gas = 2077.0    # [J/(kg K)]
gamma = 1.66


# Time and other settings
dt = 1e-5
# dt = [1e-3, 5e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
# timesteps_and_accelerations_list = []
t_max = 1
make_plots=True
verbose=True


with open(r"C:\TEMP\full_log.txt", "w") as f:
    with contextlib.redirect_stdout(f):

        # res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0, ux0, mu, sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g, dt, t_max, make_plots, verbose)
        res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g, dt, t_max, make_plots, verbose)

# res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g, dt, t_max, make_plots, verbose)


"""
for dt_test in dt:
    res = simulate_airbag(D0, L0, d_or, d_fabric, P0, shape, M_payload, u0,sigma_fabric, rho_fabric, T0, R_gas, gamma, P_amb, g, dt_test, t_max, make_plots, verbose)
    print("Calculating for", dt_test, "s")
    timesteps_and_accelerations_list.append([dt_test, res['a_peak']])
print(timesteps_and_accelerations_list)

"""

print("done — check C:\\TEMP\\full_log.txt")

print(f"{'Fabric mass':<14}: {res['m_bag']:.5f} kg")
print(f"{'Gas mass':<14}: {res['m_gas0']:.5f} kg")
print(f"{'Total mass':<14}: {res['m_total']:.5f} kg")
print(f"Peak accel  : {res['a_peak']:.5f} m/s^2  ({res['a_peak']/9.81:.2f} g)")
print(f"{'u_residual':<14}: {res['u_residual']:.5f} m/s")
print(f"Rebounded    : {res['rebounded']}")
print(f"Bottomed    : {res['bottomed']}")
print(f"Burst    : {res['burst']}")
print(f"Survived    : {res['survived']}")
