import contextlib
import os

from airbag_simulation_function import simulate_airbag
from differential_evolution_wrapper import (A_ALLOW, SF, A_LIM, U_RES_MAX,
                                            ETA_SEAM, SF_BURST, evaluate_design)
from airbag_postprocessing import run_postprocessing

# ---------------------------- output options ---------------------------------
POSTPROCESS             = True    # False -> numbers only, no plots or animations
SHOW_POPUPS             = True    # open the figures below in windows

SAVE_PLOT_MATRIX        = True
PLOT_MATRIX_FORMATS     = ("png", "pdf")

MAKE_ANIMATION          = True    # bag-only animation
SAVE_ANIMATION          = False   # .mp4 and .gif

MAKE_COMBINED_ANIMATION = False   # plot matrix with time cursor + bag
SAVE_COMBINED_ANIMATION = False   # .mp4 and .gif

OUTPUT_DIR              = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TEST_outputs")
# -----------------------------------------------------------------------------

# Environment
g = 9.81
P_amb = 101325

# Payload
M_payload = 0.5
u0 = 17.2                 # [m/s] vertical impact velocity
ux0 = 5                   # [m/s] horizontal impact velocity
scenario = (M_payload, u0, ux0)

# Geometry
shape = 1                 # 1 = cylinder, 2 = sphere
D0 = 1.1379               # [m] diameter
L0 = 0.6077               # [m] axial length (cylinder only)
d_or = 0                  # [m] orifice diameter (the optimizer always uses 0)
phi_vent = 0.0574         # [-] breathing-fabric open-area fraction
# d_or = 0 and phi_vent = 0 is a sealed bag: no venting, the payload bounces back.

# Fabric - supplier data: strength per unit width, areal weight, single-ply thickness.
# Thickness is not a strength lever on its own: n_ply plies carry n_ply times the load.
T_ult_Ncm = 300.0         # [N/cm] tensile strength per unit width (1 N/cm = 100 N/m)
areal_weight = 0.420      # [kg/m^2] areal weight (420 g/m^2)
t_ply = 0.40e-3           # [m] thickness of one ply
n_ply = 2                 # [-] number of plies -> 0.80 mm stack

# Gas
P0 = 126691.8             # [Pa] inflation pressure
T0 = 288.0                # [K]
R_gas = 2077.0            # [J/(kg K)]
gamma = 1.66

# Time and other settings
dt = 1e-5
t_max = 1
verbose = True

LOG_PATH = r"C:\TEMP\full_log.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

inputs = dict(D0=D0, L0=L0, d_or=d_or, P0=P0, shape=shape,
              M_payload=M_payload, u0=u0,
              T_ult_Ncm=T_ult_Ncm, areal_weight=areal_weight, t_ply=t_ply, n_ply=n_ply,
              T0=T0, R_gas=R_gas, gamma=gamma, P_amb=P_amb, g=g,
              dt=dt, t_max=t_max, a_allow=A_LIM, ux0=ux0, phi_vent=phi_vent,
              eta_seam=ETA_SEAM, sf_burst=SF_BURST)

with open(LOG_PATH, "w") as f:
    with contextlib.redirect_stdout(f):
        res = simulate_airbag(**inputs, make_plots=False, verbose=verbose)

print(f"done — check {LOG_PATH}")

print(f"{'Fabric mass':<14}: {res['m_bag']:.5f} kg")
print(f"{'Gas mass':<14}: {res['m_gas0']:.5f} kg")
print(f"{'Total mass':<14}: {res['m_total']:.5f} kg")
print(f"Peak accel  : {res['a_peak']:.5f} m/s^2  ({res['a_peak']/9.81:.2f} g)  [vertical]")
print(f"Peak load n : {res['a_peak_n']:.5f} m/s^2  ({res['a_peak_n']/9.81:.2f} g)  [F_n/M, sizing quantity, "
      f"limit {A_LIM:.1f} = {A_ALLOW:.0f}/{SF}]")
print(f"{'u_residual':<14}: {res['u_residual']:.5f} m/s  [vertical]")
print(f"{'u_residual_n':<14}: {res['u_residual_n']:.5f} m/s  [along n]")
print(f"{'Fabric stack':<14}: {res['n_ply']} ply x {1e3*t_ply:.3f} mm = {res['d_fabric']*1e3:.4f} mm")
print(f"{'T_allow':<14}: {res['T_allow']/100:.1f} N/cm of {n_ply*T_ult_Ncm:.1f} N/cm ultimate (eta_seam={ETA_SEAM}, SF_burst={SF_BURST})")
print(f"P_peak/burst: {res['P_peak']:.1f} / {res['P_burst']:.1f} Pa (unfactored {res['P_burst_ult']:.1f} Pa)")
print(f"Rebounded    : {res['rebounded']}")
print(f"Bottomed    : {res['bottomed']}")
print(f"Burst    : {res['burst']}")
print(f"Survived    : {res['survived']}  [burst/rebound only, not the optimizer's criteria]")

violations, feasible = evaluate_design(res, D0, scenario)
print(f"FEASIBLE     : {feasible}  [optimizer criteria: a_peak_n<=A_LIM, "
      f"u_residual_n<=U_RES_MAX={U_RES_MAX}, bottomed, no burst/rebound]")
print(f"  violations : " + ", ".join(f"{k}={val:.3g}" for k, val in violations.items()))

if POSTPROCESS:
    run_postprocessing(inputs, res, feasible,
                       show_popups=SHOW_POPUPS,
                       save_plot_matrix=SAVE_PLOT_MATRIX,
                       plot_matrix_formats=PLOT_MATRIX_FORMATS,
                       make_animation=MAKE_ANIMATION,
                       save_animation=SAVE_ANIMATION,
                       make_combined_animation=MAKE_COMBINED_ANIMATION,
                       save_combined_animation=SAVE_COMBINED_ANIMATION,
                       output_dir=OUTPUT_DIR)
