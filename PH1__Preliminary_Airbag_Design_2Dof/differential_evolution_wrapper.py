import numpy as np
from airbag_simulation_function import simulate_airbag
from scipy.optimize import differential_evolution


# ---------- run size ----------
QUICK_TEST = True   # True -> reduced sweep for test runs, False -> full sweep

if QUICK_TEST:
    W_LAND_VALUES = [10]
    SEEDS         = (1,)
    POPSIZE       = 2
    MAXITER       = 6
else:
    W_LAND_VALUES = [10, 50, 100, 200]
    SEEDS         = (1, 2, 3)
    POPSIZE       = 5
    MAXITER       = 20


# ---------- requirements ----------
A_ALLOW   = 500.0          # [m/s^2] allowable payload acceleration along n
SF        = 1.2            # [-]     safety factor
A_LIM     = A_ALLOW / SF   # [m/s^2] limit on the peak bag load F_n/M
U_RES_MAX = 2.0            # [m/s]   max residual speed along n at full stroke
PEN       = 1.0e4          # penalty weight, must dominate mass

# venting sub-iterations during the search only (~2x faster, ~1% error on a_peak_n);
# the winner is re-simulated at full fidelity
MAX_INNER_ITERS_SEARCH = 20

SHAPE_CODE = {"sphere": 2, "cylinder": 1}

all_winners = []


# ---------- impact scenario (fixed) ----------
M_payload = 0.5          # kg
u0        = 17.2         # m/s   vertical impact velocity
ux0       = 5.0          # m/s   horizontal impact velocity
scenario  = (M_payload, u0, ux0)

# ---------- environment ----------
P_amb = 101325.0         # Pa
g     = 9.81             # m/s^2
env   = (P_amb, g)

T0 = 288.0               # K, fixed

# ---------- materials: (sigma_fabric, rho_fabric) ----------
materials = {
    "matA": (400e6, 1400.0),
    "matB": (250e6, 1100.0),
}

# ---------- gases: (R_gas, gamma) ----------
gases = {                # R [J/kg·K]
    "N2": (296.8, 1.40),
    "He": (2077.0, 1.66),
    "Ar": (208.1, 1.67),
}

# material only changes mass and burst pressure, not the dynamics -> one is enough for a quick test
MATERIAL_NAMES = ["matB"] if QUICK_TEST else list(materials.keys())
GAS_NAMES = list(gases.keys())

# ---------- design-space bounds ----------
D0_MAX       = 1.5             # [m]
ASPECT_RANGE = (0.5, 4.0)      # [-]  L0 / D0, cylinder only
DFAB_RANGE   = (0.0001, 0.002) # [m]
P0_MAX       = 1.4e5           # [Pa] lower bound is P_amb
PHI_RANGE    = (0.0, 0.10)     # [-]  breathing-fabric open-area fraction


def impact_axis(scenario):
    """Impact angle theta [rad] and impact speed along n [m/s]."""
    _, u0_s, ux0_s = scenario
    return np.arctan2(ux0_s, u0_s), np.hypot(u0_s, ux0_s)


def min_stroke(scenario, env):
    """Stroke needed to stop the payload at a constant load A_LIM: lower bound on D0."""
    theta, u_n0 = impact_axis(scenario)
    decel = A_LIM - env[1] * np.cos(theta)
    if decel <= 0.0:
        raise ValueError(f"A_LIM = {A_LIM:.1f} m/s^2 cannot even hold the payload against gravity")
    return u_n0**2 / (2.0 * decel)


def unpack_design(x, shape):
    """cylinder: x = [D0, L0/D0, d_fabric, P0, phi_vent]; sphere: x = [D0, d_fabric, P0, phi_vent]."""
    if shape == SHAPE_CODE["cylinder"]:
        D0, aspect, d_fabric, P0, phi_vent = x
        L0 = aspect * D0
    else:
        D0, d_fabric, P0, phi_vent = x
        L0 = D0                                      # ignored by the sphere geometry
    return dict(D0=D0, L0=L0, d_or=0.0, d_fabric=d_fabric,
                P0=P0, shape=shape, T0=T0, phi_vent=phi_vent)


def make_bounds(shape, scenario, env):
    b_D0     = (min_stroke(scenario, env), D0_MAX)
    b_aspect = ASPECT_RANGE
    b_dfab   = DFAB_RANGE
    b_P0     = (env[0], P0_MAX)
    b_phi    = PHI_RANGE
    return ([b_D0, b_aspect, b_dfab, b_P0, b_phi] if shape == SHAPE_CODE["cylinder"]
            else [b_D0, b_dfab, b_P0, b_phi])


def evaluate_design(r, D0, scenario):
    """Graded violations (0 = satisfied) and feasibility, shared by objective and summary."""
    u0_s = scenario[1]
    _, u_n0 = impact_axis(scenario)
    v = {
        "acc":   max(0.0, r["a_peak_n"] - A_LIM) / A_LIM,
        "burst": max(0.0, r["P_peak"] - r["P_burst"]) / r["P_burst"],
        "reb":   abs(r["u_rebound"]) / u0_s if r["rebounded"] else 0.0,   # vertical rebound
        "nobot": ((D0 - r["h_max"]) / D0
                  if not (r["bottomed"] or r["burst"] or r["rebounded"]) else 0.0),
        "ures":  (max(0.0, r["u_residual_n"] - U_RES_MAX) / u_n0
                  if r["bottomed"] else 0.0),
    }
    feasible = (r["bottomed"] and not r["burst"] and not r["rebounded"]
                and r["a_peak_n"] <= A_LIM and r["u_residual_n"] <= U_RES_MAX)
    return v, feasible


def airbag_objective(x, shape, mat, gas, scenario, env, W_land):
    d = unpack_design(x, shape)

    try:
        r = simulate_airbag(
            **d,
            M_payload=scenario[0], u0=scenario[1], ux0=scenario[2],
            sigma_fabric=mat[0], rho_fabric=mat[1],
            R_gas=gas[0], gamma=gas[1],
            P_amb=env[0], g=env[1],
            a_allow=A_LIM,
            max_inner_iters=MAX_INNER_ITERS_SEARCH,
        )
    except Exception as e:
        print(f"[SIM CRASH] shape={shape} x={x} → {type(e).__name__}: {e}")
        return 1.0e12

    m = r["m_total"]
    if not np.isfinite(m):
        return 1.0e12

    v, feasible = evaluate_design(r, d["D0"], scenario)
    if not feasible:
        # no mass reward: an infeasible light bag must never beat a feasible one
        return 1.0e6 + PEN * sum(v.values())

    # W_land = 100: 1 kg of bag is worth the residual speed going from u_n0 to zero
    _, u_n0 = impact_axis(scenario)
    return m + W_land * (r["u_residual_n"] / u_n0)


def print_design_summary(shape_name, mat_name, gas_name, x, shape, r):
    d = unpack_design(x, shape)
    theta, u_n0 = impact_axis(scenario)
    v, feasible = evaluate_design(r, d["D0"], scenario)
    is_cyl = shape == SHAPE_CODE["cylinder"]

    print("\n" + "="*50)
    print(f"OPTIMIZED DESIGN — {shape_name} / {mat_name} / {gas_name}")
    print("="*50)
    print(f"  Impact           : u0 = {u0:.2f} m/s, ux0 = {ux0:.2f} m/s")
    print(f"                     -> {u_n0:.2f} m/s along n, theta = {np.degrees(theta):.1f} deg")
    print(f"  Shape            : {shape_name} (code={shape})")
    print(f"  D0 (diameter)    : {d['D0']:.4f} m")
    if is_cyl:
        print(f"  L0 (length)      : {d['L0']:.4f} m  (L0/D0 = {d['L0']/d['D0']:.2f})")
    else:
        print(f"  L0               : n/a (sphere)")
    print(f"  d_fabric         : {d['d_fabric']*1e3:.4f} mm")
    print(f"  P0 (inflation)   : {d['P0']:.1f} Pa  ({d['P0']/1e5:.3f} bar)")
    print(f"  T0 (fixed)       : {d['T0']:.1f} K")
    ref_len, ref_name = (min(d['D0'], d['L0']), "min(D0, L0)") if is_cyl else (d['D0'], "D0")
    A_hole10 = (np.pi / 4) * (0.10 * ref_len) ** 2
    d_eq = np.sqrt(4 * r['A_vent0'] / np.pi)
    print(f"  phi_vent         : {100*d['phi_vent']:.2f} % of fabric area ({r['S_surface']:.3f} m^2)")
    print(f"  vent area (t=0)  : {r['A_vent0']*1e4:.1f} cm^2 = one {d_eq*1e3:.0f} mm hole, "
          f"or {r['A_vent0']/A_hole10:.0f} holes of 10 % of {ref_name}")
    print("-"*50)
    print(f"  m_total          : {r['m_total']:.4f} kg")
    print(f"  m_bag / m_gas0   : {r['m_bag']:.4f} kg / {r['m_gas0']:.4f} kg")
    print(f"  a_peak_n (F_n/M) : {r['a_peak_n']:.2f} m/s^2 = {r['a_peak_n']/g:.2f} g "
          f"(limit {A_LIM:.1f} = {A_ALLOW:.0f}/{SF})")
    print(f"  u_residual n/x/y : {r['u_residual_n']:.3f} / {r['u_residual_x']:.3f} / "
          f"{r['u_residual']:.3f} m/s  (cap along n {U_RES_MAX}, u_n0={u_n0:.2f})")
    print(f"  x travel         : {r['x_final']:.3f} m")
    print(f"  P_peak / P_burst : {r['P_peak']:.1f} / {r['P_burst']:.1f} Pa")
    print(f"  rebounded        : {r['rebounded']}")
    print(f"  bottomed         : {r['bottomed']}")
    print(f"  survived         : {r['survived']}")
    print(f"  violations       : " + ", ".join(f"{k}={val:.3g}" for k, val in v.items()))
    print(f"  FEASIBLE         : {feasible}")
    print("="*50 + "\n")


if __name__ == "__main__":
    theta, u_n0 = impact_axis(scenario)
    print(f"QUICK_TEST = {QUICK_TEST} | A_LIM = {A_ALLOW:.0f}/{SF} = {A_LIM:.1f} m/s^2 | "
          f"U_RES_MAX = {U_RES_MAX} m/s | u_n0 = {u_n0:.2f} m/s, theta = {np.degrees(theta):.1f} deg")
    print(f"bounds: D0 in [{min_stroke(scenario, env):.3f}, {D0_MAX}] m | L0/D0 in {ASPECT_RANGE} | "
          f"d_fabric in {DFAB_RANGE} m | P0 in [P_amb, {P0_MAX:.0f}] Pa | phi_vent in {PHI_RANGE}")

    for W_land in W_LAND_VALUES:
        print(f"\n{'='*60}\nW_land = {W_land}\n{'='*60}")
        results = []
        for shape_name, shape in SHAPE_CODE.items():
            for mat_name in MATERIAL_NAMES:
                mat = materials[mat_name]
                for gas_name in GAS_NAMES:
                    gas = gases[gas_name]
                    best = None
                    for seed in SEEDS:
                        # population = POPSIZE x n_variables (Sobol rounds up to a power of 2)
                        res = differential_evolution(airbag_objective, make_bounds(shape, scenario, env),
                                                    args=(shape, mat, gas, scenario, env, W_land),
                                                    init='sobol', popsize=POPSIZE, maxiter=MAXITER, tol=1e-4,
                                                    polish=False, updating='deferred', workers=-1, rng=seed)
                        if best is None or res.fun < best.fun:
                            best = res
                    results.append((shape_name, mat_name, gas_name, best))
                    print(f"  {shape_name}/{mat_name}/{gas_name}: best f={best.fun:.4f}")

        winner = min(results, key=lambda t: t[3].fun)
        shape_name, mat_name, gas_name, res = winner
        shape = SHAPE_CODE[shape_name]
        d = unpack_design(res.x, shape)
        print(repr(d))
        r = simulate_airbag(**d, M_payload=M_payload, u0=u0, ux0=ux0,
                            sigma_fabric=materials[mat_name][0], rho_fabric=materials[mat_name][1],
                            R_gas=gases[gas_name][0], gamma=gases[gas_name][1],
                            P_amb=P_amb, g=g,
                            verbose=False, make_plots=False, a_allow=A_LIM)

        print_design_summary(shape_name, mat_name, gas_name, res.x, shape, r)
        all_winners.append((W_land, shape_name, mat_name, gas_name, r['m_total'], r['u_residual_n'], r['a_peak_n']))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'W_land':<8}{'shape':<10}{'mat':<6}{'gas':<5}{'m_total':<10}{'u_res_n':<10}{'a_peak_n(g)'}")
    for w, sn, mn, gn, m, uv, av in all_winners:
        print(f"{w:<8}{sn:<10}{mn:<6}{gn:<5}{m:<10.4f}{uv:<10.3f}{av/9.81:.1f}")
