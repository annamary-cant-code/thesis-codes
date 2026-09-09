import numpy as np
from airbag_simulation_function_rev02 import simulate_airbag
from scipy.optimize import differential_evolution


A_ALLOW = 500     #  allowable acceleration [m/s^2]  -- a requirement
SF_ACC  = 0.95      # safety margin: aim 5% under the limit so the winner survives validation
PEN     = 1.0e4     # any violation must dominate mass

SHAPE_CODE = {"sphere": 2, "cylinder": 1}
W_LAND_VALUES = [10, 50, 100, 200]

all_winners = []


# ---------- fixed requirements / impact scenario (NOT optimized) ----------
M_payload = 0.5        # kg
u0        = 17.2         # m/s   vertical impact velocity
scenario  = (M_payload, u0)     # order matters

# ---------- environment ----------
P_amb = 101325.0         # Pa
g     = 9.81             # m/s^2
env   = (P_amb, g)

# ---------- material options: (sigma_fabric, rho_fabric) ----------
materials = {
    "matA": (400e6, 1400.0),
    "matB": (250e6, 1100.0),
}

# ---------- gas options: (R_gas, gamma) ----------
gases = {                # R [J/kg·K];
    "N2": (296.8, 1.40),
    "He": (2077.0, 1.66),
    "Ar": (208.1, 1.67),
}

      
def airbag_objective(x, shape, mat, gas, scenario, env, W_land):
    if shape == SHAPE_CODE["cylinder"]:                                   # cylinder -> 7
        D0, L0, d_fabric, P0, T0, dor_frac = x
    else:                                            # sphere -> 6
        D0, d_fabric, P0, T0, dor_frac = x
        L0 = D0                                      # ignored by sphere geometry
    d_or = dor_frac * D0

    try:
        r = simulate_airbag(
        D0=D0, L0=L0, d_or=d_or, d_fabric=d_fabric, P0=P0, shape=shape,
        M_payload=scenario[0], u0=scenario[1],
        sigma_fabric=mat[0], rho_fabric=mat[1],
        T0=T0, R_gas=gas[0], gamma=gas[1],
        P_amb=env[0], g=env[1], a_allow=A_ALLOW
    )
    except Exception as e:
        print(f"[SIM CRASH] shape={shape} x={x} → {type(e).__name__}: {e}")
        return 1.0e12                                # un-simulable -> huge finite cost

    m = r["m_total"]
    if not np.isfinite(m):
        return 1.0e12


    # --- hard-gate violations, each graded so DE can climb out ---
    v_burst  = max(0.0, r["P_peak"] - r["P_burst"]) / r["P_burst"]
    v_acc    = max(0.0, r["a_peak"] - SF_ACC * A_ALLOW) / A_ALLOW   # made absolute value in the simulate_airbag function already
    v_reb    = abs(r["u_rebound"]) / u0 if r["rebounded"] else 0.0
    v_nobot  = 0.0 if r["bottomed"] else 1.0   # didn't reach D0 at all (timed out floating)

    infeasible = (r["rebounded"] or not r["bottomed"]
                  or r["a_peak"] > A_ALLOW or r["P_peak"] > r["P_burst"])

    if infeasible:
        # NO mass reward here — a light rebounding bag must never beat a valid one..
        return 1.0e6 + PEN * (v_burst + v_acc + v_reb + v_nobot)

    # W_land = 100 means 1 kg of extra bag means a full u0-to-zero improvement in landing speed
    return m + W_land * (r["u_residual"] / u0)

def make_bounds(shape, P_amb):
    b_D0       = (0.1, 0.5)
    b_L0       = (0.1, 0.6)
    b_dfab     = (0.0005, 0.003) # [m]
    b_P0       = (P_amb, 1.4e5)
    b_T0       = (273.15, 320)
    b_dor_frac = (0.01, 0.2)      # fraction of D0
    return ([b_D0, b_L0, b_dfab, b_P0, b_T0, b_dor_frac] if shape == SHAPE_CODE["cylinder"]
            else [b_D0, b_dfab, b_P0, b_T0, b_dor_frac])

def unpack_design(x, shape):
    if shape == SHAPE_CODE["cylinder"]:
        D0, L0, d_fabric, P0, T0, dor_frac = x
    else:
        D0, d_fabric, P0, T0, dor_frac = x
        L0 = D0 
    d_or = dor_frac * D0              
    return dict(D0=D0, L0=L0, d_or=d_or, d_fabric=d_fabric,
                P0=P0, shape=shape, T0=T0)

def print_design_summary(shape_name, mat_name, gas_name, x, shape, r):
    d = unpack_design(x, shape)

    print("\n" + "="*50)
    print(f"OPTIMIZED DESIGN — {shape_name} / {mat_name} / {gas_name}")
    print("="*50)
    print(f"  Shape            : {shape_name} (code={shape})")
    print(f"  D0 (diameter)    : {d['D0']:.4f} m")
    if shape == SHAPE_CODE["cylinder"]:
        print(f"  L0 (length)      : {d['L0']:.4f} m")
    else:
        print(f"  L0               : n/a (sphere)")
    print(f"  d_fabric         : {d['d_fabric']*1e3:.4f} mm")
    print(f"  P0 (inflation)   : {d['P0']:.1f} Pa  ({d['P0']/1e5:.3f} bar)")
    print(f"  T0               : {d['T0']:.1f} K")
    print(f"  d_or (orifice)   : {d['d_or']*1e3:.4f} mm")
    print("-"*50)
    print(f"  m_total          : {r['m_total']:.4f} kg")
    print(f"  m_bag / m_gas0   : {r['m_bag']:.4f} kg / {r['m_gas0']:.4f} kg")
    print(f"  a_peak           : {r['a_peak']:.2f} m/s^2 {r['a_peak']/9.81:.2f} g (limit {A_ALLOW})")
    print(f"  u_residual       : {r['u_residual']:.3f} m/s  (u0={u0})")
    print(f"  P_peak / P_burst : {r['P_peak']:.1f} / {r['P_burst']:.1f} Pa")
    print(f"  rebounded        : {r['rebounded']}")
    print(f"  bottomed         : {r['bottomed']}")
    print(f"  survived         : {r['survived']}")

    is_feasible = (r["bottomed"] and not r["rebounded"]
                   and r["a_peak"] <= A_ALLOW and r["P_peak"] <= r["P_burst"])
    print(f"  FEASIBLE         : {is_feasible}")
    print("="*50 + "\n")



if __name__ == "__main__":
    for W_land in W_LAND_VALUES:
        print(f"\n{'='*60}\nW_land = {W_land}\n{'='*60}")
        results = []
        for shape_name, shape in SHAPE_CODE.items():
            for mat_name, mat in materials.items():          # 2 entries
                for gas_name, gas in gases.items():          # 3 entries
                    best = None
                    for seed in (1, 2, 3):
                        res = differential_evolution(airbag_objective, make_bounds(shape, P_amb),
                                                    args=(shape, mat, gas, scenario, env, W_land),
                                                    init='sobol', popsize=5, maxiter=20, tol=1e-4,
                                                    polish=False, updating='deferred', workers=-1, rng=seed)
                        # N arguments of the airbag_objective function to optimise. & for cylinder, 5 for sphere
                        # popsize = how many candidates in each generation are picked within the established bounds. popsize x N = comparisons to make, one per population member
                        # maxiter = how many CHALLENGING GENERATIONS are generated, each with popsize number of candidates. 
                        # total number of comparisons = maxiter * N * popsize
                        if best is None or res.fun < best.fun:
                            best = res
                    results.append((shape_name, mat_name, gas_name, best))
                    print(f"  {shape_name}/{mat_name}/{gas_name}: best f={best.fun:.4f}")

        winner = min(results, key=lambda t: t[3].fun)
        shape_name, mat_name, gas_name, res = winner
        shape = SHAPE_CODE[shape_name]
        d = unpack_design(res.x, shape)
        print(repr(d))
        r = simulate_airbag(**d, M_payload=M_payload, u0=u0,
                            sigma_fabric=materials[mat_name][0], rho_fabric=materials[mat_name][1],
                            R_gas=gases[gas_name][0], gamma=gases[gas_name][1], 
                            P_amb=P_amb, g=g,
                            verbose=False, make_plots=False, a_allow=A_ALLOW)
        
        print_design_summary(shape_name, mat_name, gas_name, res.x, shape, r)
        all_winners.append((W_land, shape_name, mat_name, gas_name, r['m_total'], r['u_residual'], r['a_peak']))


    # summary table — printed ONCE, after all W_land values are done
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'W_land':<8}{'shape':<10}{'mat':<6}{'gas':<5}{'m_total':<10}{'u_res':<10}{'a_peak(g)'}")
    for w, sn, mn, gn, m, uv, av in all_winners:
        print(f"{w:<8}{sn:<10}{mn:<6}{gn:<5}{m:<10.4f}{uv:<10.3f}{av/9.81:.1f}")