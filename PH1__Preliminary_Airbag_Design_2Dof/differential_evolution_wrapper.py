import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from airbag_simulation_fast import simulate_airbag_fast   # numba twin, search phase only
from airbag_simulation_function import simulate_airbag    # reference, winner re-simulation
from scipy.optimize import OptimizeResult, differential_evolution


# ---------- run size ----------
# python differential_evolution_wrapper.py [quick|long] [--workers N]
RUN_SIZES = {
    # material only changes mass and burst pressure, not the dynamics -> one is enough for a quick test
    "quick": dict(U_RES_VALUES=[2.0], SEEDS=(1,), POPSIZE=2, MAXITER=6, N_MATERIALS=1),
    # MAXITER 150: in test runs the best f was within 0.1 % of its 200-generation value by gen ~135-145
    "long":  dict(U_RES_VALUES=[0.5, 1.0, 1.5, 2.0], SEEDS=(1, 2, 3, 4, 5), POPSIZE=12, MAXITER=150, N_MATERIALS=3),
}

# one row per finished DE run; rows already in the file are not run again
CSV_KEY_FIELDS = ["u_res_max", "shape", "material", "gas", "seed", "popsize", "maxiter"]
CSV_FIELDS = CSV_KEY_FIELDS + ["f", "nit", "nfev", "seconds", "D0", "L0", "n_ply", "P0", "phi_vent", "x"]


# ---------- requirements ----------
A_ALLOW   = 2206.5         # [m/s^2] allowable payload acceleration along n (225 g)
SF        = 1.2            # [-]     safety factor
A_LIM     = A_ALLOW / SF   # [m/s^2] limit on the peak bag load Fn/M
PEN       = 1.0e4          # penalty weight, must dominate mass

# burst knockdowns, applied to the fabric's per-width ultimate strength
ETA_SEAM  = 0.70           # [-] seam efficiency: the seam fails before the parent fabric
SF_BURST  = 1.50           # [-] design factor on the allowable membrane tension

# venting sub-iterations during the search only; the winner is re-simulated at full fidelity
# (simulate_airbag, 50 sub-iterations). Coarse phase (dt = 2e-4 s): 20; fine phase (dt = 1e-5 s): 5.
# Against full fidelity: max a_peak_n error ~1.7 % (2 fine sub-iterations gave ~4 %, 20 gave ~1.2 %).
MAX_INNER_ITERS_SEARCH      = 20
MAX_INNER_ITERS_SEARCH_FINE = 5
# results of a different search model go to a different CSV, so they are never mixed or reused
SEARCH_TAG = f"numba_vent{MAX_INNER_ITERS_SEARCH}-{MAX_INNER_ITERS_SEARCH_FINE}"

SHAPE_CODE = {"sphere": 2, "cylinder": 1}

all_winners = []


# ---------- impact scenario (fixed) ----------
M_payload = 1.0          # kg
uy0       = 25        # m/s   vertical impact velocity
ux0       = 7.5          # m/s   horizontal impact velocity
scenario  = (M_payload, uy0, ux0)

# ---------- environment ----------
P_amb = 101325.0         # Pa
g     = 9.80665          # m/s^2
env   = (P_amb, g)

T0 = 288.0               # K, fixed

# ---------- fabrics ----------
# Supplier data, as airbag fabric is actually sold:
#   N_ult_Ncm    [N/cm]   tensile strength per unit width, i.e. ultimate membrane force (1 N/cm = 100 N/m)
#   areal_weight [kg/m^2] areal weight (datasheets quote g/m^2, so 0.42 = 420 g/m^2)
#   t_ply        [m]      thickness of ONE ply, for the stack thickness and packing only
# Strength is per unit width, so thickness buys strength only through the ply count:
# n plies carry n times the load and weigh n times as much.
#
materials = {
    "Nylon 6,6, silicone-coated": dict(N_ult_Ncm=600.0, areal_weight=0.180, t_ply=0.30e-3),
    "Polyester (PET), PU-coated": dict(N_ult_Ncm=550.0, areal_weight=0.190, t_ply=0.28e-3),
    "Aramid (Kevlar), coated": dict(N_ult_Ncm=1000.0, areal_weight=0.260, t_ply=0.35e-3),
}

# ---------- gases: (R_gas, gamma) ----------
gases = {                # R [J/kg·K]
    "N2": (296.8, 1.400),
    "He": (2076.9, 1.667),
    "Ar": (208.1, 1.667),
    "CO2": (188.9, 1.289),
    "dry air": (287.0, 1.400),
}

GAS_NAMES = list(gases.keys())

# ---------- design-space bounds ----------
D0_MAX       = 1.5             # [m]
ASPECT_RANGE = (0.5, 4.0)      # [-]  L0 / D0, cylinder only
NPLY_RANGE   = (1, 4)          # [-]  number of fabric plies (integer variable)
P0_MAX       = 1.4e5           # [Pa] lower bound is P_amb
PHI_RANGE    = (0.0, 0.10)     # [-]  breathing-fabric open-area fraction


def impact_axis(scenario):
    """Impact angle theta [rad] and impact speed along n [m/s]."""
    _, uy0_s, ux0_s = scenario
    return np.arctan2(ux0_s, uy0_s), np.hypot(uy0_s, ux0_s)


def min_stroke(scenario, env):
    """Stroke needed to stop the payload at a constant load A_LIM: lower bound on D0."""
    theta, un0 = impact_axis(scenario)
    decel = A_LIM - env[1] * np.cos(theta)
    if decel <= 0.0:
        raise ValueError(f"A_LIM = {A_LIM:.1f} m/s^2 cannot even hold the payload against gravity")
    return un0**2 / (2.0 * decel)


def unpack_design(x, shape):
    """cylinder: x = [D0, L0/D0, n_ply, P0, phi_vent]; sphere: x = [D0, n_ply, P0, phi_vent].
    n_ply is an integer variable (see make_integrality); the round() guards against a
    caller that optimised without the integrality mask."""
    if shape == SHAPE_CODE["cylinder"]:
        D0, aspect, n_ply, P0, phi_vent = x
        L0 = aspect * D0
    else:
        D0, n_ply, P0, phi_vent = x
        L0 = D0                                      # ignored by the sphere geometry
    return dict(D0=D0, L0=L0, d_or=0.0, n_ply=int(round(n_ply)),
                P0=P0, shape=shape, T0=T0, phi_vent=phi_vent)


def make_bounds(shape, scenario, env):
    b_D0     = (min_stroke(scenario, env), D0_MAX)
    b_aspect = ASPECT_RANGE
    b_nply   = NPLY_RANGE
    b_P0     = (env[0], P0_MAX)
    b_phi    = PHI_RANGE
    return ([b_D0, b_aspect, b_nply, b_P0, b_phi] if shape == SHAPE_CODE["cylinder"]
            else [b_D0, b_nply, b_P0, b_phi])


def make_integrality(shape):
    """Boolean mask over the design vector: only the ply count is an integer."""
    return ([False, False, True, False, False] if shape == SHAPE_CODE["cylinder"]
            else [False, True, False, False])


def outcome_label(r):
    """Which stopping event ended the run. Exactly one applies."""
    if r["burst"]:
        return "burst"
    if r["bottomed"]:
        return "bottomed"
    if r["rebounded"]:
        return "rebounded"
    return "settled"


def landing_speed(r):
    """Speed along n for a successful outcome [m/s]:
    bottomed -> speed when the full stroke is reached; settled -> |speed| at t_max;
    burst / rebounded -> nan (no landing)."""
    if r["bottomed"]:
        return r["u_residual_n"]
    if r["settled"]:
        return abs(r["u_final_n"])
    return float("nan")


def evaluate_design(r, D0, scenario, u_res_max):
    """Graded violations (0 = satisfied) and feasibility, shared by objective and summary.
    Success = bottomed or settled, with peak load <= A_LIM and landing speed <= u_res_max."""
    uy0_s = scenario[1]
    _, un0 = impact_axis(scenario)
    landed = r["bottomed"] or r["settled"]
    u_land = landing_speed(r)
    viol = {
        "acc":   max(0.0, r["a_peak_n"] - A_LIM) / A_LIM,
        "burst": max(0.0, r["P_peak"] - r["P_burst"]) / r["P_burst"],
        "reb":   abs(r["u_rebound_y"]) / uy0_s if r["rebounded"] else 0.0,   # vertical rebound
        "ures":  (max(0.0, u_land - u_res_max) / un0 if landed else 0.0),
    }
    feasible = landed and r["a_peak_n"] <= A_LIM and u_land <= u_res_max
    return viol, feasible


def airbag_objective(x, shape, mat, gas, scenario, env, u_res_max):
    d = unpack_design(x, shape)

    try:
        r = simulate_airbag_fast(
            **d,
            M_payload=scenario[0], uy0=scenario[1], ux0=scenario[2],
            **mat,
            R_gas=gas[0], gamma=gas[1],
            P_amb=env[0], g=env[1],
            a_allow=A_LIM, eta_seam=ETA_SEAM, sf_burst=SF_BURST,
            max_inner_iters=MAX_INNER_ITERS_SEARCH,
            max_inner_iters_fine=MAX_INNER_ITERS_SEARCH_FINE,
        )
    except Exception as e:
        print(f"[SIM CRASH] shape={shape} x={x} → {type(e).__name__}: {e}")
        return 1.0e12

    m = r["m_total"]
    if not np.isfinite(m):
        return 1.0e12

    viol, feasible = evaluate_design(r, d["D0"], scenario, u_res_max)
    if not feasible:
        # no mass reward: an infeasible light bag must never beat a feasible one
        return 1.0e6 + PEN * sum(viol.values())

    # feasible: score is the physical mass (fabric + initial gas) [kg];
    # the residual speed enters only as the constraint u_residual_n <= u_res_max
    return m


def print_design_summary(shape_name, mat_name, gas_name, x, shape, r, u_res_max):
    d = unpack_design(x, shape)
    theta, un0 = impact_axis(scenario)
    viol, feasible = evaluate_design(r, d["D0"], scenario, u_res_max)
    is_cyl = shape == SHAPE_CODE["cylinder"]

    print("\n" + "="*50)
    print(f"OPTIMIZED DESIGN — {shape_name} / {mat_name} / {gas_name}")
    print("="*50)
    print(f"  Impact           : uy0 = {uy0:.2f} m/s, ux0 = {ux0:.2f} m/s")
    print(f"                     -> {un0:.2f} m/s along n, theta = {np.degrees(theta):.1f} deg")
    print(f"  Shape            : {shape_name} (code={shape})")
    print(f"  D0 (diameter)    : {d['D0']:.4f} m")
    if is_cyl:
        print(f"  L0 (length)      : {d['L0']:.4f} m  (L0/D0 = {d['L0']/d['D0']:.2f})")
    else:
        print(f"  L0               : n/a (sphere)")
    print(f"  fabric plies     : {r['n_ply']} x {1e3*r['d_fabric']/r['n_ply']:.3f} mm "
          f"= {r['d_fabric']*1e3:.4f} mm stack")
    print(f"  N_allow          : {r['N_allow']/100.0:.1f} N/cm of the "
          f"{r['n_ply']*materials[mat_name]['N_ult_Ncm']:.1f} N/cm ultimate "
          f"(eta_seam={ETA_SEAM}, SF_burst={SF_BURST})")
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
    print(f"  a_peak_n (Fn/M)  : {r['a_peak_n']:.2f} m/s^2 = {r['a_peak_n']/g:.2f} g "
          f"(limit {A_LIM:.1f} = {A_ALLOW:.0f}/{SF})")
    print(f"  u_residual n/x/y : {r['u_residual_n']:.3f} / {r['u_residual_x']:.3f} / "
          f"{r['u_residual_y']:.3f} m/s  (cap along n {u_res_max}, un0={un0:.2f})")
    print(f"  x travel         : {r['x_final']:.3f} m")
    print(f"  P_peak / P_burst : {r['P_peak']:.1f} / {r['P_burst']:.1f} Pa  "
          f"(unfactored burst {r['P_burst_ult']:.1f} Pa)")
    print(f"  rebounded        : {r['rebounded']}")
    print(f"  bottomed         : {r['bottomed']}")
    print(f"  settled          : {r['settled']}")
    print(f"  outcome          : {outcome_label(r)}")
    print(f"  landing speed n  : {landing_speed(r):.3f} m/s  (cap {u_res_max}; "
          f"u_final_n at end of run = {r['u_final_n']:.3f} m/s)")
    print(f"  survived         : {r['survived']}")
    print(f"  violations       : " + ", ".join(f"{k}={val:.3g}" for k, val in viol.items()))
    print(f"  FEASIBLE         : {feasible}")
    print("="*50 + "\n")


def load_finished_runs(path):
    """DE runs saved by an earlier, possibly interrupted, launch: {key: result}."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # a row with the wrong number of columns was cut short by a crash: that run is simply redone
    return {tuple(row[k] for k in CSV_KEY_FIELDS): OptimizeResult(fun=float(row["f"]),
                                                                  x=np.array(row["x"].split(), dtype=float))
            for row in rows if None not in row and None not in row.values()}


def save_run(path, row):
    """Append one finished DE run. The file stays closed in between, so it can be read during a run."""
    is_new = not os.path.exists(path)
    while True:
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:   # e.g. the file is open in Excel, which locks it on Windows
            print(f"  cannot write {path}: close it, retrying in 30 s", flush=True)
            time.sleep(30)


def n_available_cpus():
    """CPUs this job may use: the scheduler's allocation on a cluster, else the whole PC.
    os.cpu_count() alone would report the full cluster node, not the allocated share."""
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "NCPUS"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    if hasattr(os, "sched_getaffinity"):                  # Linux: honours cpusets
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def run_key(u_res_max, shape_name, mat_name, gas_name, seed, popsize, maxiter):
    """CSV key of one DE run, as strings (matches load_finished_runs)."""
    return tuple(str(v) for v in (u_res_max, shape_name, mat_name, gas_name, seed, popsize, maxiter))


def run_one_de(u_res_max, shape_name, mat_name, gas_name, seed, popsize, maxiter):
    """One full DE run, serial inside: the parallelism is across independent DE runs.
    updating='deferred' keeps the algorithm identical to the previous workers=-1 version,
    so a given seed gives the same result and the rows already in the CSV stay valid."""
    shape = SHAPE_CODE[shape_name]
    t_start = time.perf_counter()
    # population = POPSIZE x n_variables (Sobol rounds up to a power of 2)
    res = differential_evolution(airbag_objective, make_bounds(shape, scenario, env),
                                 args=(shape, materials[mat_name], gases[gas_name], scenario, env, u_res_max),
                                 integrality=make_integrality(shape),
                                 init='sobol', popsize=popsize, maxiter=maxiter, tol=1e-4,
                                 polish=False, updating='deferred', workers=1, rng=seed)
    key = run_key(u_res_max, shape_name, mat_name, gas_name, seed, popsize, maxiter)
    d_run = unpack_design(res.x, shape)
    row = dict(zip(CSV_KEY_FIELDS, key), f=repr(float(res.fun)),
               nit=res.nit, nfev=res.nfev,
               seconds=round(time.perf_counter() - t_start, 1),
               D0=d_run["D0"], L0=d_run["L0"], n_ply=d_run["n_ply"],
               P0=d_run["P0"], phi_vent=d_run["phi_vent"],
               x=" ".join(repr(float(v)) for v in res.x))
    return key, row, OptimizeResult(fun=float(res.fun), x=np.asarray(res.x, dtype=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_size", nargs="?", default="quick", choices=RUN_SIZES)
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel DE runs (default: all CPUs allocated to the job)")
    cli = parser.parse_args()
    run_size = cli.run_size
    U_RES_VALUES = RUN_SIZES[run_size]["U_RES_VALUES"]
    SEEDS        = RUN_SIZES[run_size]["SEEDS"]
    POPSIZE      = RUN_SIZES[run_size]["POPSIZE"]
    MAXITER      = RUN_SIZES[run_size]["MAXITER"]
    MATERIAL_NAMES = list(materials.keys())[:RUN_SIZES[run_size]["N_MATERIALS"]]

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"de_results_{run_size}_{SEARCH_TAG}.csv")
    finished = load_finished_runs(results_path)

    # compile the numba kernel once here, so the workers only load it from the cache
    # (instead of every worker compiling it at the same time)
    _b = make_bounds(SHAPE_CODE["sphere"], scenario, env)
    airbag_objective(np.array([0.5 * (lo + hi) for lo, hi in _b]), SHAPE_CODE["sphere"],
                     materials[MATERIAL_NAMES[0]], gases[GAS_NAMES[0]], scenario, env, U_RES_VALUES[0])

    theta, un0 = impact_axis(scenario)
    print(f"run size = {run_size} | seeds = {SEEDS} | POPSIZE = {POPSIZE} | MAXITER = {MAXITER} | "
          f"search model = {SEARCH_TAG}")
    print(f"results file = {results_path} ({len(finished)} DE runs already saved)")
    print(f"A_LIM = {A_ALLOW:.1f}/{SF} = {A_LIM:.1f} m/s^2 | "
          f"u_res_max sweep = {U_RES_VALUES} m/s | un0 = {un0:.2f} m/s, theta = {np.degrees(theta):.1f} deg")
    print(f"bounds: D0 in [{min_stroke(scenario, env):.3f}, {D0_MAX}] m | L0/D0 in {ASPECT_RANGE} | "
          f"n_ply in {NPLY_RANGE} | P0 in [P_amb, {P0_MAX:.0f}] Pa | phi_vent in {PHI_RANGE}")
    print(f"burst knockdown: eta_seam = {ETA_SEAM} / SF_burst = {SF_BURST} "
          f"-> {100*ETA_SEAM/SF_BURST:.0f} % of the fabric's ultimate N/cm")

    # ---------- phase 1: every missing DE run, in parallel ----------
    # The runs are independent, so they are farmed out whole to a process pool (one DE per
    # core, no per-generation barrier). The main process alone writes the CSV, as each run
    # finishes, so an interrupted job keeps every run that has completed.
    combos = [(u, sn, mn, gn, seed) for u in U_RES_VALUES for sn in SHAPE_CODE
              for mn in MATERIAL_NAMES for gn in GAS_NAMES for seed in SEEDS]
    todo = [c for c in combos if run_key(*c, POPSIZE, MAXITER) not in finished]
    todo.sort(key=lambda c: c[1] != "cylinder")   # slower cylinder runs first: shorter tail at the end
    n_workers = max(1, min(cli.workers or n_available_cpus(), len(todo)))
    print(f"{len(todo)} of {len(combos)} DE runs to do, on {n_workers} parallel workers", flush=True)

    if todo:
        t_phase = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(run_one_de, *c, POPSIZE, MAXITER): c for c in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                u, sn, mn, gn, seed = futures[fut]
                try:
                    key, row, res = fut.result()
                except Exception as e:   # this run is simply redone at the next launch
                    print(f"  [{i}/{len(todo)}] FAILED u_res={u} {sn}/{mn}/{gn} seed={seed}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                save_run(results_path, row)
                finished[key] = res
                print(f"  [{i}/{len(todo)}] u_res={u} {sn}/{mn}/{gn} seed={seed}: f={res.fun:.4f} "
                      f"({row['seconds']} s, nfev={row['nfev']}) | elapsed "
                      f"{(time.perf_counter() - t_phase)/60:.1f} min", flush=True)

    # ---------- phase 2: best seed per combination, winner per u_res_max ----------
    for u_res_max in U_RES_VALUES:
        print(f"\n{'='*60}\nu_res_max = {u_res_max} m/s\n{'='*60}")
        results = []
        for shape_name in SHAPE_CODE:
            for mat_name in MATERIAL_NAMES:
                for gas_name in GAS_NAMES:
                    runs = [finished[k] for seed in SEEDS
                            if (k := run_key(u_res_max, shape_name, mat_name, gas_name, seed, POPSIZE, MAXITER))
                            in finished]
                    if len(runs) < len(SEEDS):
                        print(f"  {shape_name}/{mat_name}/{gas_name}: {len(SEEDS) - len(runs)} seed(s) missing")
                    if not runs:
                        continue
                    best = min(runs, key=lambda res: res.fun)
                    results.append((shape_name, mat_name, gas_name, best))
                    print(f"  {shape_name}/{mat_name}/{gas_name}: best f={best.fun:.4f}")

        if not results:
            print("  no finished DE run for this u_res_max: relaunch to complete it")
            continue
        winner = min(results, key=lambda t: t[3].fun)
        shape_name, mat_name, gas_name, res = winner
        shape = SHAPE_CODE[shape_name]
        d = unpack_design(res.x, shape)
        print(repr(d))
        r = simulate_airbag(**d, M_payload=M_payload, uy0=uy0, ux0=ux0,
                            **materials[mat_name],
                            R_gas=gases[gas_name][0], gamma=gases[gas_name][1],
                            P_amb=P_amb, g=g, eta_seam=ETA_SEAM, sf_burst=SF_BURST,
                            verbose=False, make_plots=False, a_allow=A_LIM)

        print_design_summary(shape_name, mat_name, gas_name, res.x, shape, r, u_res_max)
        _, feas_w = evaluate_design(r, d["D0"], scenario, u_res_max)
        all_winners.append((u_res_max, shape_name, mat_name, gas_name, outcome_label(r),
                            r['m_total'], landing_speed(r), r['a_peak_n'], feas_w))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'u_res_max':<10}{'shape':<10}{'mat':<6}{'gas':<9}{'outcome':<11}"
          f"{'m_total':<10}{'u_land':<10}{'a_peak_n(g)':<13}{'feasible'}")
    for w, sn, mn, gn, oc, m, uv, av, fe in all_winners:
        print(f"{w:<10.2f}{sn:<10}{mn:<6}{gn:<9}{oc:<11}"
              f"{m:<10.4f}{uv:<10.3f}{av/g:<13.1f}{fe}")
