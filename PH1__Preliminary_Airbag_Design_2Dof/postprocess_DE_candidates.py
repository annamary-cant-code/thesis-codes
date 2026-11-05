"""Post-processing of the long DE run (de_results_long_<SEARCH_TAG>.csv).

The DE search used the numba twin with reduced venting sub-iterations; feasibility near the
constraints is therefore re-checked here with the reference model (simulate_airbag).

    1. full-fidelity re-simulation of EVERY DE run (all seeds) -> DE_outputs/DE_candidates_fullfid.csv
       + re-ranked winners per u_res_max                     -> DE_outputs/DE_winners_fullfid.csv
    2. one 4x4 plot matrix per candidate (best seed per u_res/shape/material/gas)
                                                             -> DE_outputs/matrices/ures_<u>/...png
    3. overlay matrices: full-fidelity winner per u_res_max (one figure),
       and lightest feasible design per gas at each u_res_max -> DE_outputs/overlays/
       each as *_impact (impact transient) and *_full (whole run)
    4. design-space summary plots                            -> DE_outputs/summary/

    python postprocess_DE_candidates.py [--workers N] [--no-matrices] [--force] [--t-cut T]
                                        [--a-tol-pct TOL]

Full-fidelity feasibility allows a_peak_n up to A_LIM (1 + TOL/100), see A_TOL_PCT; the strict
verdict is kept in the table as feasible_strict.

The full-fidelity table and the per-candidate matrices are cached: rerunning only redoes what is
missing (or everything with --force). Overlays and summary plots are always redrawn.
"""
import argparse
import contextlib
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")                         # files only, also inside the worker processes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import differential_evolution_wrapper as W
from airbag_postprocessing import (N_COLS, N_ROWS, PLOT_MATRIX_DPI, _plot_specs,
                                   replay_histories, run_postprocessing)
from airbag_simulation_function import simulate_airbag

HERE        = os.path.dirname(os.path.abspath(__file__))
DE_CSV      = os.path.join(HERE, f"de_results_long_{W.SEARCH_TAG}.csv")
OUT_DIR     = os.path.join(HERE, "DE_outputs")
FULLFID_CSV = os.path.join(OUT_DIR, "DE_candidates_fullfid.csv")
WINNERS_CSV = os.path.join(OUT_DIR, "DE_winners_fullfid.csv")
MATRIX_DIR  = os.path.join(OUT_DIR, "matrices")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")
FIG_FORMATS = ("png", "pdf")

# simulate_airbag defaults, as used by the DE wrapper's winner re-simulation;
# replay_histories needs them explicitly and must match
DT, T_MAX = 1e-4, 1.0

# Tolerance on a_peak_n <= A_LIM in the full-fidelity check. The DE drives a_peak_n onto the
# limit, so the reference model lands a hair either side of it (observed overshoot < 0.04 %
# of A_LIM, i.e. < 0.1 g against a 225 g allowable with SF 1.2 already applied). A strict check
# turns that model noise into a random feasible/infeasible label. feasible_strict keeps the
# strict verdict in the table; --a-tol-pct 0 makes the strict verdict the ranking criterion.
A_TOL_PCT = 0.1                                       # [% of A_LIM]

KEY = ["u_res_max", "shape", "material", "gas"]       # one DE "combination"; seeds within it
F_FEASIBLE = 1.0e5                                    # search objective: feasible -> mass, else >= 1e6

MAT_SHORT = {"Nylon 6,6, silicone-coated": "Nylon",
             "Polyester (PET), PU-coated": "PET",
             "Aramid (Kevlar), coated": "Aramid"}
MAT_MARKER = {"Nylon": "o", "PET": "s", "Aramid": "^"}

# reference palette (dataviz skill, references/palette.md), light mode
# gases are categorical identities: fixed slot order, never cycled
GAS_COLOR = {"N2": "#2a78d6", "He": "#eb6834", "Ar": "#1baf7a", "CO2": "#eda100", "dry air": "#e87ba4"}
GAS_LS    = {"N2": "-", "He": "-", "Ar": "--", "CO2": "-.", "dry air": ":"}
# speed caps are ordinal: one blue ramp, step 250 (u_res_max = 0.5) -> 700 (2.0)
CAP_RAMP  = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CAP_LS    = ["-", "--", "-.", ":"]
LIMIT_COLOR, MUTED, GRID = "#e34948", "#9a9993", "#e6e5e0"


# ---------------------------------------------------------------- helpers
def build_inputs(row):
    """simulate_airbag / replay_histories kwargs for one DE row."""
    shape = W.SHAPE_CODE[row["shape"]]
    d = W.unpack_design(np.array(str(row["x"]).split(), dtype=float), shape)
    R_gas, gamma = W.gases[row["gas"]]
    return dict(**d, M_payload=W.M_payload, uy0=W.uy0, ux0=W.ux0, **W.materials[row["material"]],
                R_gas=R_gas, gamma=gamma, P_amb=W.P_amb, g=W.g,
                eta_seam=W.ETA_SEAM, sf_burst=W.SF_BURST, a_allow=W.A_LIM, dt=DT, t_max=T_MAX)


def fname(row):
    return (f"{row['shape']}_{MAT_SHORT[row['material']]}_{row['gas'].replace(' ', '')}"
            f"_seed{int(row['seed'])}")


def ures_dir(u):
    return f"ures_{float(u):.1f}"


def save(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in FIG_FORMATS:
        fig.savefig(f"{out_base}.{ext}", dpi=PLOT_MATRIX_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(out_base, HERE)}.{{{','.join(FIG_FORMATS)}}}")


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- 1. full-fidelity table
def fullfid_job(row):
    """Reference re-simulation of one DE run: margins, outcome, full-fidelity feasibility."""
    inputs = build_inputs(row)
    u_cap = float(row["u_res_max"])
    out = {k: row[k] for k in KEY + ["seed"]}
    out.update(f_search=float(row["f"]), feasible_search=bool(float(row["f"]) < F_FEASIBLE),
               D0=inputs["D0"], L0=inputs["L0"],
               aspect=inputs["L0"] / inputs["D0"] if row["shape"] == "cylinder" else np.nan,
               n_ply=inputs["n_ply"], P0=inputs["P0"], phi_vent=inputs["phi_vent"])
    try:
        r = simulate_airbag(**inputs, verbose=False, make_plots=False)
    except Exception as e:
        out.update(outcome=f"crash: {type(e).__name__}", feasible_strict=False, x=row["x"])
        return out
    viol, feasible = W.evaluate_design(r, inputs["D0"], W.scenario, u_cap)
    u_land = W.landing_speed(r)
    out.update(
        m_total=r["m_total"], m_bag=r["m_bag"], m_gas0=r["m_gas0"],
        a_peak_n=r["a_peak_n"], a_peak_n_g=r["a_peak_n"] / W.g,
        a_margin_pct=100 * (W.A_LIM - r["a_peak_n"]) / W.A_LIM,          # > 0: below the limit
        u_land=u_land, u_margin=u_cap - u_land,                           # > 0: below the cap
        P_peak=r["P_peak"], P_burst=r["P_burst"],
        burst_margin_pct=100 * (r["P_burst"] - r["P_peak"]) / r["P_burst"],
        outcome=W.outcome_label(r), t_stop=r["t_stop"], x_final=r["x_final"],
        A_vent0_cm2=1e4 * r["A_vent0"],
        **{f"viol_{k}": v for k, v in viol.items()},
        feasible_strict=bool(feasible), x=row["x"])
    return out


def apply_tolerance(ff, a_tol_pct):
    """feasible_fullfid: landed, a_peak_n <= A_LIM (1 + a_tol_pct/100), u_land <= u_res_max."""
    ff = ff.copy()
    ff["feasible_fullfid"] = (ff["outcome"].isin(["bottomed", "settled"])
                              & (ff["a_margin_pct"] >= -a_tol_pct) & (ff["u_margin"] >= 0))
    return mark_candidates(ff)


def mark_candidates(ff):
    """candidate: one seed per combination = lightest full-fidelity feasible seed, else the
    seed with the lowest search objective. winner: lightest feasible candidate per u_res_max.
    de_winner: what the DE wrapper picked (lowest search objective per u_res_max)."""
    ff = ff.copy()
    ff["rank_key"] = np.where(ff["feasible_fullfid"], ff["m_total"], np.inf)
    ff = ff.sort_values(KEY + ["rank_key", "f_search"])
    ff["candidate"] = ~ff.duplicated(KEY, keep="first")
    ff["winner"] = False
    ff["de_winner"] = False
    for u, grp in ff.groupby("u_res_max"):
        feas = grp[grp["candidate"] & grp["feasible_fullfid"]]
        if len(feas):
            ff.loc[feas["m_total"].idxmin(), "winner"] = True
        ff.loc[grp["f_search"].idxmin(), "de_winner"] = True
    return ff.drop(columns="rank_key").sort_values(KEY + ["seed"]).reset_index(drop=True)


def build_fullfid_table(de, workers, force, a_tol_pct):
    if os.path.exists(FULLFID_CSV) and not force:
        ff = pd.read_csv(FULLFID_CSV)
        if len(ff) == len(de) and "feasible_strict" in ff:
            print(f"[1] full-fidelity table cached ({len(ff)} rows): {os.path.relpath(FULLFID_CSV, HERE)}")
            ff = apply_tolerance(ff, a_tol_pct)        # the tolerance may differ from the last run
            ff.to_csv(FULLFID_CSV, index=False)
            return ff
    print(f"[1] full-fidelity re-simulation of {len(de)} DE runs on {workers} workers ...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fullfid_job, de.to_dict("records"), chunksize=4))
    ff = apply_tolerance(pd.DataFrame(rows), a_tol_pct)
    os.makedirs(OUT_DIR, exist_ok=True)
    ff.to_csv(FULLFID_CSV, index=False)
    print(f"    done in {time.perf_counter() - t0:.0f} s -> {os.path.relpath(FULLFID_CSV, HERE)}")
    return ff


def report_table(ff, a_tol_pct):
    for col, tag in (("feasible_strict", "strict"),
                     ("feasible_fullfid", f"a_peak_n tolerance {a_tol_pct} %")):
        print(f"\n    search-model vs full-fidelity feasibility, all seeds ({tag}):")
        print(pd.crosstab([ff["shape"]], [ff["feasible_search"], ff[col]],
                          rownames=["shape"], colnames=["search", "fullfid"]).to_string())
    s = ff[ff["feasible_search"]]
    print(f"    a_margin_pct of the search-feasible runs: min {s['a_margin_pct'].min():.4f} %, "
          f"median {s['a_margin_pct'].median():.4f} %, max {s['a_margin_pct'].max():.4f} %")
    flips = ff[ff["feasible_search"] & ~ff["feasible_fullfid"]]
    if len(flips):
        cause = flips[["viol_acc", "viol_burst", "viol_reb", "viol_ures"]].gt(0).sum()
        print(f"    {len(flips)} runs feasible in the search but not at full fidelity; "
              f"violated: " + ", ".join(f"{k[5:]}={v}" for k, v in cause.items()))

    cols = ["u_res_max", "shape", "material", "gas", "seed", "outcome", "m_total", "a_peak_n_g",
            "a_margin_pct", "u_land", "u_margin", "burst_margin_pct", "feasible_strict",
            "feasible_fullfid"]
    rows = []
    for u, grp in ff.groupby("u_res_max"):
        for tag in ("de_winner", "winner"):
            sel = grp[grp[tag]]
            if len(sel):
                rows.append(sel.iloc[0][cols].to_dict() | {"pick": tag})
    win = pd.DataFrame(rows)[["pick"] + cols]
    win.to_csv(WINNERS_CSV, index=False)
    print(f"\n    winners per u_res_max (DE pick vs full-fidelity re-rank) -> "
          f"{os.path.relpath(WINNERS_CSV, HERE)}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(win.assign(material=win["material"].map(MAT_SHORT)).round(4).to_string(index=False))


# ---------------------------------------------------------------- 2. per-candidate matrices
def matrix_job(row):
    out_dir = os.path.join(MATRIX_DIR, ures_dir(row["u_res_max"]))
    base = fname(row)
    inputs = build_inputs(row)
    r = simulate_airbag(**inputs, verbose=False, make_plots=False)
    feasible = bool(row["feasible_fullfid"])
    geo = (f"D0 = {inputs['D0']:.3f} m, L0/D0 = {row['aspect']:.2f}" if row["shape"] == "cylinder"
           else f"D0 = {inputs['D0']:.3f} m")
    title = (f"u_res_max = {row['u_res_max']} m/s  |  {row['shape']} / {MAT_SHORT[row['material']]} / "
             f"{row['gas']} / seed {int(row['seed'])}  |  {W.outcome_label(r)}, "
             f"{'FEASIBLE' if feasible else 'INFEASIBLE'}"
             f"{' (within a_peak_n tolerance)' if feasible and not row['feasible_strict'] else ''}\n"
             f"{geo}, n_ply = {inputs['n_ply']}, P0 = {inputs['P0'] / 1e5:.3f} bar, "
             f"phi_vent = {100 * inputs['phi_vent']:.2f} %  |  m = {r['m_total']:.4f} kg, "
             f"a_peak_n = {r['a_peak_n'] / W.g:.1f} g, u_land = {W.landing_speed(r):.3f} m/s")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        run_postprocessing(inputs, r, feasible, show_popups=False, save_plot_matrix=True,
                           plot_matrix_formats=("png",), make_animation=False,
                           output_dir=out_dir, basename=base, suptitle=title)
    plt.close("all")
    warn = [l for l in log.getvalue().splitlines() if "WARNING" in l]
    return base, warn


def build_matrices(ff, workers, force):
    cands = ff[ff["candidate"]]
    wanted = {os.path.join(MATRIX_DIR, ures_dir(r["u_res_max"]), fname(r) + ".png")
              for r in cands.to_dict("records")}
    if os.path.isdir(MATRIX_DIR):   # matrices of seeds that are no longer the candidate
        for root, _, files in os.walk(MATRIX_DIR):
            for f in files:
                if f.endswith(".png") and os.path.join(root, f) not in wanted:
                    os.remove(os.path.join(root, f))
    todo = [row for row in cands.to_dict("records")
            if force or not os.path.exists(os.path.join(MATRIX_DIR, ures_dir(row["u_res_max"]),
                                                        fname(row) + ".png"))]
    print(f"\n[2] per-candidate plot matrices: {len(todo)} of {len(cands)} to draw", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for base, warn in pool.map(matrix_job, todo, chunksize=2):
            for w in warn:
                print(f"    {base}: {w}")
    print(f"    done in {time.perf_counter() - t0:.0f} s -> {os.path.relpath(MATRIX_DIR, HERE)}")


# ---------------------------------------------------------------- 3. overlay matrices
def impact_end(rep, a_lim):
    """End of the impact transient: last time the bag load exceeds 5 % of A_LIM.
    After it the payload sinks slowly while the bag vents (until it bottoms or t_max)."""
    load = rep["Fn_hist"] / W.M_payload
    above = np.flatnonzero(np.nan_to_num(load) > 0.05 * a_lim)
    return rep["t_hist"][above[-1]] if len(above) else rep["t_hist"][-1]


def overlay_matrix(cases, title, out_base, t_cut_impact=None):
    """cases: list of dict(row, label, color, ls). One 4x4 matrix, one curve per case, drawn
    twice: <out_base>_impact (time axis cut after the impact transient; t_cut_impact [s], default
    1.5 x the latest transient end) and <out_base>_full (whole run)."""
    reps = []
    for c in cases:
        inputs = build_inputs(c["row"])
        reps.append((inputs, replay_histories(**inputs)))
    if t_cut_impact is None:
        t_cut_impact = 1.5 * max(impact_end(rep, W.A_LIM) for _, rep in reps)
    _overlay_figure(cases, reps, f"{title}  —  impact transient", out_base + "_impact", t_cut_impact)
    _overlay_figure(cases, reps, f"{title}  —  full run", out_base + "_full", 0)


def _overlay_figure(cases, reps, title, out_base, t_cut):
    fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(16, 9.5), sharex=True, layout="constrained")
    axes = list(axs.flat)
    for c, (inputs, rep) in zip(cases, reps):
        t = rep["t_hist"]
        keep = t <= t_cut if t_cut > 0 else np.ones_like(t, dtype=bool)
        specs = _plot_specs(rep, inputs["M_payload"])
        for ax, (data, _, _) in zip(axes, specs):
            ax.plot(t[keep], np.asarray(data)[keep], color=c["color"], ls=c["ls"], lw=1.6, label=c["label"])
        axes[[s[2] for s in specs].index("Displacement Along n")].axhline(
            inputs["D0"], color=c["color"], ls=":", lw=1.0)

    specs = _plot_specs(reps[0][1], reps[0][0]["M_payload"])
    for ax, (_, ylabel, ttl) in zip(axes, specs):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.tick_params(labelsize=8)
        style_axes(ax)
    by_title = {s[2]: (ax, np.asarray(s[0])) for ax, s in zip(axes, specs)}
    for ttl in ("Horizontal Acceleration", "Vertical Acceleration", "Bag Load Along n (F_n/M)"):
        ax, data = by_title[ttl]
        neg = (not ttl.startswith("Bag Load")) and data[np.nanargmax(np.abs(data))] <= 0
        lim = -W.A_LIM if neg else W.A_LIM
        ax.axhline(lim, color=LIMIT_COLOR, ls="--", lw=1)
        lo, hi = ax.get_ylim()                                    # keep the limit in view
        ax.set_ylim(min(lo, 1.06 * lim), max(hi, 1.06 * lim))
        ax.annotate(f"{'-' if neg else ''}A_LIM", xy=(1, lim), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=8, color="#52514e")
    by_title["Displacement Along n"][0].annotate("dotted: D0 of each design", xy=(0.98, 0.04),
                                                 xycoords="axes fraction", ha="right",
                                                 fontsize=8, color="#52514e")
    by_title["Internal Pressure"][0].ticklabel_format(axis="y", style="sci", scilimits=(5, 5))
    for ax in axes[-N_COLS:]:
        ax.set_xlabel("Time [s]", fontsize=9)
    if t_cut > 0:
        axes[0].set_xlim(0, t_cut)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(cases), 3), fontsize=9,
               frameon=False)
    fig.suptitle(title, fontsize=13)
    save(fig, out_base)


def build_overlays(ff, t_cut):
    print("\n[3] overlay matrices")
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    caps = sorted(ff["u_res_max"].unique())

    # (a) full-fidelity winner per speed cap
    cases = []
    for i, u in enumerate(caps):
        w = ff[(ff["u_res_max"] == u) & ff["winner"]]
        if not len(w):
            print(f"    u_res_max = {u}: no full-fidelity feasible design, left out")
            continue
        row = w.iloc[0]
        cases.append(dict(row=row, color=CAP_RAMP[i % 4], ls=CAP_LS[i % 4],
                          label=f"u_res_max = {u} m/s: {row['shape']} / {MAT_SHORT[row['material']]} / "
                                f"{row['gas']}, m = {row['m_total']:.3f} kg"))
    if cases:
        overlay_matrix(cases, "Full-fidelity optimum per residual-speed cap",
                       os.path.join(OVERLAY_DIR, "overlay_winners_per_ures"), t_cut)

    # (b) per gas, at each cap: lightest feasible design of that gas (any shape / material)
    for u in caps:
        cases, missing = [], []
        for gas in W.GAS_NAMES:
            sel = ff[(ff["u_res_max"] == u) & (ff["gas"] == gas) & ff["candidate"] & ff["feasible_fullfid"]]
            if not len(sel):
                missing.append(gas)
                continue
            row = sel.loc[sel["m_total"].idxmin()]
            cases.append(dict(row=row, color=GAS_COLOR[gas], ls=GAS_LS[gas],
                              label=f"{gas}: {row['shape']} / {MAT_SHORT[row['material']]}, "
                                    f"m = {row['m_total']:.3f} kg"))
        note = f"  (no feasible design: {', '.join(missing)})" if missing else ""
        if cases:
            overlay_matrix(cases, f"Lightest full-fidelity feasible design per gas, u_res_max = {u} m/s{note}",
                           os.path.join(OVERLAY_DIR, f"overlay_gases_{ures_dir(u)}"), t_cut)


# ---------------------------------------------------------------- 4. design-space summary
def best_feasible(ff):
    """Lightest full-fidelity feasible candidate per (u_res_max, material, gas), any shape."""
    feas = ff[ff["candidate"] & ff["feasible_fullfid"]].copy()
    feas["mat"] = feas["material"].map(MAT_SHORT)
    return feas.loc[feas.groupby(["u_res_max", "mat", "gas"])["m_total"].idxmin()]


def plot_mass_vs_ures(ff):
    best = best_feasible(ff)
    mats = list(MAT_SHORT.values())
    fig, axs = plt.subplots(1, len(mats), figsize=(13, 4.2), sharey=True, layout="constrained")
    for ax, mat in zip(axs, mats):
        for gas in W.GAS_NAMES:
            s = best[(best["mat"] == mat) & (best["gas"] == gas)].sort_values("u_res_max")
            if not len(s):
                continue
            ax.plot(s["u_res_max"], s["m_total"], color=GAS_COLOR[gas], ls=GAS_LS[gas], lw=2,
                    marker=MAT_MARKER[mat], ms=8, mec="white", mew=1.5, label=gas)
            sph = s[s["shape"] == "sphere"]
            ax.plot(sph["u_res_max"], sph["m_total"], ls="none", marker=MAT_MARKER[mat], ms=11,
                    mfc="none", mec="#0b0b0b", mew=1)
        ax.set_title(mat, fontsize=11)
        ax.set_xlabel("u_res_max [m/s]")
        ax.set_xticks(sorted(ff["u_res_max"].unique()))
        style_axes(ax)
    axs[0].set_ylabel("lightest feasible m_total [kg]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, title="gas")
    ring = "  (black ring: sphere)" if (best["shape"] == "sphere").any() else "  (all cylinders)"
    fig.suptitle(f"Lightest full-fidelity feasible design vs residual-speed cap{ring}", fontsize=12)
    save(fig, os.path.join(SUMMARY_DIR, "mass_vs_ures"))


def plot_heatmap(ff):
    from matplotlib.colors import LinearSegmentedColormap
    best = best_feasible(ff)
    caps = sorted(ff["u_res_max"].unique())
    mats = list(MAT_SHORT.values())
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#f0efec")
    vmin, vmax = best["m_total"].min(), best["m_total"].max()
    fig, axs = plt.subplots(1, len(caps), figsize=(3.4 * len(caps) + 1, 4), layout="constrained")
    for ax, u in zip(np.atleast_1d(axs), caps):
        grid = np.full((len(W.GAS_NAMES), len(mats)), np.nan)
        shp = np.full(grid.shape, "", dtype=object)
        for _, r in best[best["u_res_max"] == u].iterrows():
            i, j = W.GAS_NAMES.index(r["gas"]), mats.index(r["mat"])
            grid[i, j], shp[i, j] = r["m_total"], r["shape"][0].upper()
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                txt = "—" if np.isnan(v) else f"{v:.3f}\n{shp[i, j]}"
                dark = not np.isnan(v) and (v - vmin) / (vmax - vmin + 1e-12) > 0.5
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if dark else "#0b0b0b")
        ax.set_xticks(range(len(mats)), mats)
        ax.set_yticks(range(len(W.GAS_NAMES)), W.GAS_NAMES)
        ax.set_title(f"u_res_max = {u} m/s", fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axs, label="lightest feasible m_total [kg]", shrink=0.85)
    fig.suptitle("Lightest full-fidelity feasible mass per gas x material "
                 "(C = cylinder, S = sphere, — = no feasible design)", fontsize=12)
    save(fig, os.path.join(SUMMARY_DIR, "heatmap_mass_gas_material"))


def plot_parallel(ff):
    """Cylinder design vectors normalised to the DE bounds; grey = all feasible candidates."""
    lo_hi = W.make_bounds(W.SHAPE_CODE["cylinder"], W.scenario, W.env)
    cols = ["D0", "aspect", "n_ply", "P0", "phi_vent"]
    names = ["D0 [m]", "L0/D0 [-]", "n_ply [-]", "P0 [bar]", "phi_vent [%]"]
    scale = [1, 1, 1, 1e-5, 100]
    feas = ff[ff["candidate"] & ff["feasible_fullfid"] & (ff["shape"] == "cylinder")]
    norm = lambda df: np.column_stack([(df[c] - lo) / (hi - lo) for c, (lo, hi) in zip(cols, lo_hi)])
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    xs = np.arange(len(cols))
    for y in norm(feas):
        ax.plot(xs, y, color=MUTED, alpha=0.35, lw=1)
    caps = sorted(ff["u_res_max"].unique())
    for i, u in enumerate(caps):
        w = feas[(feas["u_res_max"] == u) & feas["winner"]]
        if len(w):
            row = w.iloc[0]
            ax.plot(xs, norm(w)[0], color=CAP_RAMP[i % 4], ls=CAP_LS[i % 4], lw=2.4, marker="o", ms=8,
                    mec="white", mew=1.5,
                    label=f"winner u_res_max = {u}: {MAT_SHORT[row['material']]} / {row['gas']}, "
                          f"m = {row['m_total']:.3f} kg")
    for k, (lo, hi) in enumerate(lo_hi):
        ax.annotate(f"{hi * scale[k]:.3g}", (k, 1), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#52514e")
        ax.annotate(f"{lo * scale[k]:.3g}", (k, 0), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=8, color="#52514e")
        ax.axvline(k, color="#c3c2b7", lw=1, zorder=0)
    ax.set_xticks(xs, names)
    ax.set_yticks([])
    ax.set_ylim(-0.08, 1.08)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.legend(loc="outside lower center", fontsize=8, frameon=False, ncol=2)
    ax.set_title("Cylinder design vectors within the DE bounds "
                 f"(grey: all {len(feas)} full-fidelity feasible candidates)", fontsize=11)
    save(fig, os.path.join(SUMMARY_DIR, "parallel_coordinates"))


def plot_seed_scatter(ff):
    """Per combination with a search-feasible best seed: the seeds' search mass relative to the
    best seed. Seeds that ended infeasible in the search get an 'x' in a band above the plot."""
    has_feas = ff.groupby(KEY)["feasible_search"].transform("any")
    d = ff[has_feas].copy()
    d["f_best"] = d.groupby(KEY)["f_search"].transform("min")
    d["dev_pct"] = 100 * (d["f_search"] - d["f_best"]) / d["f_best"]
    order = (d.groupby(KEY)["f_best"].first().reset_index()
             .sort_values(["u_res_max", "f_best"]).reset_index(drop=True))
    order["ix"] = np.arange(len(order))
    d = d.merge(order[KEY + ["ix"]], on=KEY)
    caps = sorted(ff["u_res_max"].unique())
    good = d[d["feasible_search"]]
    y_top = max(1.0, good["dev_pct"].max()) * 1.6
    y_band = y_top * 1.8
    fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
    for i, u in enumerate(caps):
        s = d[d["u_res_max"] == u]
        g_ok = s[s["feasible_search"] & s["feasible_fullfid"]]
        g_bad = s[s["feasible_search"] & ~s["feasible_fullfid"]]
        lost = s[~s["feasible_search"]]
        ax.plot(g_ok["ix"], g_ok["dev_pct"], ls="none", marker="o", ms=6, color=CAP_RAMP[i % 4],
                mec="white", mew=0.8, label=f"u_res_max = {u} m/s")
        ax.plot(g_bad["ix"], g_bad["dev_pct"], ls="none", marker="o", ms=6, mfc="none",
                mec=CAP_RAMP[i % 4], mew=1.2)
        ax.plot(lost["ix"], np.full(len(lost), y_band), ls="none", marker="x", ms=6,
                color=CAP_RAMP[i % 4], mew=1.4)
        if len(s):
            ax.annotate(f"u_res_max = {u}", xy=(s["ix"].mean(), 0), xytext=(0, -14),
                        textcoords="offset points", xycoords=("data", "axes fraction"),
                        ha="center", fontsize=8, color="#52514e")
        if i:
            ax.axvline(s["ix"].min() - 0.5, color="#c3c2b7", lw=1)
    ax.plot([], [], ls="none", marker="o", mfc="none", mec="#52514e",
            label="infeasible at full fidelity (with tolerance)")
    ax.plot([], [], ls="none", marker="x", color="#52514e", mew=1.4, label="seed infeasible in the search")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.003, y_band * 1.6)
    ax.axhline(y_top * 1.25, color="#c3c2b7", lw=0.8)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1)
    ax.annotate("1 %", xy=(1, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", fontsize=8, color="#52514e")
    ax.set_xticks([])
    ax.set_xlabel("\nDE combinations (shape / material / gas), sorted by best mass within each cap")
    ax.set_ylabel("search mass vs best seed [%]")
    style_axes(ax)
    fig.legend(loc="outside lower center", fontsize=8, frameon=False, ncol=6)
    ax.set_title("Seed-to-seed spread of the DE result, 5 seeds per combination", fontsize=11)
    save(fig, os.path.join(SUMMARY_DIR, "seed_spread"))


def build_summary(ff):
    print("\n[4] design-space summary plots")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    plot_mass_vs_ures(ff)
    plot_heatmap(ff)
    plot_parallel(ff)
    plot_seed_scatter(ff)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPUs)")
    ap.add_argument("--no-matrices", action="store_true", help="skip the per-candidate plot matrices")
    ap.add_argument("--force", action="store_true", help="redo the cached table and matrices")
    ap.add_argument("--t-cut", type=float, default=None,
                    help="time-axis end of the *_impact overlays [s]; default 1.5 x latest transient end")
    ap.add_argument("--a-tol-pct", type=float, default=A_TOL_PCT,
                    help=f"tolerance on a_peak_n <= A_LIM [%% of A_LIM] (default {A_TOL_PCT}; 0 = strict)")
    cli = ap.parse_args()
    workers = cli.workers or W.n_available_cpus()

    de = pd.read_csv(DE_CSV)
    print(f"DE results: {os.path.relpath(DE_CSV, HERE)} ({len(de)} runs, "
          f"{de.groupby(KEY).ngroups} combinations) | a_peak_n tolerance {cli.a_tol_pct} % of A_LIM")
    ff = build_fullfid_table(de, workers, cli.force, cli.a_tol_pct)
    report_table(ff, cli.a_tol_pct)
    if not cli.no_matrices:
        build_matrices(ff, workers, cli.force)
    build_overlays(ff, cli.t_cut)
    build_summary(ff)
