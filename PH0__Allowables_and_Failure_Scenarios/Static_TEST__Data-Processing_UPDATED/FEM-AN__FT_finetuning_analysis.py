"""
FEM-AN__FT_finetuning_analysis.py

Overlays FEM force-displacement curves from a sweep of candidate FT values,
extracts the initial linear (elastic) region up to the first detected
stiffness drop, extrapolates that linear fit across the plot, colors/labels
each curve by its FT value (parsed from filename), and draws a manual
"target failure load" threshold line -- so the FT value whose curve first
crosses the target closest to where the real curve drops off can be read
off by eye.

USAGE:
    python FEM-AN__FT_finetuning_analysis.py <folder> --target-load 250

    python FEM-AN__FT_finetuning_analysis.py <folder> --target-load 250 \
        --window 15 --drop-threshold 0.10 --n-init-windows 3

INPUT FILES
-----------
Same paired-export mechanism as FEM-AN__stiffness_comparison_EXP-vs-FEM.py:
FEM doesn't export one ready-made force-vs-displacement file -- each
candidate FT run is two separate time-history exports, a "<tag>_disp" and a
"<tag>_force" file, both sampled at the same time steps (same first column).
Drop both files for a run into the data folder and they're picked up
automatically -- no config edit needed. A "<tag>_disp" file without a
matching "<tag>_force" (or vice versa) is skipped with a printed warning.

Force exports are in kN; scaled by FORCE_SCALE (1000) on load so all
downstream analysis/plotting is in N.

FILENAME -> FT ASSUMPTION:
    Trailing digit group in "<tag>" maps to FT via
        value = int(digits) / 10**len(digits) * 1000
    e.g. tag "045" (from "045_disp" / "045_force") -> 0.045 * 1000 = FT = 45
    (GPa -> MPa conversion). This REQUIRES consistent zero-padding across all
    tags (e.g. always 3 digits). Edit parse_ft() regex/scaling if your
    naming differs.
"""

import re
import glob
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================== CONFIGURATION ==============================
DATA_FOLDER = Path(__file__).parent / "fem_ft-finetuning_curves"
TARGET_FAILURE_LOAD = 100                    # [N]

# Every FEM force export is in kN, so this is applied to every curve.
X_SCALE = 1.0
FORCE_SCALE = 1000.0   # kN -> N
# ============================================================================


def parse_ft(tag: str) -> float:
    """Extract FT from trailing digit group in a '<tag>' (e.g. from
    '<tag>_disp' / '<tag>_force'), converted GPa -> MPa (x1000)."""
    match = re.search(r'(\d+)$', tag)
    if not match:
        raise ValueError(f"No trailing digit group found in tag: {tag}")
    digits = match.group(1)
    return (int(digits) / 10 ** len(digits)) * 1000


def discover_fem_curve_pairs(folder):
    """Finds every '<tag>_disp' file in folder with a matching '<tag>_force'
    next to it (e.g. '045_disp' + '045_force' -> tag '045'). New FT sweeps
    just need their two exported files dropped in here -- nothing to
    configure. Returns a sorted list of (tag, disp_path, force_path)."""
    pairs = []
    for disp_path in sorted(glob.glob(os.path.join(folder, "*_disp"))):
        tag = os.path.basename(disp_path)[: -len("_disp")]
        force_path = os.path.join(folder, f"{tag}_force")
        if not os.path.isfile(force_path):
            print(f"  SKIPPED '{tag}': found '{os.path.basename(disp_path)}' but no "
                  f"matching '{tag}_force' file next to it.")
            continue
        pairs.append((tag, disp_path, force_path))
    return pairs


def parse_history_file(path):
    """Parses one FEM time-history export: a point-count header line,
    then 'time  value' pairs. Returns (time, value) arrays."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    rows = []
    for line in lines[1:]:  # skip the point-count header line
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    rows = np.array(rows)
    return rows[:, 0], rows[:, 1]


def build_combined_fem_curve(tag, disp_path, force_path, x_scale=X_SCALE, y_scale=FORCE_SCALE):
    """Combines a '<tag>_disp' and '<tag>_force' time-history pair (both
    sampled at the same time steps) into one displacement-vs-force curve."""
    disp_time, disp_val = parse_history_file(disp_path)
    force_time, force_val = parse_history_file(force_path)

    n = min(len(disp_val), len(force_val))
    if len(disp_val) != len(force_val):
        print(f"  WARNING '{tag}': disp has {len(disp_val)} points, force has {len(force_val)} "
              f"-- using the first {n} common to both, verify the exports actually line up.")
    disp_time, disp_val = disp_time[:n], disp_val[:n]
    force_time, force_val = force_time[:n], force_val[:n]

    if not np.allclose(disp_time, force_time, atol=1e-6):
        print(f"  WARNING '{tag}': time columns in the disp and force files don't match -- "
              f"they may not actually be paired correctly.")

    # FEM commonly reports compression as a negative displacement (platen
    # moving in the solver's negative direction) -- take the magnitude so it
    # climbs left-to-right.
    disp_val = np.abs(disp_val) * x_scale
    force_val = force_val * y_scale
    return disp_val, force_val


def find_stiffness_drop(disp, force, window=15, drop_threshold=0.10, n_init_windows=3):
    """
    Sliding-window linear regression along the curve to find local stiffness (slope).

    Reference (initial) stiffness k0 = slope of the best-R^2 window among the first
    n_init_windows window-widths of data.

    Returns:
        drop_idx : index of first point where local slope has fallen below
                   k0 * (1 - drop_threshold). Falls back to last valid window
                   if no drop is detected (curve stayed linear throughout).
        k0       : reference (initial) stiffness [force/disp]
    """
    n = len(disp)
    if n < 3:
        raise ValueError(f"Curve too short ({n} pts) to fit any line — need at least 3 points.")

    if n < window + 1:
        window = max(2, n - 1)  # clamp window down to whatever the curve can support

    slopes = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    for i in range(n - window):
        x = disp[i:i + window]
        y = force[i:i + window]
        res = linregress(x, y)
        slopes[i] = res.slope
        r2[i] = res.rvalue ** 2

    if n_init_windows < 1:
        n_init_windows = 1  # need at least one window's worth of data to get a reference stiffness

    init_end = min(n_init_windows * window, n - window)
    ref_idx = int(np.nanargmax(r2[:init_end]))
    k0 = slopes[ref_idx]

    drop_idx = None
    for i in range(ref_idx, n - window):
        if slopes[i] < k0 * (1 - drop_threshold):
            drop_idx = i
            break

    if drop_idx is None:
        drop_idx = n - window - 1  # no drop found -> curve stayed linear

    return drop_idx, k0


def main():
    parser = argparse.ArgumentParser(
        description="Overlay FEM force-disp curves with linear-elastic extrapolation to "
                    "first stiffness drop, labeled by FT."
    )
    parser.add_argument("folder", type=str, nargs="?", default=None,
                         help="Folder containing paired '<tag>_disp' / '<tag>_force' curves. "
                              "Overrides DATA_FOLDER constant if given.")
    parser.add_argument("--target-load", type=float, default=None,
                         help="Target failure load (red threshold line). Overrides TARGET_FAILURE_LOAD constant if given.")
    parser.add_argument("--window", type=int, default=15,
                         help="Sliding window size [points] for local stiffness regression")
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                         help="Fractional stiffness drop defining 'first drop' (0.10 = 10%% below initial slope)")
    parser.add_argument("--n-init-windows", type=int, default=3,
                         help="Number of window-widths used to establish reference (initial) stiffness")
    parser.add_argument("--out", type=str, default=None,
                         help="Output PNG path (default: <folder>/ft_finetuning_overlay.png)")
    args = parser.parse_args()

    target_load = args.target_load if args.target_load is not None else TARGET_FAILURE_LOAD
    folder = Path(args.folder if args.folder is not None else DATA_FOLDER)

    pairs = discover_fem_curve_pairs(str(folder))
    if not pairs:
        raise FileNotFoundError(f"No paired '<tag>_disp' / '<tag>_force' curves found in {folder}")
    print(f"Found {len(pairs)} FEM curve(s) in '{folder}/'.")

    curves = []
    for tag, disp_path, force_path in pairs:
        ft = parse_ft(tag)
        disp, force = build_combined_fem_curve(tag, disp_path, force_path)
        curves.append((tag, ft, disp, force))

    ft_vals = [ft for _, ft, _, _ in curves]
    vmin, vmax = min(ft_vals), max(ft_vals)
    norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-9)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9, 6))

    for tag, ft, disp, force in curves:
        try:
            drop_idx, k0 = find_stiffness_drop(
                disp, force,
                window=args.window,
                drop_threshold=args.drop_threshold,
                n_init_windows=args.n_init_windows,
            )
        except ValueError as e:
            print(f"[skip] {tag}: {e}")
            continue

        color = cmap(norm(ft))
        ax.plot(disp, force, color=color, lw=1.5, label=f"FT={ft:g}")

        # linear fit on data up to the drop point, extrapolated across the full x-range
        x_lin = disp[:drop_idx]
        y_lin = force[:drop_idx]
        fit = linregress(x_lin, y_lin)
        x_extrap = np.array([0.0, disp[-1]])
        y_extrap = fit.intercept + fit.slope * x_extrap
        ax.plot(x_extrap, y_extrap, color=color, lw=1, ls="--", alpha=0.6)

        # mark detected stiffness-drop point
        ax.scatter(disp[drop_idx], force[drop_idx], color=color, marker="x", zorder=5)

    ax.axhline(target_load, color="red", lw=2, label="Target failure load")

    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Force-displacement for FT finetuning")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else folder / "ft_finetuning_overlay.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
