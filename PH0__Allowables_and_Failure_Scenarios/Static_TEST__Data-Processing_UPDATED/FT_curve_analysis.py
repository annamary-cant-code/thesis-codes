"""
FT_curve_analysis.py

Overlays force-displacement .dat curves from a folder, extracts the initial linear
(elastic) region up to the first detected stiffness drop, extrapolates that linear
fit across the plot, labels each curve by its FT value (parsed from filename),
and draws a manual "target failure load" threshold line.

USAGE:
    python FT_curve_analysis.py <folder> --target-load 250

    python FT_curve_analysis.py <folder> --target-load 250 \
        --window 15 --drop-threshold 0.10 --n-init-windows 3

INPUT FILE FORMAT:
    XYDATA-style .dat file:
        XYDATA, Curve 1
        <disp>    <force>
        ...
        ENDATA
    (displacement in column 1, force in column 2, scientific notation, comma after
    XYDATA on line 1). Force in the .dat file is in kN; it is scaled by FORCE_SCALE
    (1000) on load so all downstream analysis/plotting is in N.
    Edit load_curve() if a given file breaks this convention.

FILENAME -> FT ASSUMPTION:
    Trailing digit group maps to FT via value = int(digits) / 10**len(digits) * 1000
    e.g. force_disp_045.dat -> 0.045 * 1000 = FT = 45  (GPa -> MPa conversion)
    This REQUIRES consistent zero-padding across all filenames (e.g. always 3 digits).
    Edit parse_ft() regex/scaling if your naming differs.
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================================================
# MANUAL iNPUTS
# ============================================================
DATA_FOLDER = r"C:\0_CODES_MScThesis\CODES\PH0__Allowables_and_Failure_Scenarios\Static_TEST__Data-Processing_UPDATED\DAT_for_FINETUNING"
TARGET_FAILURE_LOAD = 100                    # [N/mm]
FORCE_SCALE = 1000                           # .dat force from kN -> N
# ============================================================


def parse_ft(filename: str) -> float:
    """Extract FT from trailing digit group in filename stem, converted GPa -> MPa (x1000)."""
    stem = Path(filename).stem
    match = re.search(r'(\d+)$', stem)
    if not match:
        raise ValueError(f"No trailing digit group found in filename: {filename}")
    digits = match.group(1)
    return (int(digits) / 10 ** len(digits)) * 1000


def load_curve(filepath: Path):
    """
    Load an XYDATA-style .dat file:
        XYDATA, Curve 1
        <disp>    <force>
        ...
        ENDATA
    Skips the header/footer lines and parses the numeric rows in between.
    Force column is in kN in the file; scaled by FORCE_SCALE to N here.
    """
    disp, force = [], []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.upper().startswith(("XYDATA", "ENDATA")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            disp.append(x)
            force.append(y)

    if not disp:
        raise ValueError(f"No numeric XY rows parsed from {filepath}")

    disp = np.array(disp)
    force = np.array(force) * FORCE_SCALE
    return disp, force


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
        description="Overlay force-disp curves with linear-elastic extrapolation to "
                    "first stiffness drop, labeled by FT."
    )
    parser.add_argument("folder", type=str, nargs="?", default=None,
                         help="Folder containing .dat curves. Overrides DATA_FOLDER constant if given.")
    parser.add_argument("--target-load", type=float, default=None,
                         help="Target failure load (red threshold line). Overrides TARGET_FAILURE_LOAD constant if given.")
    parser.add_argument("--window", type=int, default=15,
                         help="Sliding window size [points] for local stiffness regression")
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                         help="Fractional stiffness drop defining 'first drop' (0.10 = 10%% below initial slope)")
    parser.add_argument("--n-init-windows", type=int, default=3,
                         help="Number of window-widths used to establish reference (initial) stiffness")
    parser.add_argument("--out", type=str, default=None,
                         help="Output PNG path (default: <folder>/ft_overlay.png)")
    args = parser.parse_args()

    target_load = args.target_load if args.target_load is not None else TARGET_FAILURE_LOAD
    folder = Path(args.folder if args.folder is not None else DATA_FOLDER)
    files = sorted(folder.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"No .dat files found in {folder}")

    ft_vals = [parse_ft(f.name) for f in files]
    vmin, vmax = min(ft_vals), max(ft_vals)
    norm = plt.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-9)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9, 6))

    for f, ft in zip(files, ft_vals):
        disp, force = load_curve(f)

        try:
            drop_idx, k0 = find_stiffness_drop(
                disp, force,
                window=args.window,
                drop_threshold=args.drop_threshold,
                n_init_windows=args.n_init_windows,
            )
        except ValueError as e:
            print(f"[skip] {f.name}: {e}")
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

    out_path = Path(args.out) if args.out else folder / "ft_overlay.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()