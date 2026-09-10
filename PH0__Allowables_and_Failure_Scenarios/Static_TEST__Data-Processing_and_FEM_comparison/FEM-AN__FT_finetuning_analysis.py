"""
FEM-AN__FT_finetuning_analysis.py

Overlays FEM force-displacement curves from a sweep of candidate FT values
[MPa], marks each run's breaking load [N], colors/labels each curve by its FT
value (parsed from filename) and its breaking load, and draws a manual
"target failure load" threshold line -- so the FT that puts the model's
failure load on target can be read straight off the legend.

Failure load only: stiffness is not fitted or reported here, that is what
FEM-AN__stiffness_comparison_EXP-vs-FEM.py is for.

The breaking load is the PEAK load a curve carries before it sheds load
(marked "o"). The material is glass: it is linear-elastic right up to
fracture, so there is no yield or softening knee before failure and none is
looked for -- the peak IS the failure.

Each curve is trimmed shortly after its drop (see trim_after_failure): the
drop is shown whole with a short settled tail beneath it, and the long
post-fracture plateau -- which says nothing about the failure load -- is
discarded. The axes follow the trimmed curves.

USAGE:
    python FEM-AN__FT_finetuning_analysis.py <folder> --target-load 250

    python FEM-AN__FT_finetuning_analysis.py <folder> --target-load 250 \
        --post-drop-frac 0.20 --settle-frac 0.05 --tail-frac 1.0

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

# ============================== CONFIGURATION ==============================
DATA_FOLDER = Path(__file__).parent / "fem_ft-finetuning_curves"
TARGET_FAILURE_LOAD = 98.7                    # [N]  (FT sweep values are in MPa)

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


def trim_after_failure(disp, force, fail_idx, settle_frac=0.05, tail_frac=1.0):
    """
    Cut the curve off shortly after the failure drop.

    The runs carry on for a long way past failure -- a crushed-material
    plateau several times the displacement of interest -- and none of it says
    anything about the failure load. Keeping it squeezes the loading branch
    and the drop itself into a sliver of the axes.

    The cut is placed from the curve's own geometry rather than a hard-coded
    displacement, so it adapts to each run: find where the load stops falling
    (the bottom of the drop), then keep a further tail_frac of the drop's own
    displacement width beyond it. The drop is therefore always shown whole,
    with a proportionate stretch of settled post-failure load beneath it for
    context, and nothing after that.

    Args:
        settle_frac : the drop is over once the load has climbed back this
                      fraction of its depth above its lowest point -- enough
                      to step over the ripple in the plateau.
        tail_frac   : how much of the drop's displacement width to keep past
                      the bottom (1.0 = tail as wide as the drop).

    Returns:
        end : slice end index -- plot disp[:end], force[:end].
    """
    n = len(force)
    post = force[fail_idx:]
    running_min = np.minimum.accumulate(post)
    depth = force[fail_idx] - running_min

    # while the load is still falling post == running_min, so this stays False;
    # it first trips once the curve has turned back up off the bottom
    rebound = np.flatnonzero(post > running_min + settle_frac * depth)
    if rebound.size == 0:
        return n  # never settles -> nothing to trim

    bottom_idx = fail_idx + int(np.argmin(post[:rebound[0] + 1]))
    width = disp[bottom_idx] - disp[fail_idx]
    beyond = np.flatnonzero(disp > disp[bottom_idx] + tail_frac * width)
    end = int(beyond[0]) + 1 if beyond.size else n
    return min(max(end, bottom_idx + 1), n)


def find_failure_point(force, post_drop_frac=0.20):
    """
    Locate the failure (breaking) point: the PEAK load the curve carries before
    it sheds a significant fraction of it.

    This is the quantity to compare against the target failure load. It is not
    the same as the first stiffness drop -- the curve keeps taking more load for
    a while after it starts softening (~35 N more on these runs), so reading the
    failure load off the stiffness-drop point reports it far too low.

    The curve is scanned for the first point that has shed more than
    post_drop_frac of the run's overall peak load relative to the running
    maximum; the failure point is then the highest load reached before it.

    The shed is measured against the run's OVERALL peak, not against the
    running maximum alone -- a purely relative test ("fell 20% below the
    running max") is scale-free, so the sub-newton wiggle in the pre-contact
    region trips it and the reported failure load collapses to ~0.2 N.
    Anchoring to the overall peak makes the test read "the load dropped by
    more than 20% of the largest load in this run", which noise can't reach.

    Returns:
        fail_idx : index of the peak (breaking) load
        failed   : False if no load shed was found at all, meaning the run was
                   probably not carried through to failure and fail_idx is just
                   the largest load reached so far.
    """
    force = np.asarray(force, dtype=float)
    peak = float(np.max(force))
    if peak <= 0:
        return int(np.argmax(force)), False
    shed_from_running_max = np.maximum.accumulate(force) - force
    shed = np.flatnonzero(shed_from_running_max > post_drop_frac * peak)
    if shed.size == 0:
        return int(np.argmax(force)), False
    return int(np.argmax(force[:shed[0]])), True


def main():
    parser = argparse.ArgumentParser(
        description="Overlay FEM force-disp curves, mark each run's breaking load, and compare "
                    "against a target failure load. Labeled by FT [MPa]."
    )
    parser.add_argument("folder", type=str, nargs="?", default=None,
                         help="Folder containing paired '<tag>_disp' / '<tag>_force' curves. "
                              "Overrides DATA_FOLDER constant if given.")
    parser.add_argument("--target-load", type=float, default=None,
                         help="Target failure load [N] (red threshold line). Overrides "
                              "TARGET_FAILURE_LOAD constant if given.")
    parser.add_argument("--post-drop-frac", type=float, default=0.20,
                         help="Load shed marking genuine failure, as a fraction of the run's overall peak "
                              "load (0.20 = load fell by 20%% of the peak). The breaking force is the "
                              "highest load reached before this shed.")
    parser.add_argument("--settle-frac", type=float, default=0.05,
                         help="Curve is considered settled once it climbs this fraction of the drop's "
                              "depth back above its lowest point (marks the bottom of the drop).")
    parser.add_argument("--tail-frac", type=float, default=1.0,
                         help="How much of the drop's displacement width to keep past the bottom of the "
                              "drop (1.0 = tail as wide as the drop). Everything after is discarded.")
    parser.add_argument("--xmax", type=float, default=None,
                         help="Right x-limit [mm]. Default: the end of the trimmed curves.")
    parser.add_argument("--ymax", type=float, default=None,
                         help="Top y-limit [N]. Default: 1.2x the largest of the breaking loads and the target.")
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

    summary = []
    x_ends = []
    for tag, ft, disp, force in curves:
        # breaking force = peak load carried before the curve sheds load
        fail_idx, failed = find_failure_point(force, post_drop_frac=args.post_drop_frac)
        fail_load = force[fail_idx]
        if not failed:
            print(f"  WARNING '{tag}': load never dropped {args.post_drop_frac:.0%} below its peak -- "
                  f"this run may not have been carried to failure; reporting the highest load reached.")

        # keep the loading branch and the drop, discard the plateau after it
        end = trim_after_failure(disp, force, fail_idx,
                                 settle_frac=args.settle_frac, tail_frac=args.tail_frac)
        disp, force = disp[:end], force[:end]
        x_ends.append(disp[-1])

        summary.append((ft, fail_load))

        color = cmap(norm(ft))
        ax.plot(disp, force, color=color, lw=1.5,
                label=f"FT = {ft:g} MPa  ->  {fail_load:.1f} N")
        ax.scatter(disp[fail_idx], fail_load, color=color, marker="o", s=55,
                   zorder=6, edgecolors="black", linewidths=0.8)

    ax.axhline(target_load, color="red", lw=2, label=f"Target failure load = {target_load:g} N")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="0.25", s=55,
               label="Breaking load")

    if summary:
        print()
        print(f"{'FT [MPa]':>10}  {'F_fail [N]':>11}  {'vs target':>10}")
        for ft, fail_load in sorted(summary):
            print(f"{ft:>10.6g}  {fail_load:>11.1f}  {fail_load - target_load:>+10.1f}")
        best = min(summary, key=lambda r: abs(r[1] - target_load))
        print()
        print(f"Closest to the {target_load:g} N target: FT = {best[0]:g} MPa "
              f"(F_fail = {best[1]:.1f} N)")
        print()

    # axes follow the trimmed curves, so the loading branch and the drop fill the plot
    if x_ends:
        x_max = max(x_ends)
        y_max = max([load for _, load in summary] + [target_load])
        ax.set_xlim(left=-0.02 * x_max, right=args.xmax if args.xmax is not None else x_max)
        ax.set_ylim(bottom=-0.08 * y_max, top=args.ymax if args.ymax is not None else 1.2 * y_max)

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
