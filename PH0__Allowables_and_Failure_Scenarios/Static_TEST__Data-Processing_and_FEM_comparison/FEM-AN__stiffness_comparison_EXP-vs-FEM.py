"""
Overlays every phial's load-displacement curve on one plot to visually
compare stiffness (steepness of the climb), colored by batch.

Each curve is:
  1. Trimmed at the START: any leading stretch where the load is still just
     sitting in the noise floor near zero (grip slack / pre-contact) is
     dropped, so every curve begins right where it genuinely starts climbing.
  2. Shifted so that climb-start point becomes (0, 0) for every phial --
     otherwise curves are staggered by however long each one's dead zone
     happened to be, which makes slope harder to compare by eye.
  3. Cut at the END at that phial's first-break index (from find_first_break)
     -- nothing post-failure is physically meaningful here.

EXCLUSIONS
----------
- Manual: anything listed in MANUAL_EXCLUDE (e.g. "54B"), no questions asked.
- Automatic: if a phial's load never dips into the noise floor at all AND its
  very first sample is already well above it, that's a real preload/offset,
  not just noise -- forcing it to (0,0) would be fabricating a start that
  never happened, so it's excluded and logged instead.

With up to ~76 curves overlaid, this uses ONE legend entry per BATCH (not per
curve -- 76 legend rows would be unreadable), drawing each curve semi-
transparent so overlapping stiffness trends are still visible.

FEM CURVES
----------
FEM results aren't exported as one ready-made force-vs-displacement .dat --
each mode is two separate history exports, a "<tag>_disp" and a "<tag>_force"
file, both time series sampled at the same time steps (same first column).
Drop both files for a mode into FEM_CURVES_DIR and they're picked up
automatically -- no config edit needed, so new modes as the FEM analyses
progress just mean adding two files. A "<tag>_disp" file without a matching
"<tag>_force" (or vice versa) is skipped with a printed warning rather than
silently ignored.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from failure_extraction import find_first_break

# ============================== CONFIGURATION ==============================
CURVES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_curves")

OUTPUT_PLOT_EXPERIMENTAL_ONLY = "stiffness_overlay_experimental_only.png"
OUTPUT_PLOT_WITH_FEM = "stiffness_overlay_experimental_vs_FEM.png"

# Folder to drop paired "<tag>_disp" / "<tag>_force" FEM history exports
# into -- every pair found here becomes one overlaid curve, automatically.
FEM_CURVES_DIR = "fem_load-vs-disp_curves"

# Every FEM force export is in kN, so this is applied to every curve by
# default. x_scale stays 1.0 -- displacement exports are already in mm.
FEM_DEFAULT_X_SCALE = 1.0
FEM_DEFAULT_Y_SCALE = 1000.0   # kN -> N

# Per-mode override, keyed by the same "<tag>" used in its filenames (e.g.
# "modeA"), for the rare curve that doesn't follow the kN default above.
# Leave a mode out to use FEM_DEFAULT_X_SCALE / FEM_DEFAULT_Y_SCALE.
FEM_CURVE_SCALE_OVERRIDES = {
    # "modeA": {"y_scale": 1.0},   # e.g. if one export is already in N
}
FEM_FIT_LINEWIDTH = 1.0   # thinner than the raw curves, so it reads as a guide, not a data series

MANUAL_EXCLUDE = ["54B"]   # add more "<number><batch>" strings here as needed

NOISE_THRESHOLD_N = 2.0    # load below this is considered "still in the noise floor"
SMOOTH_WINDOW = 9          # light smoothing before detecting the climb-start boundary

# --- first-break detection parameters, same as the rest of the pipeline ---
FB_SMOOTH_WINDOW = 11
FB_PROMINENCE_ABS_N = 22.0
FB_PROMINENCE_FRACTION = 0.0
FB_MIN_PEAK_LOAD_N = 2.0
FB_REFINE_RADIUS = 5

BATCH_COLORS = {"A": "tab:orange", "B": "tab:cyan"}
CURVE_ALPHA = 0.5

# FEM curves are drawn on top of the experimental cloud, so they deliberately
# stay clear of the experimental orange/cyan (and of blue, which reads as cyan
# once the curves overlap). Strong, saturated, mutually distinct hues only.
FEM_COLORS = [
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#e377c2",  # magenta
    "#8c564b",  # brown
    "#000000",  # black
    "#bcbd22",  # olive
    "#7f0000",  # dark red
    "#4b0082",  # indigo
    "#006400",  # dark green
]
# ============================================================================


def parse_manual_exclude(entries):
    """'54B' -> (54, 'B'). Leading zeros in the number don't matter for the
    comparison, so '05A' and '5A' both refer to the same phial."""
    parsed = set()
    for e in entries:
        m = re.match(r"^\s*(\d+)\s*([A-Za-z]+)\s*$", e)
        if m:
            parsed.add((int(m.group(1)), m.group(2).upper()))
    return parsed


def find_climb_start(load, noise_threshold=NOISE_THRESHOLD_N, smooth_window=SMOOTH_WINDOW):
    """
    Index of the last point after which load stays above the noise floor
    all the way to the end of the array passed in (i.e. to the break).
    Scanning BACKWARD from the end for the boundary of the trailing
    "above-threshold" run -- rather than forward for the first crossing --
    means a single noisy blip early in the flat region can't be mistaken
    for the real climb start, since it wouldn't be part of the run that
    actually reaches all the way to the end.

    Returns len(load) if the load never gets above the noise floor at all.
    """
    load = np.asarray(load, dtype=float)
    n = len(load)
    if n < smooth_window:
        smoothed = load
    else:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(load, kernel, mode="same")

    above = smoothed > noise_threshold
    idx = n - 1
    while idx >= 0 and above[idx]:
        idx -= 1
    return idx + 1  # first index of the trailing above-threshold run


def discover_fem_curve_pairs(folder):
    """Finds every '<tag>_disp' file in folder with a matching '<tag>_force'
    next to it (e.g. 'modeA_disp' + 'modeA_force' -> tag 'modeA'). New FEM
    modes just need their two exported files dropped in here -- nothing to
    configure. Returns a sorted list of (tag, disp_path, force_path)."""
    if not os.path.isdir(folder):
        print(f"  FEM curves folder '{folder}' does not exist -- no FEM curves to overlay.")
        return []

    pairs = []
    for disp_path in sorted(glob.glob(os.path.join(folder, "*_disp"))):
        tag = os.path.basename(disp_path)[: -len("_disp")]
        force_path = os.path.join(folder, f"{tag}_force")
        if not os.path.isfile(force_path):
            print(f"  SKIPPED FEM curve '{tag}': found '{os.path.basename(disp_path)}' but no "
                  f"matching '{tag}_force' file next to it.")
            continue
        pairs.append((tag, disp_path, force_path))
    return pairs


def mode_label_from_tag(tag):
    """'modeA' -> 'A'; anything not starting with 'mode' is used as-is, so
    future files don't have to follow that exact naming pattern."""
    return re.sub(r"(?i)^mode", "", tag) or tag


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


def build_combined_fem_curve(tag, disp_path, force_path,
                              x_scale=FEM_DEFAULT_X_SCALE, y_scale=FEM_DEFAULT_Y_SCALE):
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
    # climbs left-to-right like the (already shifted-to-start-at-zero)
    # experimental curves do.
    disp_val = np.abs(disp_val) * x_scale
    force_val = force_val * y_scale
    return disp_val, force_val


def plot_fem_curve(xs, ys, mode_label, color, linewidth=2, fit_linewidth=FEM_FIT_LINEWIDTH):
    """Fits a linear regression to an already-combined (displacement, force)
    FEM curve, prints the gradient, and plots both the raw curve and the
    fitted line on the CURRENT axes -- call this after the phial curves are
    plotted but before plt.legend(), so it shows up in the same legend."""
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"FEM Mode {mode_label} linear fit: gradient = {slope:.6g} N/mm  intercept = {intercept:.6g}")

    plt.plot(xs, ys, color=color, linewidth=linewidth, label=f"FEM - Mode {mode_label}")
    x_fit = np.array([xs.min(), xs.max()])
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color=color, linestyle="--", linewidth=fit_linewidth,
              label=f"FEM - Mode {mode_label} fit (E={slope:.4g} N/mm)")


def process_one_file(filepath, manual_exclude_set):
    data = pd.read_excel(filepath, sheet_name="Data")
    meta = pd.read_excel(filepath, sheet_name="Metadata")
    meta_dict = dict(zip(meta["Field"], meta["Value"]))

    phial_number = int(meta_dict.get("Phial number"))
    batch = str(meta_dict.get("Batch"))

    if (phial_number, batch) in manual_exclude_set:
        return None, f"Phial {phial_number}{batch}: excluded (manual exclusion list)"

    break_result = find_first_break(
        data["Time_s"].values, data["Load_N"].values,
        smooth_window=FB_SMOOTH_WINDOW, prominence_abs_n=FB_PROMINENCE_ABS_N,
        prominence_fraction=FB_PROMINENCE_FRACTION, min_peak_load_n=FB_MIN_PEAK_LOAD_N,
        refine_radius=FB_REFINE_RADIUS,
    )
    break_idx = break_result["first_break_index"]

    disp = data["Extension_mm"].values[:break_idx + 1]
    load = data["Load_N"].values[:break_idx + 1]

    # Does the curve actually BEGIN near zero? This is checked at the true
    # start of the curve (a few points, averaged for noise robustness) --
    # NOT at the climb-start crossing point found below, which will always
    # sit right around the threshold almost by definition and so can't be
    # used to judge whether there was a real preload.
    starting_baseline = np.mean(load[:min(5, len(load))])
    if abs(starting_baseline) > NOISE_THRESHOLD_N:
        return None, (f"Phial {phial_number}{batch}: excluded (does not start near zero force -- "
                       f"load at t=0 is {starting_baseline:.1f} N)")

    climb_start = find_climb_start(load)
    if climb_start >= len(load):
        return None, f"Phial {phial_number}{batch}: excluded (load never rises above noise floor)"

    baseline = load[climb_start]

    disp_shifted = disp[climb_start:] - disp[climb_start]
    load_shifted = load[climb_start:] - baseline

    return {
        "phial_number": phial_number, "batch": batch,
        "disp": disp_shifted, "load": load_shifted,
    }, None


def main():
    manual_exclude_set = parse_manual_exclude(MANUAL_EXCLUDE)
    filepaths = sorted(glob.glob(os.path.join(CURVES_DIR, "*.xlsx")))
    print(f"Found {len(filepaths)} phial curve files in '{CURVES_DIR}/'.")

    curves = []
    for fp in filepaths:
        result, exclusion_note = process_one_file(fp, manual_exclude_set)
        if exclusion_note:
            print(" ", exclusion_note)
        if result is not None:
            curves.append(result)

    print(f"\nPlotting {len(curves)} of {len(filepaths)} curves.")

    # ---- Plot 1: experimental data only ----
    plt.figure(figsize=(11, 6))
    seen_batches = {}
    for c in curves:
        color = BATCH_COLORS.get(c["batch"], "gray")
        plt.plot(c["disp"], c["load"], color=color, alpha=CURVE_ALPHA, linewidth=1)
        seen_batches.setdefault(c["batch"], 0)
        seen_batches[c["batch"]] += 1

    # one proxy legend entry per batch, not per curve
    for batch, count in sorted(seen_batches.items()):
        plt.plot([], [], color=BATCH_COLORS.get(batch, "gray"), linewidth=2,
                  label=f"Batch {batch} (n={count})")

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.title("Stiffness comparison — experimental data only")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_EXPERIMENTAL_ONLY, dpi=150)
    print(f"\nSaved experimental-only overlay plot: {OUTPUT_PLOT_EXPERIMENTAL_ONLY}")

    # ---- Plot 2: same axes, with every FEM curve found in FEM_CURVES_DIR added on top ----
    fem_pairs = discover_fem_curve_pairs(FEM_CURVES_DIR)
    print(f"\nFound {len(fem_pairs)} FEM curve(s) in '{FEM_CURVES_DIR}/'.")
    for i, (tag, disp_path, force_path) in enumerate(fem_pairs):
        overrides = FEM_CURVE_SCALE_OVERRIDES.get(tag, {})
        xs, ys = build_combined_fem_curve(
            tag, disp_path, force_path,
            x_scale=overrides.get("x_scale", FEM_DEFAULT_X_SCALE),
            y_scale=overrides.get("y_scale", FEM_DEFAULT_Y_SCALE),
        )
        plot_fem_curve(xs, ys, mode_label_from_tag(tag), FEM_COLORS[i % len(FEM_COLORS)])

    plt.title("Stiffness comparison — TEST vs. FEM")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_WITH_FEM, dpi=150)
    plt.close()
    print(f"Saved experimental-vs-FEM overlay plot: {OUTPUT_PLOT_WITH_FEM}")


if __name__ == "__main__":
    main()