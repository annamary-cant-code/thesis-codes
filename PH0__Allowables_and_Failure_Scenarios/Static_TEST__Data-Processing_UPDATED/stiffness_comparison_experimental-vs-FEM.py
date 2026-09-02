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
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from failure_extraction import find_first_break

# ============================== CONFIGURATION ==============================
CURVES_DIR = r"C:\0_CODES_MScThesis\CODES\PH0__Allowables_and_Failure_Scenarios\Static_TEST__Data-Processing_UPDATED\output_curves"
OUTPUT_PLOT = "stiffness_overlay_SHELL_mat_glass_elastic.png"
REFERENCE_CURVE = r"C:\0_CODES_MScThesis\CODES\PH0__Allowables_and_Failure_Scenarios\Static_TEST__Data-Processing_UPDATED\force_displacement_shell_elastic.dat"

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


def plot_reference_curve(filepath, color="red", linewidth=2, label="Reference curve",
                          x_scale=1.0, y_scale=1000.0):
    """Parses an 'XYDATA, Curve N' style .dat file (skips any line that isn't
    exactly two whitespace-separated floats), fits a linear regression to it,
    prints the gradient, plots BOTH the raw curve and the fitted line, on the
    CURRENT axes -- call this after the phial curves are plotted but before
    plt.legend(), so it shows up in the same legend.
    x_scale/y_scale are plain multipliers, in case the .dat file turns out
    to be in different units than mm/N -- 1.0 (no change) until you know."""
    xs, ys = [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    xs = np.array(xs) * x_scale
    ys = np.array(ys) * y_scale

    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"Reference curve linear fit: gradient = {slope:.6g} N/mm  intercept = {intercept:.6g}")

    plt.plot(xs, ys, color=color, linewidth=linewidth, label=label)
    x_fit = np.array([xs.min(), xs.max()])
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.5,
              label=f"Linear fit (gradient={slope:.4g} N/mm)")


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

    plt.figure(figsize=(8, 6))
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

    plot_reference_curve(REFERENCE_CURVE)

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.title("Stiffness comparison — all phials aligned to climb start")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"\nSaved overlay plot: {OUTPUT_PLOT}")

    


if __name__ == "__main__":
    main()