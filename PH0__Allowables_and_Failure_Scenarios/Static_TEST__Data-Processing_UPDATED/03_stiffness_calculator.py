"""
Computes the tangent (elastic) modulus -- the slope of the steady, linear
pre-rupture climb -- for every phial curve, and marks the fitted region on a
QA plot per phial.

THE CORE PROBLEM
-----------------
A load-displacement curve isn't linear everywhere: there's usually a low-load
"toe" region first (grip slack / specimen seating), then a genuinely linear
climb, then possibly some softening right before failure. A plain derivative
would be garbage in the toe region and noisy everywhere else. We need to find
WHICH stretch of the curve is actually linear, not just differentiate blindly.

METHOD: sliding-window R^2-maximizing search
---------------------------------------------
For every candidate window [i, j] of the pre-break data, fit a straight
line and compute R^2. Keep the LONGEST window whose R^2 clears
R2_THRESHOLD. This directly optimizes for "a stretch trustworthy enough to
call a modulus" rather than relying on a proxy signal like curvature -- and
it naturally excludes both the toe (poor R^2 if included) and any
pre-failure softening (same reason), without hand-tuned cutoffs for either
end of the window.

Made fast via prefix sums: the regression slope/intercept/R^2 for ANY window
can be computed in O(1) from four running sums, rather than re-summing the
window's points from scratch every time. Validated against numpy.polyfit
and against synthetic toe+linear+softening test curves during development
(recovers the true window boundaries and slope to within noise).

Only ever searches BEFORE the first-break index from find_first_break --
anything at or after that is post-failure and not physically meaningful for
a modulus.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from failure_extraction import find_first_break

# ============================== CONFIGURATION ==============================
CURVES_DIR = "output_curves"
SUMMARY_EXCEL = "phial_stiffness_summary.xlsx"
CLEAN_MODULUS_EXCEL = "phial_modulus_clean.xlsx"

PLOT_ALL_CURVES = True
PLOTS_DIR = "modulus_plots"

MIN_POINTS = 20          # shortest stretch allowed to count as "the linear region"
R2_THRESHOLD = 0.999     # how linear a window must be to qualify

# --- first-break detection parameters, same as 02_process_failure_data.py ---
SMOOTH_WINDOW = 11
PROMINENCE_ABS_N = 22
PROMINENCE_FRACTION = 0.0
MIN_PEAK_LOAD_N = 2.0
REFINE_RADIUS = 5
# ============================================================================


def _window_stats_fn(x, y):
    """Prefix-sum closure giving O(1) (slope, intercept, r2) for any window."""
    Sx = np.concatenate([[0], np.cumsum(x)])
    Sy = np.concatenate([[0], np.cumsum(y)])
    Sxx = np.concatenate([[0], np.cumsum(x * x)])
    Sxy = np.concatenate([[0], np.cumsum(x * y)])
    Syy = np.concatenate([[0], np.cumsum(y * y)])

    def stats(i, j):  # inclusive indices
        k = j - i + 1
        sx, sy = Sx[j + 1] - Sx[i], Sy[j + 1] - Sy[i]
        sxx, sxy, syy = Sxx[j + 1] - Sxx[i], Sxy[j + 1] - Sxy[i], Syy[j + 1] - Syy[i]
        denom_slope = k * sxx - sx ** 2
        if denom_slope <= 0:
            return None
        slope = (k * sxy - sx * sy) / denom_slope
        intercept = (sy - slope * sx) / k
        denom_r2 = (k * sxx - sx ** 2) * (k * syy - sy ** 2)
        r2 = ((k * sxy - sx * sy) ** 2 / denom_r2) if denom_r2 > 0 else 1.0
        return slope, intercept, r2

    return stats


def best_linear_window(x, y, min_points=MIN_POINTS, r2_threshold=R2_THRESHOLD):
    """Longest window meeting r2_threshold; falls back to the best-R2 window
    at the minimum length (flagged) if nothing clears the bar."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    stats = _window_stats_fn(x, y)

    best = None  # (length, i, j, slope, intercept, r2)
    for i in range(n - min_points + 1):
        for j in range(i + min_points - 1, n):
            res = stats(i, j)
            if res is None:
                continue
            slope, intercept, r2 = res
            if r2 >= r2_threshold:
                length = j - i + 1
                if best is None or length > best[0]:
                    best = (length, i, j, slope, intercept, r2)

    if best is not None:
        length, i, j, slope, intercept, r2 = best
        return dict(start_idx=i, end_idx=j, n_points=length,
                    slope=slope, intercept=intercept, r2=r2, flag="")

    best_r2 = -1
    fallback = None
    for i in range(n - min_points + 1):
        j = i + min_points - 1
        res = stats(i, j)
        if res is not None and res[2] > best_r2:
            fallback = (i, j, *res)
            best_r2 = res[2]
    if fallback is None:
        return dict(start_idx=None, end_idx=None, n_points=0,
                    slope=None, intercept=None, r2=None,
                    flag="NO_VALID_WINDOW - verify manually")
    i, j, slope, intercept, r2 = fallback
    return dict(start_idx=i, end_idx=j, n_points=j - i + 1,
                slope=slope, intercept=intercept, r2=r2,
                flag="NO_WINDOW_MET_R2_THRESHOLD - relaxed to best available, verify manually")


def process_one_file(filepath):
    data = pd.read_excel(filepath, sheet_name="Data")
    meta = pd.read_excel(filepath, sheet_name="Metadata")
    meta_dict = dict(zip(meta["Field"], meta["Value"]))

    break_result = find_first_break(
        data["Time_s"].values, data["Load_N"].values,
        smooth_window=SMOOTH_WINDOW, prominence_abs_n=PROMINENCE_ABS_N,
        prominence_fraction=PROMINENCE_FRACTION, min_peak_load_n=MIN_PEAK_LOAD_N,
        refine_radius=REFINE_RADIUS,
    )
    break_idx = break_result["first_break_index"]

    disp = data["Extension_mm"].values[:break_idx]
    load = data["Load_N"].values[:break_idx]

    if len(disp) < MIN_POINTS:
        window = dict(start_idx=None, end_idx=None, n_points=0, slope=None,
                      intercept=None, r2=None, flag="TOO_FEW_PRE-BREAK_POINTS")
    else:
        window = best_linear_window(disp, load)

    if PLOT_ALL_CURVES:
        plot_curve(filepath, data, meta_dict, break_result, window)

    return {
        "Phial_Number": meta_dict.get("Phial number"),
        "Batch": meta_dict.get("Batch"),
        "Loading_Speed_mm_per_min": meta_dict.get("Speed_mm_per_min"),
        "Breaking_Mode": meta_dict.get("Breaking_mode"),
        "Tangent_Modulus_N_per_mm": window["slope"],
        "Fit_Intercept_N": window["intercept"],
        "Fit_R2": window["r2"],
        "Window_Start_mm": disp[window["start_idx"]] if window["start_idx"] is not None else None,
        "Window_End_mm": disp[window["end_idx"]] if window["end_idx"] is not None else None,
        "N_Points_Used": window["n_points"],
        "Flag": window["flag"],
    }


def plot_curve(filepath, data, meta_dict, break_result, window):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]

    plt.figure(figsize=(7, 5))
    plt.plot(data["Extension_mm"], data["Load_N"], linewidth=1, color="steelblue")

    if window["start_idx"] is not None:
        disp = data["Extension_mm"].values
        i, j = window["start_idx"], window["end_idx"]
        x_fit = disp[i:j + 1]
        y_fit = window["slope"] * x_fit + window["intercept"]
        plt.plot(x_fit, y_fit, color="orange", linewidth=3,
                  label=f"Tangent modulus = {window['slope']:.1f} N/mm (R²={window['r2']:.4f})")

    if break_result["first_break_index"] is not None:
        idx = break_result["first_break_index"]
        plt.plot(data["Extension_mm"].iloc[idx], data["Load_N"].iloc[idx],
                  "o", color="red", markersize=9, label="First break")

    title = f"Phial {meta_dict.get('Phial number')}{meta_dict.get('Batch')} — " \
            f"{meta_dict.get('Speed_mm_per_min')} mm/min — mode {meta_dict.get('Breaking_mode')}"
    if window["flag"]:
        title += f"\n[{window['flag']}]"
    plt.title(title, fontsize=9)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base}.png"), dpi=150)
    plt.close()


def main():
    filepaths = sorted(glob.glob(os.path.join(CURVES_DIR, "*.xlsx")))
    print(f"Found {len(filepaths)} phial curve files in '{CURVES_DIR}/'.")

    rows = [process_one_file(fp) for fp in filepaths]

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["Batch", "Phial_Number"]).reset_index(drop=True)
    summary.to_excel(SUMMARY_EXCEL, index=False)

    flagged = summary[summary["Flag"] != ""]
    print(f"Wrote {len(summary)} rows to '{SUMMARY_EXCEL}'.")
    if len(flagged):
        print(f"\n{len(flagged)} phial(s) flagged for manual review:")
        print(flagged[["Phial_Number", "Batch", "Flag"]].to_string(index=False))

    # "Flag" is empty string "" in memory right now, but an empty string
    # written to Excel round-trips back as a blank cell (NaN) -- so we check
    # for BOTH here with fillna(""), rather than a bare == "" that would only
    # work today and silently break if this ever runs on a reloaded file.
    clean = summary[summary["Flag"].fillna("") == ""].drop(columns=["Flag"])
    clean.to_excel(CLEAN_MODULUS_EXCEL, index=False)
    print(f"Wrote {len(clean)} clean (unflagged) rows to '{CLEAN_MODULUS_EXCEL}'.")

    if PLOT_ALL_CURVES:
        print(f"Plots saved to '{PLOTS_DIR}/'.")


if __name__ == "__main__":
    main()