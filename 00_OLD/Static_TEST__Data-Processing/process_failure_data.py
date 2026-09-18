"""
Second-stage processing: reads the per-phial .xlsx curves produced by
extract_phial_curves.py, and produces:

1. One load-displacement plot per phial (toggle-able with PLOT_ALL_CURVES,
   so you can turn plotting off on re-runs once you've already reviewed them).
   Each plot marks every detected candidate peak (hollow marker) and the
   FIRST one in a solid red marker, so you can visually sanity-check the
   detection algorithm across all 76 phials at a glance.
2. A summary Excel (SUMMARY_EXCEL) with one row per phial: Phial_Number,
   Batch, Loading_Speed_mm_per_min, Breaking_Mode, and Failure_Load_N (the
   FIRST break, not the curve's maximum).

See failure_extraction.py's module docstring for why "first break" needs a
prominence + smoothing based detector rather than a naive local-max check,
and why the defaults here are set the way they are.

QA COLUMNS (extra, beyond what you asked for):
Num_Peaks_Detected / All_Peak_Loads_N / Flag are included so you can spot-
check questionable phials without opening every plot. A Flag is set when no
clean peak was found (falls back to the curve's global max) or when very
few data points were available. Delete these columns from the output if you
don't want them — they don't affect Failure_Load_N.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from failure_extraction import find_first_break

# ============================== CONFIGURATION ==============================
CURVES_DIR = "C:\Temp\output_curves"         # folder of per-phial xlsx from step 1
SUMMARY_EXCEL = "C:\Temp\phial_failure_summary.xlsx"

PLOT_ALL_CURVES = True                # set False to skip plotting on re-runs
PLOTS_DIR = "C:\Temp\curve_plots"

# --- peak-detection parameters (see failure_extraction.py for rationale) ---
SMOOTH_WINDOW = 11
PROMINENCE_ABS_N = 22
PROMINENCE_FRACTION = 0.0
MIN_PEAK_LOAD_N = 2.0
REFINE_RADIUS = 5
# ============================================================================


def process_one_file(filepath):
    data = pd.read_excel(filepath, sheet_name="Data")
    meta = pd.read_excel(filepath, sheet_name="Metadata")
    meta_dict = dict(zip(meta["Field"], meta["Value"]))

    result = find_first_break(
        data["Time_s"].values,
        data["Load_N"].values,
        smooth_window=SMOOTH_WINDOW,
        prominence_abs_n=PROMINENCE_ABS_N,
        prominence_fraction=PROMINENCE_FRACTION,
        min_peak_load_n=MIN_PEAK_LOAD_N,
        refine_radius=REFINE_RADIUS,
    )

    if PLOT_ALL_CURVES:
        plot_curve(filepath, data, meta_dict, result)

    return {
        "Phial_Number": meta_dict.get("Phial number"),
        "Batch": meta_dict.get("Batch"),
        "Loading_Speed_mm_per_min": meta_dict.get("Speed_mm_per_min"),
        "Breaking_Mode": meta_dict.get("Breaking_mode"),
        "Failure_Load_N": result["first_break_load_n"],
        "Failure_Displacement_mm": data["Extension_mm"].iloc[result["first_break_index"]]
            if result["first_break_index"] is not None else None,
        "Num_Peaks_Detected": result["num_peaks_detected"],
        "All_Peak_Loads_N": ", ".join(f"{v:.2f}" for v in result["all_peak_loads_n"]),
        "Flag": result["flag"],
    }


def plot_curve(filepath, data, meta_dict, result):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]

    plt.figure(figsize=(7, 5))
    plt.plot(data["Extension_mm"], data["Load_N"], linewidth=1)

    # mark every detected candidate peak (QA visibility)
    for load_val in result["all_peak_loads_n"]:
        idx = (data["Load_N"] - load_val).abs().idxmin()
        plt.plot(data["Extension_mm"].iloc[idx], load_val, "o",
                  markerfacecolor="none", markeredgecolor="orange", markersize=8)

    # highlight the FIRST break specifically
    if result["first_break_index"] is not None:
        idx = result["first_break_index"]
        plt.plot(data["Extension_mm"].iloc[idx], data["Load_N"].iloc[idx],
                  "o", color="red", markersize=9, label="First break")
        plt.legend()

    title = f"Phial {meta_dict.get('Phial number')}{meta_dict.get('Batch')} — " \
            f"{meta_dict.get('Speed_mm_per_min')} mm/min — mode {meta_dict.get('Breaking_mode')}"
    if result["flag"]:
        title += f"\n[{result['flag']}]"
    plt.title(title, fontsize=9)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base}.png"), dpi=150)
    plt.close()


def main():
    filepaths = sorted(glob.glob(os.path.join(CURVES_DIR, "*.xlsx")))
    print(f"Found {len(filepaths)} phial curve files in '{CURVES_DIR}/'.")

    rows = []
    for fp in filepaths:
        rows.append(process_one_file(fp))

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["Batch", "Phial_Number"]).reset_index(drop=True)
    summary.to_excel(SUMMARY_EXCEL, index=False)

    flagged = summary[summary["Flag"] != ""]
    print(f"Wrote {len(summary)} rows to '{SUMMARY_EXCEL}'.")
    if len(flagged):
        print(f"\n{len(flagged)} phial(s) flagged for manual review:")
        print(flagged[["Phial_Number", "Batch", "Flag"]].to_string(index=False))

    if PLOT_ALL_CURVES:
        print(f"Plots saved to '{PLOTS_DIR}/'.")


if __name__ == "__main__":
    main()
