"""
Fits a 2-parameter Weibull distribution to failure loads, separately for each
batch (A and B), and plots a frequency-distribution graph per batch:
  - histogram of the actual measured failure loads (density-scaled)
  - the fitted Weibull curve overlaid as a smooth continuous interpolation
  - vertical lines marking the 1st and 10th percentile of the FITTED curve

IMPORTANT LABELING NOTE (read before using these numbers for anything formal):
The two marked values are the 1st/10th percentile of the fitted Weibull
distribution itself -- i.e. "1% / 10% of this fitted curve lies below this
load". They are NOT A-basis/B-basis allowables. True A/B-basis values also
account for the uncertainty in the fit itself (since it came from a limited
sample) and are always more conservative (lower) than these percentiles.
That's deliberately left out here per your request for the simple version --
just don't call these numbers "A-basis"/"B-basis" in a report without
re-adding that step.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
SUMMARY_EXCEL = "C:\Temp\phial_failure_summary.xlsx"   # output of process_failure_data.py
OUTPUT_DIR = "C:\Temp\weibull_plots"
PERCENTILES = [(0.01, "1st percentile"), (0.05, "5th percentile")]
N_CURVE_POINTS = 500
# ============================================================================


def fit_and_plot(batch_label, group, out_dir):
    loads = group["Failure_Load_N"].values.astype(float)
    speeds = group["Loading_Speed_mm_per_min"].values
    n = len(loads)

    beta, _, alpha = stats.weibull_min.fit(loads, floc=0)

    x_curve = np.linspace(0, loads.max() * 1.3, N_CURVE_POINTS)
    pdf_curve = stats.weibull_min.pdf(x_curve, c=beta, scale=alpha)

    # Two stacked panels sharing the x-axis: distribution on top, individual
    # points (jittered, colored by speed) in a thin strip underneath.
    fig, (ax_hist, ax_strip) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        constrained_layout=True,
    )

    # ax_hist.hist(loads, bins="auto", density=True, alpha=0.5,
    #              color="steelblue", edgecolor="white", label=f"Measured failure loads (n={n})")
    ax_hist.plot(x_curve, pdf_curve, color="black", linewidth=2,
                 label=f"Fitted Weibull (β={beta:.2f}, α={alpha:.1f})")

    pct_colors = ["crimson", "darkorange"]
    for (p, label), color in zip(PERCENTILES, pct_colors):
        val = stats.weibull_min.ppf(p, c=beta, scale=alpha)
        ax_hist.axvline(val, color=color, linestyle="--", linewidth=1.8,
                         label=f"{label} = {val:.1f} N")
        ax_strip.axvline(val, color=color, linestyle="--", linewidth=1.8)

    # Individual points, one scatter call per speed so each gets its own
    # color and its own automatic legend entry. All sit on one flat line —
    # no jitter, since real load values are unlikely to coincide closely.
    unique_speeds = sorted(pd.unique(speeds))
    speed_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for speed_val, color in zip(unique_speeds, speed_colors):
        mask = speeds == speed_val
        ax_strip.scatter(loads[mask], np.zeros(mask.sum()), color=color, s=28,
                          alpha=0.8, edgecolor="black", linewidth=0.4,
                          label=f"{speed_val} mm/min (n={mask.sum()})")

    ax_strip.set_yticks([])
    ax_strip.set_ylim(-0.6, 0.6)
    ax_strip.set_xlabel("Failure load (N)")
    ax_strip.legend(loc="upper right", fontsize=8, ncol=len(unique_speeds))
    ax_strip.grid(True, axis="x", alpha=0.3)

    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title(f"Batch {batch_label} — failure load distribution (n={n})")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, f"weibull_batch_{batch_label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "Batch": batch_label,
        "n": n,
        "beta_shape": beta,
        "alpha_scale": alpha,
        "P1_percentile_N": stats.weibull_min.ppf(0.01, c=beta, scale=alpha),
        "P5_percentile_N": stats.weibull_min.ppf(0.05, c=beta, scale=alpha),
        "plot_file": out_path,
    }


def plot_combined(df, out_dir):
    """
    One figure combining both batches: fitted Weibull curves overlaid on top
    (no histogram -- two overlaid histograms get messy), and a strip of every
    individual point underneath. Since there are now two independent
    categorical variables (batch, speed) instead of one, the strip encodes
    them on two separate visual channels rather than as flat combined
    categories: marker SHAPE = batch, marker COLOR = speed. That way you can
    read "which batch" and "which speed" independently instead of memorizing
    arbitrary combined swatches.
    """
    batches = sorted(df["Batch"].unique())
    all_speeds = sorted(df["Loading_Speed_mm_per_min"].unique())
    max_load = df["Failure_Load_N"].max()
 
    batch_curve_colors = ["tab:orange", "tab:cyan", "gold", "magenta"]  # vivid, chosen to avoid the speed palette's hues below
    batch_curve_styles = ["-", "--", "-.", ":"]
    batch_markers = ["o", "s", "^", "D"]
    speed_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]
    speed_color_map = {s: c for s, c in zip(all_speeds, speed_colors)}
 
    fig, (ax_top, ax_strip) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        constrained_layout=True,
    )
 
    x_curve = np.linspace(0, max_load * 1.3, N_CURVE_POINTS)
 
    n_batches = len(batches)
    batch_row_y = {b: (n_batches - 1 - i) for i, b in enumerate(batches)}  # first batch on top
 
    for batch_label, curve_color, curve_style, marker in zip(batches, batch_curve_colors, batch_curve_styles, batch_markers):
        group = df[df["Batch"] == batch_label]
        loads = group["Failure_Load_N"].values.astype(float)
        speeds = group["Loading_Speed_mm_per_min"].values
        n = len(loads)
        row_y = batch_row_y[batch_label]
 
        beta, _, alpha = stats.weibull_min.fit(loads, floc=0)
        pdf_curve = stats.weibull_min.pdf(x_curve, c=beta, scale=alpha)
        ax_top.plot(x_curve, pdf_curve, color=curve_color, linestyle=curve_style, linewidth=2,
                    label=f"Batch {batch_label} Weibull (β={beta:.2f}, α={alpha:.1f}, n={n})")
 
        linestyles = ["--", ":"]
        for (p, label), ls in zip(PERCENTILES, linestyles):
            val = stats.weibull_min.ppf(p, c=beta, scale=alpha)
            ax_top.axvline(val, color=curve_color, linestyle=ls, linewidth=1.6,
                            label=f"Batch {batch_label} {label} = {val:.1f} N")
            ax_strip.axvline(val, color=curve_color, linestyle=ls, linewidth=1.6)
 
        # one scatter call per (batch, speed) combo actually present, so shape
        # (this batch) x color (that speed) each get one clean legend entry.
        # All of this batch's points sit on its own row (row_y), so batches
        # never overlap vertically -- shape still separately confirms batch.
        for speed_val in sorted(pd.unique(speeds)):
            mask = speeds == speed_val
            ax_strip.scatter(loads[mask], np.full(mask.sum(), row_y),
                              marker=marker, color=speed_color_map[speed_val],
                              s=32, alpha=0.85, edgecolor="black", linewidth=0.4,
                              label=f"Batch {batch_label} · {speed_val} mm/min (n={mask.sum()})")
 
    ax_strip.set_yticks(list(batch_row_y.values()))
    ax_strip.set_yticklabels([f"Batch {b}" for b in batch_row_y.keys()])
    ax_strip.set_ylim(-0.7, n_batches - 1 + 0.7)
    ax_strip.set_xlabel("Failure load (N)")
    ax_strip.legend(loc="upper right", fontsize=7, ncol=2)
    ax_strip.grid(True, axis="x", alpha=0.3)
 
    ax_top.set_ylabel("Probability density")
    ax_top.set_title("Batch A vs Batch B — failure load distributions")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)
 
    out_path = os.path.join(out_dir, "weibull_combined.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Combined plot saved: {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_excel(SUMMARY_EXCEL)

    results = []
    for batch_label, group in df.groupby("Batch"):
        results.append(fit_and_plot(batch_label, group, OUTPUT_DIR))
        print(f"Batch {batch_label}: saved {results[-1]['plot_file']}")

    summary = pd.DataFrame(results)
    print("\n", summary.to_string(index=False))
    summary.to_excel(os.path.join(OUTPUT_DIR, "weibull_fit_summary.xlsx"), index=False)

    plot_combined(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()