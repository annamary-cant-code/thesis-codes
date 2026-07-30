"""
Standalone: reads a single X/Y curve from a .dat file (format: one header
line like "XYDATA, Curve 1", then whitespace-separated "X   Y" numeric rows),
plots it, fits a straight line across the whole curve, prints the gradient,
and draws the fit on top of the data.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
DAT_FILE = r"C:\DATI\Linear_regression_analyser\export_050mm.dat"
OUTPUT_PLOT = "curve_fit.png"
# ============================================================================


def parse_xydata_dat(filepath):
    """Keeps only lines that are exactly two whitespace-separated floats;
    skips headers, labels, and blank lines regardless of their exact form."""
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
    return np.array(xs), np.array(ys)


def main():
    x, y = parse_xydata_dat(DAT_FILE)
    print(f"Parsed {len(x)} points from '{DAT_FILE}'.")

    slope, intercept = np.polyfit(x, y, 1)
    print(f"Linear fit: gradient = {slope:.6g}   intercept = {intercept:.6g}")

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, color="red", linewidth=2, label="Curve")

    x_fit = np.array([x.min(), x.max()])
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.5,
              label=f"Linear fit (gradient={slope:.4g})")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Curve with linear regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()