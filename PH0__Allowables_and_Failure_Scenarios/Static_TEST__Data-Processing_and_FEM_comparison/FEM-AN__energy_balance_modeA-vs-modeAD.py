"""
FEM-AN__energy_balance_modeA-vs-modeAD.py

Plots the global energy balance (GLSTAT output) of the two FEM runs -- mode A
(dome) and mode AD (dome + base) -- as TWO separate figures, one per mode,
saved into the energy_balance folder next to the data they are built from.

Each figure shows, against time:
    external work, internal energy, kinetic energy, sliding interface energy
    and total energy.

UNITS. The model runs in the consistent set ms / mm / kg / kN / GPa, so the
exports are already in:
    time    [ms]
    energy  [J]   (= kN*mm)
and are plotted as they are, with no rescaling.

The two figures share their axis limits, so when they sit one above the other
the two modes can be compared directly.

INPUT FILES
-----------
energy_balance/modeA_energyBalance.csv
energy_balance/modeAD_energyBalance.csv

GLSTAT CSV exports: a title line (",Glstat Data"), then the column header
(Time, external_work, internal_energy, kinetic_energy,
sliding_interface_energy, total_energy) and one row per output step. Each row
ends in a trailing comma, which shows up as an empty extra column and is
dropped on load.

USAGE:
    python FEM-AN__energy_balance_modeA-vs-modeAD.py
    python FEM-AN__energy_balance_modeA-vs-modeAD.py --no-show
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
HERE = Path(__file__).parent
ENERGY_FOLDER = HERE / "energy_balance"

# One run per figure: (mode code, mode name, input CSV, output PNG).
RUNS = [
    ("A",  "Dome",        "modeA_energyBalance.csv",  "energy_balance_modeA.png"),
    ("AD", "Dome + Base", "modeAD_energyBalance.csv", "energy_balance_modeAD.png"),
]

# Wide and flat, like the FT-finetuning overlays, so the two figures stack one
# above the other on a page.
FIG_SIZE = (10, 4.5)

# The energy terms drawn, in legend order: column -> (label, colour, style, width).
# Same colour and style for a term in both figures.
# Muted palette for print: the balance pair (external work vs. total energy)
# in near-black, colour kept for the terms that make the total up.
ENERGY_TERMS = {
    "external_work":            ("External work",            "#222222", "--", 1.8),
    "total_energy":             ("Total energy",             "#222222", "-",  2.4),
    "internal_energy":          ("Internal energy",          "#2E6F9E", "-",  1.6),   # steel blue
    "kinetic_energy":           ("Kinetic energy",           "#E09F3E", "-",  1.6),   # ochre
    "sliding_interface_energy": ("Sliding interface energy", "#9E2A2B", "-",  1.6),   # brick red
}

# EVENTS marked on each mode's figure, at the output step nearest each time:
# (time [ms], box text, box offset (dx, dy) [points], box alignment). Set by
# hand. Mode A:
#   3.80 ms -- external work and total energy first differ by more than 2%
#              (and stay apart from there on)
#   4.30 ms -- the failure load in the force-displacement curve (96.9 N,
#              A_060), after which the total energy stops rising
# Mode AD (loads from AD_060):
#   4.10 ms -- first break, the base (68.8 N); barely visible in the energies
#   5.35 ms -- second break, the dome (99.1 N), the peak force
#   5.45 ms -- external work and total energy first differ by more than 2%,
#              one step after the dome break
# A mode left out gets no markers.
EVENT_MARKERS = {
    "A": [
        (3.80, "Energies separate (> 2%)", (-30, 40), "right"),
        (4.30, "Failure load (96.9 N)",    (35, -30), "left"),
    ],
    "AD": [
        (4.10, "Base break (68.8 N)",      (-30, 40), "right"),
        (5.35, "Dome break (99.1 N)",      (-40, 45), "right"),
        (5.45, "Energies separate (> 2%)", (35, -50), "left"),
    ],
}
EVENT_COLOR = "#222222"
# ============================================================================


def load_energy_balance(path):
    """Read one GLSTAT CSV into a DataFrame: title line skipped, trailing empty
    column dropped, header names stripped of whitespace."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing energy balance export '{path.name}' in {path.parent}.")
    data = pd.read_csv(path, skiprows=1)
    data.columns = data.columns.str.strip()
    data = data.loc[:, [c for c in data.columns if c and not c.startswith("Unnamed")]]
    missing = [c for c in ["Time", *ENERGY_TERMS] if c not in data.columns]
    if missing:
        raise ValueError(f"'{path.name}' has no column(s) {missing}; its columns are "
                         f"{list(data.columns)}.")
    return data


def nearest_step(data, t):
    """Index of the output step nearest to time t [ms]."""
    return int(np.argmin(np.abs(data["Time"].to_numpy() - t)))


def plot_energy_balance(code, name, data, xlim, ylim, out_path, events=()):
    """One mode's energy balance against time, with each event in `events`
    -- (step index, box text, offset, alignment) -- marked on it."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for column, (label, color, style, width) in ENERGY_TERMS.items():
        ax.plot(data["Time"], data[column], color=color, linestyle=style, linewidth=width,
                label=label)

    for idx, text, (dx, dy), ha in events:
        t_e = float(data["Time"].iloc[idx])
        # on the total energy: the curve that departs from the external work
        e_e = float(data["total_energy"].iloc[idx])
        ax.axvline(t_e, color=EVENT_COLOR, lw=0.9, ls=":", zorder=2)
        ax.scatter(t_e, e_e, color=EVENT_COLOR, marker="o", s=30, zorder=6,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(f"{text}\nt = {t_e:.2f} ms",
                    xy=(t_e, e_e), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=EVENT_COLOR, ha=ha, multialignment="left", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=EVENT_COLOR, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=EVENT_COLOR, lw=1.1))

    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_axisbelow(True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Energy [J]")
    ax.set_title(f"FEM energy balance - Mode {code} ({name})")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    print(f"  Saved plot to {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot the FEM global energy balance of the mode A and mode AD runs, one "
                    "figure per mode, saved into the energy_balance folder.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the figures without opening them.")
    args = parser.parse_args()

    runs = []
    for code, name, csv_name, png_name in RUNS:
        data = load_energy_balance(ENERGY_FOLDER / csv_name)
        runs.append((code, name, data, ENERGY_FOLDER / png_name))
        final = data.iloc[-1]
        print(f"Mode {code} ({name}): {len(data)} steps, t = 0 - {final['Time']:g} ms")
        print(f"    at the end: external work {final['external_work']:.4e} J, "
              f"total energy {final['total_energy']:.4e} J "
              f"(difference {final['external_work'] - final['total_energy']:+.4e} J)")

    # shared limits, so the two stacked figures read against each other
    t_max = max(float(d["Time"].max()) for _, _, d, _ in runs)
    e_max = max(float(d[list(ENERGY_TERMS)].max().max()) for _, _, d, _ in runs)
    e_min = min(float(d[list(ENERGY_TERMS)].min().min()) for _, _, d, _ in runs)
    pad = 0.05 * (e_max - min(e_min, 0.0))
    xlim = (0.0, t_max)
    ylim = (min(e_min, 0.0) - pad, e_max + pad)

    for code, name, data, out_path in runs:
        events = []
        for t_event, text, offset, ha in EVENT_MARKERS.get(code, []):
            idx = nearest_step(data, t_event)
            row = data.iloc[idx]
            gap = (row["external_work"] - row["total_energy"]) / row["external_work"]
            print(f"  Mode {code} '{text}': t = {row['Time']:.4f} ms, "
                  f"external work = {row['external_work']:.4e} J, "
                  f"total energy = {row['total_energy']:.4e} J (gap {gap:.1%})")
            events.append((idx, text, offset, ha))
        plot_energy_balance(code, name, data, xlim, ylim, out_path, events)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
