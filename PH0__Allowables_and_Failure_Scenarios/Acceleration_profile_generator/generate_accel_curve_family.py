"""
Generate a family of piecewise-linear (trapezoidal) acceleration-time curves
that all dissipate the same delta-v (i.e. equal area under a-t), write them as
*DEFINE_CURVE input for LS-DYNA, and plot them.

Unit system: mm, ms, kg, kN, GPa  =>  velocity [mm/ms], acceleration [mm/ms^2]

Curve shape (4 points, matches the sketch):
    (0, 0) -> (t1, a_peak) -> (t2, a_peak) -> (t3, 0)
Rise (0->t1) and fall (t2->t3) are symmetric: each is RISE_FRACTION * t_total.
Plateau length = (1 - 2*RISE_FRACTION) * t_total.

Derivation:
    delta_v   = sqrt(vx0^2 + vy0^2)                       [mm/ms]
    area      = 0.5 * (t3 + (t2 - t1)) * a_peak = delta_v
    with t1 = f*t_total, t2 = (1-f)*t_total, t3 = t_total, f = RISE_FRACTION:
    area = t_total * (1 - f) * a_peak = delta_v
    =>  t_total = delta_v / (a_peak * (1 - f))

NOTE ON PICKING a_peak
----------------------
The duration is NOT a free parameter: it falls out of delta_v and a_peak by the
relation above. Choosing a_peak is really choosing how long the pulse lasts:

    a_peak = delta_v / (t_total * (1 - RISE_FRACTION))

The phenomenon lasts PHENOMENON_DURATION_MS, so a pulse that exactly fills the
window needs a_peak = delta_v / (PHENOMENON_DURATION_MS * (1 - f)) -- that is
the SMALLEST admissible peak. The values in PEAK_ACCELERATIONS bracket it from
above so every curve fits inside the window; anything below it is rejected by
build_trapezoid. Each curve's duration is printed as a percentage of the window
so this stays visible.

INPUT UNITS
-----------
PEAK_ACCELERATIONS may be typed in [g] or in [mm/ms^2]; set INPUT_UNITS to say
which. The conversion is 1 g = 9.80665 m/s^2 = 9.80665e-3 mm/ms^2 in this unit
system (1 m/s^2 = 1000 mm / (1000 ms)^2 = 1e-3 mm/ms^2). Typing in g is usually
what you want, since payload tolerance limits are quoted in g.

Whatever the input, the curve files are ALWAYS written in mm/ms^2 -- that is
what the solver reads -- and each file carries a '$' comment header recording
its a_peak in both units. Filenames are tagged with the value as typed plus its
unit ('..._apeak_20p390g.k' vs '..._apeak_0p200mmms2.k') so the two cannot
be confused on disk.

OUTPUT FORMAT
-------------
Each file is a self-contained LS-DYNA keyword block (*DEFINE_CURVE with the
HyperMesh $HMNAME/$HWCOLOR/$HMCURVE header lines) that can be pulled straight
into a model with *INCLUDE. By default every file uses the same LCID
(CURVE_ID_START), so swapping the include swaps the pulse without touching the
rest of the deck. Set UNIQUE_CURVE_IDS = True to give each curve its own LCID
if several are to be included in the same model.

Two plots are produced of the same family, one in [mm/ms^2] and one in [g],
irrespective of INPUT_UNITS.
"""

import os
import time
import math

import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Initial velocity components to be dissipated [mm/ms]
VX0 = 17.2
VY0 = 5.0

# Duration of the phenomenon [ms]. Every curve must fit inside this window.
PHENOMENON_DURATION_MS = 200.0

# Units the PEAK_ACCELERATIONS below are written in: "g" or "mm/ms^2".
# Everything downstream (curve files, plots) is always in the solver's
# mm/ms^2 regardless -- this only says how YOUR input is to be read.
INPUT_UNITS = "g"

# Family of peak accelerations to generate curves for, in INPUT_UNITS.
# See "NOTE ON PICKING a_peak" above: these set the pulse durations.
PEAK_ACCELERATIONS = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Fraction of total curve duration used for rise (and, symmetrically, fall)
# Must be in (0, 0.5). 0.5 => pure triangle, no plateau.
RISE_FRACTION = 0.2

# Output location and naming
OUTPUT_DIR = "curve_family_output"
FILE_PREFIX = "accel_curve"
FILE_EXTENSION = ".k"

# --- *DEFINE_CURVE header -----------------------------------------------------
# LCID written in the keyword. False => every file gets CURVE_ID_START (one
# include per run, the model always references the same curve). True => IDs
# CURVE_ID_START, CURVE_ID_START+1, ... (several includes in one model).
CURVE_ID_START = 2
UNIQUE_CURVE_IDS = False
# Curve title shown in HyperMesh ($HMNAME); the a_peak tag is appended to it.
CURVE_NAME_PREFIX = "curve_acceleration_SLED"
# HyperMesh display color ($HWCOLOR)
HW_COLOR = 24
# Scale factors on abscissa / ordinate (SFA / SFO)
SFA = 1.0
SFO = 1.0

# Decimal precision used inside the fixed-width fields
DECIMALS = 6

# --- plotting ----------------------------------------------------------------
MAKE_PLOTS = True
PLOT_MM_MS2 = "accel_curve_family_mm_per_ms2.png"
PLOT_G = "accel_curve_family_g.png"
DPI = 150
FIGSIZE = (8, 5)

# x-axis span: True => always show the full phenomenon window (pulses in
# context), False => auto-fit to the pulses themselves.
PLOT_FULL_WINDOW = True

# Diagnostic printout of computed t1/t2/t3/area per curve
VERBOSE = True

# Number of retries for directory creation (OneDrive sync race guard)
MKDIR_RETRIES = 5
MKDIR_RETRY_DELAY_S = 0.2

# 1 g expressed in the solver's acceleration unit [mm/ms^2]
G_IN_MM_PER_MS2 = 9.80665e-3

# =============================================================================


def to_mm_per_ms2(value, units):
    """Convert an acceleration given in `units` to the solver's mm/ms^2."""
    if units == "mm/ms^2":
        return value
    if units == "g":
        return value * G_IN_MM_PER_MS2
    raise ValueError(f"INPUT_UNITS must be 'g' or 'mm/ms^2', got {units!r}")


def from_mm_per_ms2(value, units):
    """Convert an acceleration in mm/ms^2 back into `units`."""
    if units == "mm/ms^2":
        return value
    if units == "g":
        return value / G_IN_MM_PER_MS2
    raise ValueError(f"INPUT_UNITS must be 'g' or 'mm/ms^2', got {units!r}")


def make_output_dir(path):
    for attempt in range(MKDIR_RETRIES):
        try:
            os.makedirs(path, exist_ok=True)
            return
        except OSError:
            if attempt == MKDIR_RETRIES - 1:
                raise
            time.sleep(MKDIR_RETRY_DELAY_S)


def compute_delta_v(vx0, vy0):
    return math.sqrt(vx0**2 + vy0**2)


def build_trapezoid(a_peak, delta_v, rise_fraction, max_duration=float("nan")):
    if not (0.0 < rise_fraction < 0.5):
        raise ValueError("RISE_FRACTION must be strictly between 0 and 0.5")
    if a_peak <= 0.0:
        raise ValueError(f"a_peak must be > 0, got {a_peak}")

    t_total = delta_v / (a_peak * (1.0 - rise_fraction))

    # max_duration = NaN means "no bound" (NaN comparisons are always False)
    if t_total > max_duration:
        a_min = delta_v / (max_duration * (1.0 - rise_fraction))
        raise ValueError(
            f"a_peak={a_peak}: t_total={t_total:.6f} ms exceeds the "
            f"{max_duration} ms phenomenon window. The smallest a_peak that "
            f"still fits is {a_min:.6f} mm/ms^2 ({a_min / G_IN_MM_PER_MS2:.2f} g)."
        )

    t1 = rise_fraction * t_total
    t2 = (1.0 - rise_fraction) * t_total
    t3 = t_total

    points = [(0.0, 0.0), (t1, a_peak), (t2, a_peak), (t3, 0.0)]

    # sanity check on area
    area = 0.5 * (t3 + (t2 - t1)) * a_peak
    return points, area, (t1, t2, t3)


def format_field(value, decimals, width=20):
    return f"{value:>{width}.{decimals}f}"


def write_curve_file(filepath, points, decimals, a_peak, t3, delta_v,
                     curve_id, curve_name):
    """Write a *DEFINE_CURVE keyword block ready to be used via *INCLUDE.

    Layout mirrors a HyperMesh export: the $HM... lines are '$' comments to
    LS-DYNA (HyperMesh reads them for the curve name/color on import), card 1
    uses 10-character fields, and the point pairs use 20-character fields.
    """
    lines = [
        f"$ a_peak = {a_peak:.6f} mm/ms^2 = {a_peak / G_IN_MM_PER_MS2:.4f} g",
        f"$ t3 = {t3:.6f} ms,  delta_v = {delta_v:.6f} mm/ms",
        "$ X = time [ms],  Y = acceleration [mm/ms^2]",
        "*DEFINE_CURVE",
        f"$HMNAME CURVES{curve_id:>10d}{curve_name}",
        f"$HWCOLOR CURVES{curve_id:>10d}{HW_COLOR:>8d}",
        f"$HMCURVE{1:>6d}{0:>5d} {curve_name}",
        "$     LCID      SIDR       SFA       SFO      OFFA      OFFO    DATTYP     LCINT",
        f"{curve_id:>10d}{'':>10}{SFA:>10.1f}{SFO:>10.1f}",
        "$" + f"{'X':>19}" + f"{'Y':>20}",
    ]
    for x, y in points:
        lines.append(format_field(x, decimals) + format_field(y, decimals))
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# Filename token for each input unit, so the tag is unambiguous on disk
UNIT_FILENAME_TOKEN = {"g": "g", "mm/ms^2": "mmms2"}


def sanitize_apeak_for_filename(a_peak_input, units):
    """Tag the file with the a_peak AS THE USER TYPED IT, plus its unit."""
    return f"{a_peak_input:.3f}".replace(".", "p") + UNIT_FILENAME_TOKEN[units]


def plot_family(curves, scale, unit, ylabel, title, outpath):
    """Plot every curve with its acceleration divided by `scale`."""
    plt.figure(figsize=FIGSIZE)

    colors = plt.cm.viridis([i / max(len(curves) - 1, 1) for i in range(len(curves))])

    for curve, color in zip(curves, colors):
        times = [x for x, _ in curve["points"]]
        accels = [y / scale for _, y in curve["points"]]
        label = (f"a_peak = {curve['a_peak'] / scale:.4g} {unit}"
                 f"  (t3 = {curve['t3']:.1f} ms)")
        plt.plot(times, accels, linewidth=2, color=color,
                 marker="o", markersize=3, label=label)

    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=8)
    if PLOT_FULL_WINDOW:
        plt.xlim(0.0, PHENOMENON_DURATION_MS)
    else:
        plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()
    return outpath


def make_plots(curves, output_dir):
    mm_path = plot_family(
        curves,
        scale=1.0,
        unit="mm/ms$^2$",
        ylabel="Acceleration (mm/ms$^2$)",
        title=r"Equal-$\Delta v$ acceleration pulse family -- solver units (mm, ms, kg)",
        outpath=os.path.join(output_dir, PLOT_MM_MS2),
    )
    g_path = plot_family(
        curves,
        scale=G_IN_MM_PER_MS2,
        unit="g",
        ylabel="Acceleration (g)",
        title=r"Equal-$\Delta v$ acceleration pulse family -- in g",
        outpath=os.path.join(output_dir, PLOT_G),
    )
    return mm_path, g_path


def main():
    delta_v = compute_delta_v(VX0, VY0)
    make_output_dir(OUTPUT_DIR)

    a_fills_window = delta_v / (PHENOMENON_DURATION_MS * (1.0 - RISE_FRACTION))

    if VERBOSE:
        print(f"input read as [{INPUT_UNITS}]; curves always written in [mm/ms^2]")
        print(f"delta_v = {delta_v:.6f} mm/ms")
        print(f"phenomenon window = {PHENOMENON_DURATION_MS:g} ms  =>  smallest "
              f"admissible a_peak = {a_fills_window:.6f} mm/ms^2 = "
              f"{a_fills_window / G_IN_MM_PER_MS2:.2f} g\n")
        print(f"{'a_in[' + INPUT_UNITS + ']':>14} {'a[mm/ms^2]':>12} {'a[g]':>9} "
              f"{'t1':>10} {'t2':>10} {'t3':>10} {'%window':>9} {'area_check':>12}")

    curves = []
    for i, a_peak_input in enumerate(PEAK_ACCELERATIONS):
        a_peak = to_mm_per_ms2(a_peak_input, INPUT_UNITS)

        points, area, (t1, t2, t3) = build_trapezoid(
            a_peak, delta_v, RISE_FRACTION, PHENOMENON_DURATION_MS
        )

        tag = sanitize_apeak_for_filename(a_peak_input, INPUT_UNITS)
        filename = f"{FILE_PREFIX}_apeak_{tag}{FILE_EXTENSION}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        curve_id = CURVE_ID_START + (i if UNIQUE_CURVE_IDS else 0)
        curve_name = f"{CURVE_NAME_PREFIX}_{tag}"
        write_curve_file(filepath, points, DECIMALS, a_peak, t3, delta_v,
                         curve_id, curve_name)

        curves.append({"a_peak": a_peak, "points": points, "t3": t3, "path": filepath})

        if VERBOSE:
            pct = 100.0 * t3 / PHENOMENON_DURATION_MS
            print(f"{a_peak_input:>14.4f} {a_peak:>12.6f} "
                  f"{a_peak / G_IN_MM_PER_MS2:>9.2f} {t1:>10.4f} "
                  f"{t2:>10.4f} {t3:>10.4f} {pct:>8.2f}% {area:>12.6f}")

    if VERBOSE:
        print(f"\n{len(curves)} curve files written to '{OUTPUT_DIR}/'")

    if MAKE_PLOTS:
        mm_path, g_path = make_plots(curves, OUTPUT_DIR)
        if VERBOSE:
            print(f"plot (mm/ms^2): {mm_path}")
            print(f"plot (g):       {g_path}")


if __name__ == "__main__":
    main()
