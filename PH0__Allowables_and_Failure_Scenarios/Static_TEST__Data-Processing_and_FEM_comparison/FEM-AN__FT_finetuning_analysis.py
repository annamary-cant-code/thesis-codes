"""
FEM-AN__FT_finetuning_analysis.py

Overlays FEM force-displacement curves from a sweep of candidate FT values
[MPa], marks each run's breaking load [N], colors/labels each curve by its FT
value (parsed from filename) and its breaking load, and draws a manual
"target failure load" threshold line -- so the FT that puts the model's
failure load on target can be read straight off the legend.

One figure PER BREAKING MODE. Each phial breaking mode (A-E, see MODE_LEGEND)
fails at its own load and is fitted with its own FT, so the curves for
different modes must not share a plot: the mode is read from the filename and
each mode gets its own figure, its own FT sweep and its own hard-coded target
load. Only the modes actually present in the data folder produce a figure.

A phial that fails in more than one mode at once is written as the letters run
together -- "AD" is a combined dome+base failure. A hybrid is treated as a
breaking mode in its own right: it fails at its own load, so it gets its own
figure, its own FT sweep and its own target load, and its curves are never
mixed in with those of the single modes it is made of.

Failure load only: stiffness is not fitted or reported here, that is what
FEM-AN__stiffness_comparison_EXP-vs-FEM.py is for.

The breaking load is the load at which the phial FIRST fractures: the highest
load carried before the curve first sheds load (marked "o"). The material is
glass -- linear-elastic right up to fracture -- so there is no yield or
softening knee before failure and none is looked for; the first load the
curve gives back is the first thing to break.

On a single mode the first shed is also the largest, so the breaking load is
simply the curve's peak. On a hybrid mode it is NOT: one of the two sites
cracks first, the load dips, and the phial carries on through the
redistributed load to a higher overall peak before the second site goes. That
later peak is the strength of an already-cracked phial, not of the phial, so
it is the first shed that gets reported. See find_failure_point.

Each curve is trimmed shortly after its collapse -- the deepest drop in the
run, which on a hybrid is the second event rather than the reported one (see
find_collapse_point/trim_after_failure). The collapse is shown whole with a
short settled tail beneath it, and the long post-fracture plateau -- which
says nothing about the failure load -- is discarded. The axes of each figure
follow that mode's own trimmed curves.

USAGE:
    python FEM-AN__FT_finetuning_analysis.py

    python FEM-AN__FT_finetuning_analysis.py <folder> --modes A,D \
        --target-load A=98.7 D=150 --no-show

    python FEM-AN__FT_finetuning_analysis.py <folder> \
        --post-drop-frac 0.01 --arm-frac 0.10 --settle-frac 0.05 --tail-frac 1.0

INPUT FILES
-----------
Same paired-export mechanism as FEM-AN__stiffness_comparison_EXP-vs-FEM.py:
FEM doesn't export one ready-made force-vs-displacement file -- each
candidate FT run is two separate time-history exports, a "<tag>_disp" and a
"<tag>_force" file, both sampled at the same time steps (same first column).
Drop both files for a run into the data folder and they're picked up
automatically -- no config edit needed, including for a mode that has never
been plotted before. A "<tag>_disp" file without a matching "<tag>_force" (or
vice versa) is skipped with a printed warning.

Force exports are in kN; scaled by FORCE_SCALE (1000) on load so all
downstream analysis/plotting is in N.

FILENAME -> MODE + FT ASSUMPTION:
    tag = "<MODE>_<FT>", e.g. "A_090-5_disp" / "A_090-5_force" -> tag
    "A_090-5" -> breaking mode A, FT = 90.5 MPa.

    MODE is one or more letters, each of which must appear in MODE_LEGEND
    (A-E). More than one letter is a hybrid mode -- "AD_060" is a dome+base
    failure at FT = 60 MPa -- and is kept distinct from every other mode,
    letter order included ("AD" and "DA" are separate figures).

    FT is written in GPa with the leading "0." dropped and the decimal point
    written as "-", ALWAYS zero-padded to 3 integer digits:
        "045"   -> 0.045 GPa -> FT = 45 MPa
        "090-5" -> 0.0905 GPa -> FT = 90.5 MPa
    The padding is enforced, not assumed: a token that isn't 3 digits (plus an
    optional "-<decimals>") is rejected rather than silently mis-scaled, since
    "90-5" would otherwise parse as 9.05 MPa. Edit parse_tag() if your naming
    differs.
"""

import re
import glob
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
DATA_FOLDER = Path(__file__).parent / "fem_ft-finetuning_curves"

# Single-letter breaking-mode codes. Kept in step with MODE_LEGEND in
# 05_breaking_mode_analysis.py, which is the source of truth for them -- it
# can't be imported from here because its filename starts with a digit.
MODE_LEGEND = {
    "A": "Dome",
    "B": "Bulging",
    "C": "Indentation",
    "D": "Base",
    "E": "Cylinder",
}

# Target failure load [N] per breaking mode -- the red threshold line each
# mode's FT sweep is being fitted against. Filled in by hand from the
# experimental campaign. A mode left at None, or missing from this dict
# entirely, still gets its figure and its curves, just without the threshold
# line, so a new mode's runs can be dropped in and looked at before its target
# is known.
#
# Hybrid modes are keyed by their own letter string ("AD"), not derived from
# the single modes: a combined failure has its own experimental load.
TARGET_FAILURE_LOAD = {
    "A": 98.7,   # Dome
    "B": None,   # Bulging
    "C": None,   # Indentation
    "D": None,   # Base
    "E": None,   # Cylinder
    "AD": 98.7,  # Dome + Base
}

# Every FEM force export is in kN, so this is applied to every curve.
X_SCALE = 1.0
FORCE_SCALE = 1000.0   # kN -> N

# Load shed that counts as the run's catastrophic collapse, as a fraction of
# its overall peak. Only ever used to decide where to cut the plot off (see
# find_collapse_point) -- no reported load depends on it, which is why it is a
# constant here rather than another command-line knob. The collapses in these
# runs shed 40-60%, the events before them under 10%, so anything in between
# separates them.
COLLAPSE_FRAC = 0.20
# ============================================================================

# "<MODE>_<3 digits>[-<decimals>]", e.g. "A_045", "A_090-5" or "AD_060".
# MODE is one or more letters: several letters is a hybrid mode (see
# check_mode), so the letters are matched as a group and validated one by one.
TAG_PATTERN = re.compile(r'^([A-Za-z]+)_(\d{3})(?:-(\d+))?$')


def check_mode(mode: str, source: str):
    """Validate a mode code -- one letter, or several for a hybrid -- against
    MODE_LEGEND, letter by letter. Raises ValueError naming the offending
    letter; `source` is quoted in the message to say where it came from."""
    unknown = [letter for letter in mode if letter not in MODE_LEGEND]
    if unknown:
        raise ValueError(
            f"{source} names breaking mode '{mode}', whose letter(s) "
            f"{'/'.join(unknown)} aren't among {'/'.join(MODE_LEGEND)}. A mode is one "
            f"letter, or several run together for a hybrid failure (e.g. 'AD' for "
            f"dome+base). Add the letter to MODE_LEGEND if it's a real mode."
        )


def mode_label(mode: str):
    """Human-readable name for a mode code: 'A' -> 'Dome', 'AD' -> 'Dome + Base'."""
    return " + ".join(MODE_LEGEND[letter] for letter in mode)


def parse_tag(tag: str):
    """Split a '<tag>' (e.g. from '<tag>_disp' / '<tag>_force') into its
    breaking mode and its FT [MPa]. See the FILENAME assumption in the module
    docstring. Returns (mode, ft)."""
    match = TAG_PATTERN.match(tag)
    if not match:
        raise ValueError(
            f"Tag '{tag}' doesn't match the '<MODE>_<FT>' naming, e.g. 'A_045', "
            f"'A_090-5' or 'AD_060' (mode letter(s), underscore, FT in GPa without the "
            f"leading '0.', zero-padded to 3 digits, decimals after a '-')."
        )
    mode, int_digits, frac_digits = match.group(1).upper(), match.group(2), match.group(3)
    check_mode(mode, f"Tag '{tag}'")
    digits = int_digits + (frac_digits or "")
    ft = (int(digits) / 10 ** len(digits)) * 1000   # GPa -> MPa
    return mode, ft


def discover_fem_curve_pairs(folder):
    """Finds every '<tag>_disp' file in folder with a matching '<tag>_force'
    next to it (e.g. 'A_045_disp' + 'A_045_force' -> tag 'A_045'). New FT
    sweeps, and new breaking modes, just need their two exported files dropped
    in here -- nothing to configure. Returns a sorted list of
    (tag, disp_path, force_path)."""
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


def trim_after_failure(disp, force, top_idx, bottom_idx, tail_frac=1.0):
    """
    Cut the curve off shortly after the collapse drop (from
    find_collapse_point).

    The runs carry on for a long way past failure -- a crushed-material
    plateau several times the displacement of interest -- and none of it says
    anything about the failure load. Keeping it squeezes the loading branch
    and the drop itself into a sliver of the axes.

    The cut is placed from the curve's own geometry rather than a hard-coded
    displacement, so it adapts to each run: keep tail_frac of the collapse's
    own displacement width beyond its bottom. The drop is therefore always
    shown whole, with a proportionate stretch of settled post-failure load
    beneath it for context, and nothing after that.

    Args:
        tail_frac : how much of the drop's displacement width to keep past
                    the bottom (1.0 = tail as wide as the drop).

    Returns:
        end : slice end index -- plot disp[:end], force[:end].
    """
    n = len(force)
    width = disp[bottom_idx] - disp[top_idx]
    beyond = np.flatnonzero(disp > disp[bottom_idx] + tail_frac * width)
    end = int(beyond[0]) + 1 if beyond.size else n
    return min(max(end, bottom_idx + 1), n)


def find_failure_point(force, post_drop_frac=0.01, arm_frac=0.10):
    """
    Locate the failure (breaking) point: the load at which the phial FIRST
    fractures -- the highest load carried before the curve first sheds load.

    THE FIRST EVENT, NOT THE BIGGEST. A single-mode run is linear right up to
    fracture and then collapses, so its first shed IS its collapse and the two
    readings coincide. A hybrid mode (e.g. AD) is the case that separates them:
    one of its two sites cracks first, the load dips, and the structure carries
    on through the redistributed load to a HIGHER overall peak before the
    second site goes. That later peak is not the strength of the phial -- it is
    the strength of an already-cracked phial -- so the first shed is what gets
    reported and compared against the target load.

    A shed only counts once the run has ARMED, i.e. once the load has reached
    arm_frac of the run's overall peak. This is what keeps the pre-contact
    region out: those first few increments wiggle by ~0.2 N at a load of ~0.2 N,
    which on drop size alone is not cleanly separable from a genuine 1.3 N first
    fracture, but on load level is ~70x away from it. Screening by load level
    rather than by drop size is what lets post_drop_frac come down to a value
    that can see a real first event at all.

    Args:
        post_drop_frac : load shed marking a fracture, as a fraction of the
                         run's overall peak. Must clear the solver's own ripple
                         on the loading branch (~0.5%) and stay under the
                         smallest genuine first event (~1.3% on these runs).
        arm_frac       : sheds are ignored until the load has reached this
                         fraction of the run's overall peak.

    Returns:
        fail_idx : index of the breaking load
        failed   : False if no qualifying shed was found at all, meaning the run
                   was probably not carried through to failure and fail_idx is
                   just the largest load reached so far.
    """
    force = np.asarray(force, dtype=float)
    peak = float(np.max(force))
    if peak <= 0:
        return int(np.argmax(force)), False
    running_max = np.maximum.accumulate(force)
    shed = running_max - force
    qualifies = (shed > post_drop_frac * peak) & (running_max >= arm_frac * peak)
    hits = np.flatnonzero(qualifies)
    if hits.size == 0:
        return int(np.argmax(force)), False
    return int(np.argmax(force[:hits[0]])), True


def find_collapse_point(force, collapse_frac=COLLAPSE_FRAC, settle_frac=0.05):
    """
    Locate the run's catastrophic drop as (top_idx, bottom_idx). Used only to
    decide where to trim the plot, never to report a load.

    This is deliberately NOT find_failure_point: on a hybrid the reported
    breaking load is the first small fracture, and trimming the plot there
    would cut the run off before its collapse is ever drawn. Trimming has to
    follow the big drop, whichever event that turns out to be.

    Note this cannot be done by taking the point that sits furthest below the
    run's running maximum. That point is not the bottom of the collapse: the
    running maximum stays pinned at the pre-collapse peak for the rest of the
    run, so the deepest such point is wherever the long crushed plateau happens
    to sag lowest, hundreds of increments past the collapse.

    So the collapse is found the same way failure is -- the first shed past a
    threshold, but a large one (collapse_frac) rather than a fracture-sized one.
    The walk to the bottom then starts from INSIDE the drop rather than from its
    top: started at the top it would have to cross the small first-fracture dips
    that precede the collapse on some runs, and would stop at the first of those
    instead of carrying on to the real bottom.

    Args:
        collapse_frac : shed marking the collapse, as a fraction of the run's
                        overall peak load.
        settle_frac   : the drop is over once the load has climbed back this
                        fraction of its depth above its lowest point -- enough
                        to step over the ripple in the plateau.
    """
    force = np.asarray(force, dtype=float)
    peak = float(np.max(force))
    shed = np.maximum.accumulate(force) - force
    hits = np.flatnonzero(shed > collapse_frac * peak)
    if hits.size == 0:
        return int(np.argmax(force)), len(force) - 1   # no collapse -> nothing to trim to
    inside = int(hits[0])                              # partway down the drop
    top_idx = int(np.argmax(force[:inside]))

    post = force[inside:]
    running_min = np.minimum.accumulate(post)
    depth = force[inside] - running_min
    rebound = np.flatnonzero(post > running_min + settle_frac * depth)
    stop = int(rebound[0]) + 1 if rebound.size else len(post)
    bottom_idx = inside + int(np.argmin(post[:stop]))
    return top_idx, bottom_idx


def build_ft_colors(all_fts):
    """One distinct, non-repeating color per FT value, assigned across ALL
    modes at once so a given FT keeps the same color in every mode's figure --
    the figures then sit side by side and read together.

    A qualitative colormap, not a continuous one: on a continuous map two close
    FT values come out almost identical, which is the opposite of what's wanted
    when the whole point is telling neighbouring sweep values apart."""
    unique_fts = sorted(set(all_fts))
    cmap = plt.cm.tab10 if len(unique_fts) <= 10 else plt.cm.tab20
    return {ft: cmap(i % cmap.N) for i, ft in enumerate(unique_fts)}


def plot_mode(mode, curves, target_load, ft_colors, args, out_path):
    """Build and save one mode's figure. curves is a list of
    (tag, ft, disp, force). Returns the list of (ft, fail_load) for it."""
    fig, ax = plt.subplots(figsize=(9, 6))

    summary = []
    x_ends = []
    y_peaks = []
    for tag, ft, disp, force in sorted(curves, key=lambda c: c[1]):
        # breaking force = load carried before the curve FIRST sheds load
        fail_idx, failed = find_failure_point(force, post_drop_frac=args.post_drop_frac,
                                              arm_frac=args.arm_frac)
        fail_load = force[fail_idx]
        if not failed:
            print(f"  WARNING '{tag}': load never shed {args.post_drop_frac:.1%} of its peak -- "
                  f"this run may not have been carried to failure; reporting the highest load reached.")

        # keep the loading branch and the collapse, discard the plateau after
        # it -- trimming follows the big drop, not the reported first fracture.
        # A run that never failed has no collapse to trim to: its deepest
        # "drop" is just solver ripple somewhere up the loading branch, and
        # cutting there would throw away the part of the curve being looked at.
        end = len(force)
        if failed:
            top_idx, bottom_idx = find_collapse_point(force, settle_frac=args.settle_frac)
            end = trim_after_failure(disp, force, top_idx, bottom_idx, tail_frac=args.tail_frac)
            end = max(end, fail_idx + 1)   # never trim away the marked point itself
        disp, force = disp[:end], force[:end]
        x_ends.append(disp[-1])
        y_peaks.append(float(np.max(force)))

        summary.append((ft, fail_load))

        color = ft_colors[ft]
        ax.plot(disp, force, color=color, lw=1.5,
                label=f"FT = {ft:g} MPa  ->  {fail_load:.1f} N")
        ax.scatter(disp[fail_idx], fail_load, color=color, marker="o", s=20,
                   zorder=6, edgecolors="black", linewidths=0.6)

    if target_load is not None:
        ax.axhline(target_load, color="red", lw=2,
                   label=f"Target failure load = {target_load:g} N")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="0.25", s=20,
               label="Breaking load")

    # axes follow this mode's own trimmed curves, so its loading branch and its
    # drop fill the plot whatever load the mode fails at. The top follows the
    # curves' own peaks, not their breaking loads: on a hybrid the curve keeps
    # climbing past the break, and fitting the axes to the breaking loads would
    # cut the top off every curve in the figure.
    if x_ends:
        x_max = max(x_ends)
        y_max = max(y_peaks + ([target_load] if target_load is not None else []))
        ax.set_xlim(left=-0.02 * x_max, right=args.xmax if args.xmax is not None else x_max)
        ax.set_ylim(bottom=-0.08 * y_max, top=args.ymax if args.ymax is not None else 1.2 * y_max)

    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=1,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(f"Force-displacement for FT finetuning - Mode {mode} ({mode_label(mode)})")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    print(f"  Saved plot to {out_path}")
    return summary


def print_mode_summary(mode, summary, target_load):
    """Per-mode table of FT vs breaking load, and which FT lands closest to
    that mode's target."""
    print()
    print(f"Mode {mode} ({mode_label(mode)}):")
    if target_load is None:
        print(f"{'FT [MPa]':>10}  {'F_fail [N]':>11}")
        for ft, fail_load in sorted(summary):
            print(f"{ft:>10.6g}  {fail_load:>11.1f}")
        print(f"  No target failure load set for mode {mode} -- add one to "
              f"TARGET_FAILURE_LOAD to fit against it.")
        return
    print(f"{'FT [MPa]':>10}  {'F_fail [N]':>11}  {'vs target':>10}")
    for ft, fail_load in sorted(summary):
        print(f"{ft:>10.6g}  {fail_load:>11.1f}  {fail_load - target_load:>+10.1f}")
    best = min(summary, key=lambda r: abs(r[1] - target_load))
    print(f"  Closest to the {target_load:g} N target: FT = {best[0]:g} MPa "
          f"(F_fail = {best[1]:.1f} N)")


def parse_target_overrides(items):
    """Parse --target-load 'A=98.7' 'AD=150' into {'A': 98.7, 'AD': 150.0}."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--target-load expects '<MODE>=<load>' (e.g. 'A=98.7'), got '{item}'")
        mode, _, value = item.partition("=")
        mode = mode.strip().upper()
        check_mode(mode, "--target-load")
        overrides[mode] = float(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Overlay FEM force-disp curves per breaking mode, mark each run's breaking "
                    "load, and compare against that mode's target failure load. One figure per "
                    "mode; curves labeled by FT [MPa]."
    )
    parser.add_argument("folder", type=str, nargs="?", default=None,
                         help="Folder containing paired '<MODE>_<FT>_disp' / '<MODE>_<FT>_force' "
                              "curves. Overrides DATA_FOLDER constant if given.")
    parser.add_argument("--modes", type=str, default=None,
                         help="Only plot these breaking modes, comma-separated (e.g. 'A,AD'). "
                              "A hybrid must be named in full -- 'A' does not select 'AD'. "
                              "Default: every mode found in the folder.")
    parser.add_argument("--target-load", type=str, nargs="+", default=None, metavar="MODE=LOAD",
                         help="Target failure load [N] per mode, e.g. --target-load A=98.7 D=150. "
                              "Overrides the TARGET_FAILURE_LOAD entry for those modes only.")
    parser.add_argument("--post-drop-frac", type=float, default=0.01,
                         help="Load shed marking a fracture, as a fraction of the run's overall peak load "
                              "(0.01 = load fell by 1%% of the peak). The breaking force is the highest "
                              "load reached before the FIRST such shed. Raise it if solver ripple is being "
                              "read as a fracture, lower it if a real first fracture is being walked past.")
    parser.add_argument("--arm-frac", type=float, default=0.10,
                         help="Ignore load sheds until the load has reached this fraction of the run's "
                              "overall peak. Keeps the sub-newton pre-contact wiggle from being read as "
                              "a fracture.")
    parser.add_argument("--settle-frac", type=float, default=0.05,
                         help="Collapse is considered settled once the load climbs this fraction of the "
                              "drop's depth back above its lowest point (marks the bottom of the drop, "
                              "which is where trimming measures from). Affects the plot only.")
    parser.add_argument("--tail-frac", type=float, default=1.0,
                         help="How much of the collapse's displacement width to keep past the bottom of "
                              "the drop (1.0 = tail as wide as the drop). Everything after is discarded.")
    parser.add_argument("--xmax", type=float, default=None,
                         help="Right x-limit [mm], applied to every mode's figure. Default: the end of "
                              "each mode's own trimmed curves.")
    parser.add_argument("--ymax", type=float, default=None,
                         help="Top y-limit [N], applied to every mode's figure. Default: 1.2x the largest "
                              "of each mode's own curve peaks and its target.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Directory for the per-mode PNGs (default: the data folder). Each is saved "
                              "as 'ft_finetuning_overlay_<MODE>.png'.")
    parser.add_argument("--no-show", action="store_true",
                         help="Save the figures without opening them (one window per mode otherwise).")
    args = parser.parse_args()

    folder = Path(args.folder if args.folder is not None else DATA_FOLDER)
    out_dir = Path(args.out_dir) if args.out_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = dict(TARGET_FAILURE_LOAD)
    targets.update(parse_target_overrides(args.target_load))

    pairs = discover_fem_curve_pairs(str(folder))
    if not pairs:
        raise FileNotFoundError(
            f"No paired '<MODE>_<FT>_disp' / '<MODE>_<FT>_force' curves found in {folder}")

    # group the sweep by breaking mode -- each mode is fitted separately
    by_mode = defaultdict(list)
    for tag, disp_path, force_path in pairs:
        mode, ft = parse_tag(tag)
        disp, force = build_combined_fem_curve(tag, disp_path, force_path)
        by_mode[mode].append((tag, ft, disp, force))

    wanted = None
    if args.modes:
        wanted = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
        missing = [m for m in wanted if m not in by_mode]
        if missing:
            print(f"  WARNING: no curves found for requested mode(s) {', '.join(missing)}.")
        by_mode = {m: c for m, c in by_mode.items() if m in wanted}
        if not by_mode:
            raise FileNotFoundError(f"None of the requested modes have curves in {folder}")

    found = ", ".join(f"{m} ({len(by_mode[m])} curve(s))" for m in sorted(by_mode))
    print(f"Found {len(pairs) if wanted is None else sum(len(c) for c in by_mode.values())} "
          f"FEM curve(s) in '{folder}/' across {len(by_mode)} mode(s): {found}.")

    # colors assigned across every mode at once, so one FT = one color everywhere
    ft_colors = build_ft_colors(ft for curves in by_mode.values() for _, ft, _, _ in curves)

    summaries = {}
    for mode in sorted(by_mode):
        out_path = out_dir / f"ft_finetuning_overlay_{mode}.png"
        summaries[mode] = plot_mode(mode, by_mode[mode], targets.get(mode),
                                    ft_colors, args, out_path)

    for mode in sorted(summaries):
        if summaries[mode]:
            print_mode_summary(mode, summaries[mode], targets.get(mode))
    print()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
