"""
FEM-AN__modeA-vs-modeAD_60MPa_comparison.py

Compares the two FT = 60 MPa runs against each other -- mode A (dome only)
and mode AD (dome + base) -- and produces TWO figures:

  1. STATIC  : the two force-displacement curves overlaid, each with its
               breaking point marked exactly as in the FT-finetuning figures,
               plus the single red target-failure-load line both are fitted
               against (the same 98.7 N for both modes).

  2. INTERACTIVE : mode AD alone, with a time cursor -- a slider that drives a
               vertical line across the curve, so the run can be walked through
               increment by increment and each fracture watched as it happens.
               The two fracture events of the hybrid are marked on the curve AND
               on the slider track:
                   BASE break -- the first load shed, the event already marked
                                 in the FT-finetuning figure
                   DOME break -- the highest load the already-cracked phial
                                 carries, just before the catastrophic collapse
               Neither is hard-coded: both are read off the force-displacement
               data (see find_base_event / find_dome_event).

WHY THIS PAIR IS WORTH ITS OWN SCRIPT. FEM-AN__FT_finetuning_analysis.py puts
every FT of a mode on one figure and never crosses modes, because a sweep is
fitted per mode. Here the question is the opposite one: at one FT, what does
adding the base failure to the dome failure do to the response? So the FT is
fixed and the mode varies, and the curves are coloured by MODE rather than by
FT (at a single FT the FT colouring would give both curves the same colour).

WHICH EVENT IS THE BASE AND WHICH IS THE DOME. Mode A -- the dome on its own --
breaks at ~97 N at this FT. Mode AD sheds load twice: once at ~69 N, then again
at ~99 N. The first shed is therefore too early to be the dome, so it is read as
the base cracking; the phial then carries on through the redistributed load to
the dome's own ~99 N before that goes too and the whole thing collapses. The
same reasoning is set out in find_failure_point in the FT-finetuning script,
which is why the FIRST of the two is the load reported there.

The detection logic is not re-implemented here: parsing, the first-shed search,
the collapse search and the post-collapse trim are all imported from
FEM-AN__FT_finetuning_analysis.py, so the breaking points drawn here are by
construction the same ones drawn there.

USAGE:
    python FEM-AN__modeA-vs-modeAD_60MPa_comparison.py

    python FEM-AN__modeA-vs-modeAD_60MPa_comparison.py --static-only
    python FEM-AN__modeA-vs-modeAD_60MPa_comparison.py --time-unit ms
    python FEM-AN__modeA-vs-modeAD_60MPa_comparison.py --full --no-target

INTERACTIVE FIGURE CONTROLS:
    drag the slider                        move the time cursor
    left / right arrow keys                step one solver increment
    "Base break" / "Dome break" buttons    jump the cursor onto an event
    "Reset"                                back to the start of the run

INPUT FILES
-----------
The same paired time-history exports as the FT-finetuning script, read from the
same folder: '<tag>_disp' and '<tag>_force' for tags 'A_060' and 'AD_060'
(mode letters, FT in GPa with the leading '0.' dropped). Force exports are kN
and are scaled to N on load.

Unlike the FT-finetuning script this one also keeps the TIME column: the
interactive cursor is driven by solver time, not by displacement, so that the
pre-contact increments -- which take time but almost no displacement -- are
walked through at their true cost.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ============================== CONFIGURATION ==============================
HERE = Path(__file__).parent
DATA_FOLDER = HERE / "fem_ft-finetuning_curves"

# The detection/parsing logic lives in the FT-finetuning script and is imported
# from it rather than copied, so the two sets of figures can never drift apart.
# It has to be loaded by path: its filename isn't a valid module name.
FT_ANALYSIS_SCRIPT = HERE / "FEM-AN__FT_finetuning_analysis.py"

# The two runs being compared. Same FT (60 MPa), different breaking mode.
MODE_A_TAG = "A_060"     # dome only
MODE_AD_TAG = "AD_060"   # dome + base

# One colour per MODE (not per FT -- both runs share an FT here).
MODE_COLORS = {
    "A": "tab:blue",
    "AD": "tab:orange",
}

# Colours of the three phases of the mode AD run, used to shade the interactive
# curve either side of its two fractures.
PHASE_COLORS = {
    "intact": "tab:orange",   # nothing broken yet
    "base": "tab:purple",     # base cracked, dome still carrying
    "post": "0.55",           # both gone -- crushed plateau
}
EVENT_COLORS = {
    "base": "tab:purple",
    "dome": "tab:red",
}

# Sanity check on the detected dome event, from the experimental reading of
# these runs: it is expected somewhere around 0.06 mm / 100 N. Only ever used
# to print a warning if the detection lands somewhere else entirely -- no
# marker position is taken from it.
DOME_EXPECTED_DISP = 0.06   # [mm]
DOME_EXPECTED_LOAD = 100.0  # [N]
DOME_DISP_TOL = 0.02        # [mm]
DOME_LOAD_TOL = 20.0        # [N]
# ============================================================================


def load_ft_analysis():
    """Import FEM-AN__FT_finetuning_analysis.py by path (its filename is not a
    legal module name, so a plain `import` can't reach it). Only module-level
    code runs -- its main() is behind an __main__ guard."""
    if not FT_ANALYSIS_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Can't find '{FT_ANALYSIS_SCRIPT.name}' next to this script -- it holds the "
            f"curve parsing and the breaking-point detection this one reuses.")
    spec = importlib.util.spec_from_file_location("ft_finetuning_analysis", FT_ANALYSIS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run(ft, folder, tag):
    """Read one run's '<tag>_disp' / '<tag>_force' pair into (time, disp, force).

    Same pairing and scaling as ft.build_combined_fem_curve -- displacement made
    positive, force kN -> N -- but the time column is kept as well, since the
    interactive cursor is driven by solver time."""
    disp_path = folder / f"{tag}_disp"
    force_path = folder / f"{tag}_force"
    for path in (disp_path, force_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing '{path.name}' in {folder} -- both '{tag}_disp' and '{tag}_force' "
                f"are needed to build run '{tag}'.")

    disp_time, disp_val = ft.parse_history_file(disp_path)
    force_time, force_val = ft.parse_history_file(force_path)

    n = min(len(disp_val), len(force_val))
    if len(disp_val) != len(force_val):
        print(f"  WARNING '{tag}': disp has {len(disp_val)} points, force has {len(force_val)} "
              f"-- using the first {n} common to both, verify the exports actually line up.")
    disp_time, disp_val, force_val = disp_time[:n], disp_val[:n], force_val[:n]
    if not np.allclose(disp_time, force_time[:n], atol=1e-6):
        print(f"  WARNING '{tag}': time columns in the disp and force files don't match -- "
              f"they may not actually be paired correctly.")

    return disp_time, np.abs(disp_val) * ft.X_SCALE, force_val * ft.FORCE_SCALE


def find_base_event(ft, force, args):
    """Index of the FIRST fracture -- on mode AD the base cracking. This is
    exactly ft.find_failure_point, i.e. the point already marked in the
    FT-finetuning figures, so the two figures agree by construction."""
    idx, failed = ft.find_failure_point(force, post_drop_frac=args.post_drop_frac,
                                        arm_frac=args.arm_frac)
    return idx, failed


def find_dome_event(ft, force, args, base_idx):
    """Index of the SECOND fracture -- on mode AD the dome going, after the
    base has already cracked.

    It is the top of the catastrophic collapse: the highest load the phial
    carries before the drop it never recovers from. That is what ft's
    find_collapse_point already locates (it uses it to decide where to trim the
    plot), so the same call gives the dome event here -- no separate rule and no
    hard-coded displacement.

    Returns (idx, found). found is False when the run has no collapse at all,
    in which case there is no second event to mark."""
    top_idx, bottom_idx = ft.find_collapse_point(force, settle_frac=args.settle_frac)
    if bottom_idx >= len(force) - 1 and top_idx == int(np.argmax(force)):
        # find_collapse_point's "nothing collapsed" return
        return top_idx, False
    if top_idx <= base_idx:
        print(f"  WARNING: the collapse tops out at or before the first fracture "
              f"(index {top_idx} vs {base_idx}) -- this run may not have two separable "
              f"events, so the 'dome' marker is not trustworthy.")
    return top_idx, True


def check_dome_expectation(disp, force, idx):
    """Warn if the detected dome event is nowhere near where it is expected
    (~0.06 mm / ~100 N). Purely a check -- nothing is moved."""
    d, f = disp[idx], force[idx]
    off_disp = abs(d - DOME_EXPECTED_DISP) > DOME_DISP_TOL
    off_load = abs(f - DOME_EXPECTED_LOAD) > DOME_LOAD_TOL
    if off_disp or off_load:
        print(f"  WARNING: dome event detected at {d:.4f} mm / {f:.1f} N, expected around "
              f"{DOME_EXPECTED_DISP:g} mm / {DOME_EXPECTED_LOAD:g} N. Check the run, or pass "
              f"--dome-disp to pin it somewhere else.")


def override_event(disp, target_disp):
    """Index of the sample nearest a displacement given on the command line."""
    return int(np.argmin(np.abs(disp - target_disp)))


def trimmed_end(ft, disp, force, args, keep_idx, failed):
    """Where to cut the run off: shortly after its collapse, same rule as the
    FT-finetuning figures. A run that never failed has no collapse to trim to,
    so it is kept whole. keep_idx is never trimmed away."""
    if args.full or not failed:
        return len(force)
    top_idx, bottom_idx = ft.find_collapse_point(force, settle_frac=args.settle_frac)
    end = ft.trim_after_failure(disp, force, top_idx, bottom_idx, tail_frac=args.tail_frac)
    return max(end, keep_idx + 1)


# ------------------------------- STATIC FIGURE ------------------------------

def plot_static(ft, runs, target_load, args, out_path):
    """The two curves overlaid, each with its own breaking point marked and the
    shared target failure load drawn across both."""
    fig, ax = plt.subplots(figsize=(9, 6))

    x_ends, y_peaks = [], []
    for mode, tag, _time, disp, force, marks in runs:
        end = marks["end"]
        d, f = disp[:end], force[:end]
        x_ends.append(d[-1])
        y_peaks.append(float(np.max(f)))

        color = MODE_COLORS[mode]
        base_idx = marks["base_idx"]
        ax.plot(d, f, color=color, lw=1.6,
                label=f"Mode {mode} ({ft.mode_label(mode)})  ->  {force[base_idx]:.1f} N")
        ax.scatter(disp[base_idx], force[base_idx], color=color, marker="o", s=20,
                   zorder=6, edgecolors="black", linewidths=0.6)

    if target_load is not None:
        ax.axhline(target_load, color="red", lw=2,
                   label=f"Target failure load = {target_load:g} N")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="0.25", s=20,
               label="Breaking load")

    x_max = max(x_ends)
    y_max = max(y_peaks + ([target_load] if target_load is not None else []))
    ax.set_xlim(left=-0.02 * x_max, right=args.xmax if args.xmax is not None else x_max)
    ax.set_ylim(bottom=-0.08 * y_max, top=args.ymax if args.ymax is not None else 1.2 * y_max)

    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Force-displacement at FT = 60 MPa - Mode A (Dome) vs Mode AD (Dome + Base)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  Saved static plot to {out_path}")
    return fig


# ----------------------------- INTERACTIVE FIGURE ---------------------------

def plot_interactive(ft, run, target_load, args):
    """Mode AD alone with a time cursor.

    The slider is stepped on the run's OWN time samples, so the cursor always
    sits on a real solver increment and the readout is always measured rather
    than interpolated. The two fractures are drawn on the curve and ticked onto
    the slider track, and a button jumps the cursor onto each -- so they stay
    findable however far the cursor is dragged away from them.
    """
    mode, tag, time, disp, force, marks = run
    end = marks["end"]
    base_idx, dome_idx, has_dome = marks["base_idx"], marks["dome_idx"], marks["has_dome"]
    t, d, f = time[:end], disp[:end], force[:end]

    # arrow keys drive the cursor here, so take them off matplotlib's own
    # back/forward navigation for the life of this figure
    for key in ("keymap.back", "keymap.forward"):
        plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in ("left", "right")]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=0.28, top=0.92)

    # the curve in three phases, so which side of each fracture a point is on
    # can be read off the colour alone. Segments overlap by one sample so there
    # is no gap where they meet.
    phases = [
        (0, base_idx + 1, PHASE_COLORS["intact"], "Intact"),
        (base_idx, (dome_idx + 1) if has_dome else len(t), PHASE_COLORS["base"],
         "Base cracked - dome still carrying"),
    ]
    if has_dome:
        phases.append((dome_idx, len(t), PHASE_COLORS["post"], "Base + dome gone - collapse"))
    for i0, i1, color, label in phases:
        ax.plot(d[i0:i1], f[i0:i1], color=color, lw=1.8, label=label, zorder=3)

    if target_load is not None and not args.no_target:
        ax.axhline(target_load, color="red", lw=2, zorder=2,
                   label=f"Target failure load = {target_load:g} N")

    # --- the two fractures, marked on the curve ---
    ax.scatter(d[base_idx], f[base_idx], color=EVENT_COLORS["base"], marker="o", s=70,
               zorder=7, edgecolors="black", linewidths=0.8,
               label=f"BASE break - {f[base_idx]:.1f} N @ {d[base_idx]:.4f} mm")
    ax.annotate(f"BASE break\n{f[base_idx]:.1f} N @ {d[base_idx]:.4f} mm",
                xy=(d[base_idx], f[base_idx]), xytext=(18, -46), textcoords="offset points",
                fontsize=8.5, color=EVENT_COLORS["base"], ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=EVENT_COLORS["base"], alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=EVENT_COLORS["base"], lw=1.1))
    if has_dome:
        ax.scatter(d[dome_idx], f[dome_idx], color=EVENT_COLORS["dome"], marker="*", s=200,
                   zorder=7, edgecolors="black", linewidths=0.8,
                   label=f"DOME break - {f[dome_idx]:.1f} N @ {d[dome_idx]:.4f} mm")
        ax.annotate(f"DOME break\n{f[dome_idx]:.1f} N @ {d[dome_idx]:.4f} mm",
                    xy=(d[dome_idx], f[dome_idx]), xytext=(20, 22), textcoords="offset points",
                    fontsize=8.5, color=EVENT_COLORS["dome"], ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=EVENT_COLORS["dome"], alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=EVENT_COLORS["dome"], lw=1.1))

    x_max = d[-1]
    y_max = max(float(np.max(f)), target_load if target_load is not None else 0.0)
    ax.set_xlim(left=-0.02 * x_max, right=args.xmax if args.xmax is not None else x_max)
    ax.set_ylim(bottom=-0.08 * y_max, top=args.ymax if args.ymax is not None else 1.2 * y_max)
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.set_title(f"Mode {mode} ({ft.mode_label(mode)}), FT = 60 MPa")
    ax.legend(fontsize=8, loc="upper left")

    # --- the moving cursor ---
    cursor_line = ax.axvline(d[0], color="black", lw=1.2, ls="--", zorder=8)
    cursor_dot, = ax.plot([d[0]], [f[0]], marker="o", ms=9, mfc="white", mec="black",
                          mew=1.6, zorder=9)
    time_label = "Time" if args.time_unit is None else f"Time [{args.time_unit}]"
    readout = ax.text(0.985, 0.03, "", transform=ax.transAxes, ha="right", va="bottom",
                      fontsize=9.5, family="monospace", zorder=10,
                      bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.5", alpha=0.95))

    slider_ax = fig.add_axes([0.10, 0.145, 0.80, 0.035])
    slider = Slider(slider_ax, time_label, float(t[0]), float(t[-1]),
                    valinit=float(t[0]), valstep=t, color="0.75")
    slider.valtext.set_fontsize(9)

    # --- the same two fractures, ticked onto the slider track ---
    # so the cursor itself says where the events are, not just the curve
    event_ticks = [(base_idx, "base", "BASE")]
    if has_dome:
        event_ticks.append((dome_idx, "dome", "DOME"))
    for idx, key, text in event_ticks:
        slider_ax.axvline(t[idx], color=EVENT_COLORS[key], lw=2.2, zorder=12, clip_on=False)
        slider_ax.annotate(text, xy=(t[idx], 1.0), xycoords=("data", "axes fraction"),
                           xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                           fontsize=7.5, color=EVENT_COLORS[key], fontweight="bold",
                           annotation_clip=False)

    def state_at(i):
        if has_dome and i >= dome_idx:
            return "BASE + DOME broken", PHASE_COLORS["post"]
        if i >= base_idx:
            return "BASE broken", EVENT_COLORS["base"]
        return "intact", PHASE_COLORS["intact"]

    def draw(i):
        cursor_line.set_xdata([d[i], d[i]])
        cursor_dot.set_data([d[i]], [f[i]])
        state, color = state_at(i)
        unit = "" if args.time_unit is None else f" {args.time_unit}"
        readout.set_text(f"t    = {t[i]:8.4f}{unit}\n"
                         f"disp = {d[i]:8.4f} mm\n"
                         f"F    = {f[i]:8.2f} N\n"
                         f"{state}")
        readout.get_bbox_patch().set_edgecolor(color)
        cursor_dot.set_mec(color)
        fig.canvas.draw_idle()

    def on_slide(val):
        draw(int(np.argmin(np.abs(t - val))))

    slider.on_changed(on_slide)

    def step(delta):
        i = int(np.argmin(np.abs(t - slider.val)))
        slider.set_val(float(t[int(np.clip(i + delta, 0, len(t) - 1))]))

    def on_key(event):
        if event.key == "left":
            step(-1)
        elif event.key == "right":
            step(1)
    fig.canvas.mpl_connect("key_press_event", on_key)

    buttons = []   # kept alive: matplotlib widgets stop responding once GC'd

    def add_button(x, width, label, color, idx):
        b_ax = fig.add_axes([x, 0.045, width, 0.05])
        button = Button(b_ax, label, color=color, hovercolor="0.85")
        button.label.set_fontsize(9)
        button.on_clicked(lambda _event, i=idx: slider.set_val(float(t[i])))
        buttons.append(button)

    add_button(0.10, 0.20, "<- Base break", "#e8dff3", base_idx)
    if has_dome:
        add_button(0.33, 0.20, "Dome break ->", "#f7dede", dome_idx)
    add_button(0.70, 0.20, "Reset", "0.92", 0)

    draw(0)
    print("  Interactive figure: drag the slider (or use the left/right arrow keys) to move "
          "the time cursor; the buttons jump it onto each fracture.")
    return fig, slider, buttons


# ----------------------------------- MAIN -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare the FT = 60 MPa mode A and mode AD runs: a static overlay of both "
                    "curves with their breaking points and the shared target load, and an "
                    "interactive mode-AD figure with a time cursor and its two fracture events.")
    parser.add_argument("folder", type=str, nargs="?", default=None,
                        help="Folder holding the paired '<MODE>_<FT>_disp' / '<MODE>_<FT>_force' "
                             "curves. Overrides DATA_FOLDER if given.")
    parser.add_argument("--target-load", type=float, default=None,
                        help="Target failure load [N] drawn as the red line on both curves. "
                             "Default: the mode AD entry in the FT-finetuning script's "
                             "TARGET_FAILURE_LOAD.")
    parser.add_argument("--no-target", action="store_true",
                        help="Leave the target line off the INTERACTIVE figure (it stays on the "
                             "static one, which is a comparison against it).")
    parser.add_argument("--dome-disp", type=float, default=None,
                        help="Pin the dome (second) fracture at the sample nearest this "
                             "displacement [mm] instead of detecting it as the top of the "
                             "collapse. The base (first) fracture is always detected.")
    parser.add_argument("--time-unit", type=str, default=None,
                        help="Unit to label the time cursor with (e.g. 'ms' or 's'). The exports "
                             "carry no unit, so by default the axis just reads 'Time'.")
    parser.add_argument("--post-drop-frac", type=float, default=0.01,
                        help="Load shed marking a fracture, as a fraction of the run's overall "
                             "peak. Sets the FIRST (base) event. Same meaning as in the "
                             "FT-finetuning script -- keep them equal or the figures disagree.")
    parser.add_argument("--arm-frac", type=float, default=0.10,
                        help="Ignore load sheds until the load has reached this fraction of the "
                             "run's peak (keeps the pre-contact wiggle out).")
    parser.add_argument("--settle-frac", type=float, default=0.05,
                        help="Collapse is settled once the load climbs this fraction of the "
                             "drop's depth back above its lowest point. Sets where the curves "
                             "are trimmed, and the search for the SECOND (dome) event.")
    parser.add_argument("--tail-frac", type=float, default=1.0,
                        help="How much of the collapse's displacement width to keep past the "
                             "bottom of the drop (1.0 = tail as wide as the drop).")
    parser.add_argument("--full", action="store_true",
                        help="Don't trim after the collapse -- plot the runs whole, crushed "
                             "plateau included.")
    parser.add_argument("--xmax", type=float, default=None, help="Right x-limit [mm].")
    parser.add_argument("--ymax", type=float, default=None, help="Top y-limit [N].")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for the static PNG (default: the data folder).")
    parser.add_argument("--static-only", action="store_true",
                        help="Build and save the static overlay only, no interactive figure.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the static figure without opening any window. Implies "
                             "--static-only: the interactive figure is nothing without one.")
    args = parser.parse_args()

    ft = load_ft_analysis()
    folder = Path(args.folder) if args.folder else DATA_FOLDER
    out_dir = Path(args.out_dir) if args.out_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    target_load = args.target_load
    if target_load is None:
        target_load = ft.TARGET_FAILURE_LOAD.get("AD")

    print(f"Reading FT = 60 MPa runs from '{folder}/'")

    runs = []
    for tag in (MODE_A_TAG, MODE_AD_TAG):
        mode, ft_value = ft.parse_tag(tag)
        time, disp, force = load_run(ft, folder, tag)

        base_idx, failed = find_base_event(ft, force, args)
        if not failed:
            print(f"  WARNING '{tag}': load never shed {args.post_drop_frac:.1%} of its peak -- "
                  f"this run may not have been carried to failure; marking the highest load reached.")

        # only the hybrid has a second fracture to look for: mode A breaks once
        dome_idx, has_dome = base_idx, False
        if mode == "AD" and failed:
            if args.dome_disp is not None:
                dome_idx, has_dome = override_event(disp, args.dome_disp), True
            else:
                dome_idx, has_dome = find_dome_event(ft, force, args, base_idx)
            if has_dome:
                check_dome_expectation(disp, force, dome_idx)

        keep_idx = max(base_idx, dome_idx if has_dome else base_idx)
        end = trimmed_end(ft, disp, force, args, keep_idx, failed)

        marks = {"base_idx": base_idx, "dome_idx": dome_idx,
                 "has_dome": has_dome, "end": end, "ft": ft_value}
        runs.append((mode, tag, time, disp, force, marks))

        print(f"  {tag}: mode {mode} ({ft.mode_label(mode)}), FT = {ft_value:g} MPa, "
              f"{len(time)} points")
        print(f"    1st fracture (BASE on AD): {force[base_idx]:7.2f} N @ "
              f"{disp[base_idx]:.4f} mm  (t = {time[base_idx]:.4f})")
        if has_dome:
            print(f"    2nd fracture (DOME)      : {force[dome_idx]:7.2f} N @ "
                  f"{disp[dome_idx]:.4f} mm  (t = {time[dome_idx]:.4f})")
        if target_load is not None:
            print(f"    reported breaking load vs {target_load:g} N target: "
                  f"{force[base_idx] - target_load:+.1f} N")

    static_path = out_dir / "modeA_vs_modeAD_FT060_static.png"
    plot_static(ft, runs, target_load, args, static_path)

    keep_alive = None
    if not (args.static_only or args.no_show):
        ad_run = next(r for r in runs if r[0] == "AD")
        keep_alive = plot_interactive(ft, ad_run, target_load, args)

    print()
    if not args.no_show:
        plt.show()
    return keep_alive


if __name__ == "__main__":
    main()
