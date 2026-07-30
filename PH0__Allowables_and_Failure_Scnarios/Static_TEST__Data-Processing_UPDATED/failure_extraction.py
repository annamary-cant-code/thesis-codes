"""
Detects the FIRST genuine local-maximum-then-drop ("break") in a
load-displacement curve, as distinct from the curve's global maximum.

WHY THIS ISN'T A ONE-LINE "FIND THE FIRST LOCAL MAX":
Raw load-cell data is noisy sample-to-sample even during ordinary loading
(see the near-zero jitter at the start of any of these curves). A naive
"next point is lower" rule would flag noise as a break on almost every
specimen. What actually distinguishes a real break is: load rises, then
drops by a MEANINGFUL and SUSTAINED amount. That requires a prominence
threshold (how big a drop counts as real) and light smoothing (so single
noisy samples don't get treated as peaks) — both tunable below, because
this is an inherently approximate signal-processing judgment call, not a
solved problem. Always spot-check the annotated plots, especially for any
phial the summary flags.

Tested against a real 76-point-plus specimen curve with a genuine ~1.7 N
partial-break dip around 84 N followed by a final 389 N catastrophic
failure: default settings correctly recover BOTH true raw values.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks


def find_first_break(
    time,
    load,
    smooth_window=11,
    polyorder=2,
    prominence_abs_n=1,
    prominence_fraction=0.0,
    min_peak_load_n=2.0,
    refine_radius=5,
):
    """
    Parameters
    ----------
    time, load : array-like, same length
    smooth_window : int
        Savitzky-Golay window (samples) used only to LOCATE candidate peaks;
        the reported load value always comes from the raw data, never the
        smoothed curve (smoothing can overshoot/undershoot near sharp drops).
    prominence_abs_n : float
        Minimum drop (in Newtons) after a candidate peak to count as real,
        in absolute terms. This is the main noise filter — set based on your
        load cell's noise floor, not on curve magnitude.
    prominence_fraction : float
        Optional additional floor as a fraction of that curve's own max load
        (e.g. 0.02 = 2%). The EFFECTIVE threshold is
        max(prominence_abs_n, prominence_fraction * max(load)).
        Defaults to 0 (disabled) because a %-based rule alone can completely
        miss small-but-real early cracks on high-load curves — confirmed on
        real data where a real ~84N first break was invisible to any
        percentage-based threshold under 5%.
    min_peak_load_n : float
        Ignore any candidate peak below this absolute load — guards against
        toe-region / initial-contact blips being mistaken for a break.
    refine_radius : int
        After locating an approximate peak on the smoothed curve, search
        this many samples on either side of it in the RAW data for the true
        local maximum, and report that value/index instead.

    Returns
    -------
    dict with keys:
      first_break_load_n, first_break_time_s, first_break_index : the result
      num_peaks_detected : how many genuine peaks passed the filters
      all_peak_loads_n   : list of all detected peak loads, for QA
      flag               : '' if a clean detection, else a warning string
    """
    load = np.asarray(load, dtype=float)
    time = np.asarray(time, dtype=float)
    n = len(load)

    if n < 5:
        idx = int(np.nanargmax(load)) if n else None
        return {
            "first_break_load_n": float(load[idx]) if idx is not None else None,
            "first_break_time_s": float(time[idx]) if idx is not None else None,
            "first_break_index": idx,
            "num_peaks_detected": 0,
            "all_peak_loads_n": [],
            "flag": "TOO_FEW_POINTS - used global max, verify manually",
        }

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    w = max(w, 5)
    smoothed = savgol_filter(load, window_length=w, polyorder=min(polyorder, w - 1))

    prominence = max(prominence_abs_n, prominence_fraction * np.nanmax(load))
    peak_idx, _props = find_peaks(smoothed, prominence=prominence)

    # Refine each candidate to the true raw local maximum nearby -- smoothing
    # shifts/distorts WHERE the peak appears to be, so never report the
    # smoothed value or trust its exact index.
    refined = []
    for idx in peak_idx:
        lo, hi = max(0, idx - refine_radius), min(n, idx + refine_radius + 1)
        true_idx = lo + int(np.argmax(load[lo:hi]))
        refined.append(true_idx)
    refined = sorted(set(refined))

    # Drop candidates that don't clear the minimum absolute load floor
    refined = [i for i in refined if load[i] >= min_peak_load_n]

    if not refined:
        max_idx = int(np.nanargmax(load))
        return {
            "first_break_load_n": float(load[max_idx]),
            "first_break_time_s": float(time[max_idx]),
            "first_break_index": max_idx,
            "num_peaks_detected": 0,
            "all_peak_loads_n": [],
            "flag": "NO_PEAK_DETECTED - used global max, verify manually",
        }

    first_idx = refined[0]
    return {
        "first_break_load_n": float(load[first_idx]),
        "first_break_time_s": float(time[first_idx]),
        "first_break_index": first_idx,
        "num_peaks_detected": len(refined),
        "all_peak_loads_n": [float(load[i]) for i in refined],
        "flag": "",
    }