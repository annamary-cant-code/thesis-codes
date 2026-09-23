"""
PET-G adaptor stiffness, and the compliance correction it implies for Mode A.

FEM Mode A models the phial ALONE (~3655 N/mm), but the tests load it through
the printed PET-G adaptor and the machine reports the travel of the whole
stack -- so the measured curve is softer (~1300 N/mm) and the two are not
directly comparable.

  PART 1  Back-calculates the adaptor stiffness from the series assumption,
          1/K_tot = 1/K_phial + 1/K_petg  ->  K_petg = R/(R-1) * K_tot.
  PART 2  Measures it instead, from a compression test of the adaptor alone
          (PET-G_stiffness_raw.txt), conditioned exactly like the phial curves.
  PART 3  Adds the measured adaptor compliance back onto Mode A. Springs in
          series carry the same force and their displacements add:
              d_normalised(F) = d_modeA(F) + d_petg(F)
          giving what the FEM phial would have looked like measured through
          the adaptor -- comparable to the experimental cloud.

Outputs: petg_raw_curve_analysis.png, modeA_series_normalised_vs_EXP.png
"""

import re
import os
import glob
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIGURATION ==============================
HERE = Path(__file__).resolve().parent

# Reused wholesale: curve conditioning, phial curve loading, FEM history
# parsing. Its filename is not a legal module name, hence the by-path import.
COMPARISON_SCRIPT = HERE / "FEM-AN__stiffness_comparison_EXP-vs-FEM.py"

PETG_RAW_FILE = HERE / "PET-G_stiffness_raw.txt"
FEM_CURVES_DIR = HERE / "fem_load-vs-disp_curves"
MODE_A_TAG = "modeA"

OUTPUT_PLOT_PETG_RAW = "petg_raw_curve_analysis.png"
OUTPUT_PLOT_NORMALISED = "modeA_series_normalised_vs_EXP.png"

K_PHIAL_FEM = 3655   # [N/mm] gradient of the raw FEM Mode A curve
K_TOT_EXP = 1300     # [N/mm] gradient of the measured (stack) curves

# K_petg is quoted over the load band Mode A actually reaches -- its upper
# bound comes from the FEM curve at runtime (~316 N), so the stiffness isn't
# fitted over parts of the curve the normalisation never touches.
PETG_BAND_LOW_N = 20.0

COLOR_MODE_A = "#2ca02c"      # green -- clear of the orange/cyan batch colours
COLOR_NORMALISED = "#d62728"  # red
# ============================================================================


def load_comparison_module():
    """Import FEM-AN__stiffness_comparison_EXP-vs-FEM.py by path (its filename
    is not a legal module name). Its main() sits behind an __main__ guard."""
    if not COMPARISON_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Can't find '{COMPARISON_SCRIPT.name}' next to this script -- it holds the "
            f"curve conditioning and FEM history parsing this one reuses.")
    spec = importlib.util.spec_from_file_location("stiffness_comparison", COMPARISON_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------ PART 1 -------------------------------------

def analytical_estimate(k_phial_fem=K_PHIAL_FEM, k_tot_exp=K_TOT_EXP):
    r = k_phial_fem / k_tot_exp
    k_petg = r / (r - 1) * k_tot_exp

    print("PART 1 -- analytical estimate (series back-calculation)")
    print(f"  K_phial_FEM = {k_phial_fem:.0f} N/mm    K_tot_EXP = {k_tot_exp:.0f} N/mm    R = {r:.3f}")
    print(f"  K_petg inferred = {k_petg:.0f} N/mm")
    return k_petg


# ------------------------------ PART 2 -------------------------------------

def parse_mts_raw(path):
    """Reads the BeginData/EndData block of an MTS .mss-style text export.

    Italian locale: the decimal separator is a comma and so is the column
    separator, so splitting on commas mangles every number. Each data row is
    matched instead as exactly three comma-decimals (time, load, extension),
    which also skips the unit line for free.
    """
    rows = []
    in_data = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "BeginData":
            in_data = True
            continue
        if stripped == "EndData":
            break
        if not in_data:
            continue
        numbers = re.findall(r"-?\d+,\d+", stripped)
        if len(numbers) == 3:
            rows.append([float(n.replace(",", ".")) for n in numbers])

    if not rows:
        raise ValueError(f"No data rows parsed from '{path.name}' -- check the file format.")

    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def condition_petg_curve(cmp_mod, load, disp):
    """Trims the leading dead zone and shifts to (0, 0), using the same
    climb-start detection the phial curves get."""
    climb_start = cmp_mod.find_climb_start(load)
    if climb_start >= len(load):
        raise ValueError("PET-G curve never rises above the noise floor -- nothing to analyse.")
    return (disp[climb_start:] - disp[climb_start],
            load[climb_start:] - load[climb_start],
            climb_start)


def band_stiffness(disp, load, f_low, f_high):
    """Least-squares gradient of the PET-G curve over one load band."""
    mask = (load >= f_low) & (load <= f_high)
    if mask.sum() < 3:
        raise ValueError(f"Only {mask.sum()} PET-G samples between {f_low} and {f_high} N.")
    gradient, intercept = np.polyfit(disp[mask], load[mask], 1)
    return {"gradient": gradient, "intercept": intercept,
            "d_low": disp[mask].min(), "d_high": disp[mask].max(),
            "n_points": int(mask.sum())}


def measured_analysis(cmp_mod, f_band_high, k_petg_analytical):
    _, load, disp = parse_mts_raw(PETG_RAW_FILE)
    disp_shifted, load_shifted, climb_start = condition_petg_curve(cmp_mod, load, disp)
    band = band_stiffness(disp_shifted, load_shifted, PETG_BAND_LOW_N, f_band_high)

    print("\nPART 2 -- direct measurement (PET-G_stiffness_raw.txt)")
    print(f"  {len(load)} samples, {disp[climb_start]:.3f} mm of take-up discarded, "
          f"peak {load.max():.0f} N over {disp_shifted[load_shifted.argmax()]:.3f} mm")
    print(f"  K_petg measured = {band['gradient']:.0f} N/mm "
          f"(fit over {PETG_BAND_LOW_N:.0f}-{f_band_high:.0f} N, n={band['n_points']})")
    print(f"  measured / inferred = {band['gradient'] / k_petg_analytical:.3f}")

    return {"disp": disp_shifted, "load": load_shifted, "band": band}


# ------------------------------ PART 3 -------------------------------------

def build_compliance_lookup(disp, load):
    """Monotone force -> displacement lookup for np.interp. A real test curve
    wobbles, so only the points setting a new running maximum in load are
    kept, with (0, 0) prepended."""
    running_max = np.maximum.accumulate(np.concatenate(([-np.inf], load[:-1])))
    keep = load > running_max
    return (np.concatenate(([0.0], load[keep])),
            np.concatenate(([0.0], disp[keep])))


def petg_displacement_at(force, f_lookup, d_lookup):
    """Adaptor displacement at each force. Beyond the measured range the curve
    is extended along its terminal tangent, rather than held flat as np.interp
    would do on its own -- that would pretend the adaptor had gone rigid."""
    force = np.asarray(force, dtype=float)
    out = np.interp(np.clip(force, 0.0, None), f_lookup, d_lookup)

    over = force > f_lookup[-1]
    if np.any(over):
        k_end = (f_lookup[-1] - f_lookup[-2]) / (d_lookup[-1] - d_lookup[-2])
        print(f"  WARNING: {over.sum()} FEM sample(s) above the {f_lookup[-1]:.0f} N reached in "
              f"the PET-G test -- extrapolated along the terminal tangent ({k_end:.0f} N/mm).")
        out[over] = d_lookup[-1] + (force[over] - f_lookup[-1]) / k_end
    return out


def series_normalise(disp_fem, force_fem, petg):
    """d_normalised(F) = d_modeA(F) + d_petg(F)."""
    f_lookup, d_lookup = build_compliance_lookup(petg["disp"], petg["load"])
    d_petg = petg_displacement_at(force_fem, f_lookup, d_lookup)
    return disp_fem + d_petg, d_petg


def report_normalisation(disp_fem, force_fem, disp_norm, d_petg):
    top = int(np.argmax(force_fem))
    k_fem = np.polyfit(disp_fem, force_fem, 1)[0]
    k_norm = np.polyfit(disp_norm, force_fem, 1)[0]

    print("\nPART 3 -- series normalisation of Mode A")
    print(f"  at F = {force_fem[top]:.0f} N:  phial {disp_fem[top]:.4f} mm + adaptor "
          f"{d_petg[top]:.4f} mm = {disp_norm[top]:.4f} mm "
          f"({100 * d_petg[top] / disp_norm[top]:.0f}% adaptor)")
    print(f"  K: raw Mode A {k_fem:.0f} -> normalised {k_norm:.0f} N/mm "
          f"(K_tot_EXP = {K_TOT_EXP:.0f}, ratio {k_norm / K_TOT_EXP:.3f})")
    return k_fem, k_norm


# -------------------------------- PLOTS ------------------------------------

def plot_petg_raw(petg, f_band_high):
    band = petg["band"]
    plt.figure(figsize=(8, 5.5))

    plt.plot(petg["disp"], petg["load"], color="0.35", linewidth=1.4,
             label="PET-G, trimmed & shifted to (0,0)")
    d_line = np.array([band["d_low"], band["d_high"]])
    plt.plot(d_line, band["gradient"] * d_line + band["intercept"], color=COLOR_NORMALISED,
             linestyle="--", linewidth=1.2, label=f"band fit (K={band['gradient']:.0f} N/mm)")
    plt.axhspan(PETG_BAND_LOW_N, f_band_high, color="tab:orange", alpha=0.12,
                label=f"Mode A load band ({PETG_BAND_LOW_N:.0f}-{f_band_high:.0f} N)")

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.title("PET-G adaptor — conditioned curve and band stiffness")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PETG_RAW, dpi=150)
    plt.close()
    print(f"\nSaved {OUTPUT_PLOT_PETG_RAW}")


def plot_comparison(cmp_mod, curves, disp_fem, force_fem, disp_norm, k_fem, k_norm):
    """Experimental cloud, raw Mode A, normalised Mode A. Same axes and
    conventions as the full EXP-vs-FEM overlay, other FEM modes left out."""
    plt.figure(figsize=(11, 6))

    seen_batches = {}
    for c in curves:
        plt.plot(c["disp"], c["load"], color=cmp_mod.BATCH_COLORS.get(c["batch"], "gray"),
                 alpha=cmp_mod.CURVE_ALPHA, linewidth=1)
        seen_batches[c["batch"]] = seen_batches.get(c["batch"], 0) + 1
    for batch, count in sorted(seen_batches.items()):
        plt.plot([], [], color=cmp_mod.BATCH_COLORS.get(batch, "gray"), linewidth=2,
                 label=f"Batch {batch} (n={count})")

    plt.plot(disp_fem, force_fem, color=COLOR_MODE_A, linewidth=2,
             label=f"FEM — Mode A, phial only (K={k_fem:.0f} N/mm)")
    plt.plot(disp_norm, force_fem, color=COLOR_NORMALISED, linewidth=2,
             label=f"FEM — Mode A + PET-G in series (K={k_norm:.0f} N/mm)")

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.title("Mode A corrected for PET-G adaptor compliance (springs in series)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_NORMALISED, dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_PLOT_NORMALISED}")


# --------------------------------- MAIN ------------------------------------

def load_mode_a(cmp_mod):
    """Reads the Mode A disp/force pair, scaled as the overlay script scales
    it (kN -> N, displacement made positive)."""
    disp_path = FEM_CURVES_DIR / f"{MODE_A_TAG}_disp"
    force_path = FEM_CURVES_DIR / f"{MODE_A_TAG}_force"
    for path in (disp_path, force_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing '{path.name}' in {FEM_CURVES_DIR} -- both '{MODE_A_TAG}_disp' and "
                f"'{MODE_A_TAG}_force' are needed to build the Mode A curve.")
    return cmp_mod.build_combined_fem_curve(MODE_A_TAG, str(disp_path), str(force_path))


def load_experimental_curves(cmp_mod):
    exclude = cmp_mod.parse_manual_exclude(cmp_mod.MANUAL_EXCLUDE)
    filepaths = sorted(glob.glob(os.path.join(cmp_mod.CURVES_DIR, "*.xlsx")))
    curves = [c for c in (cmp_mod.process_one_file(fp, exclude)[0] for fp in filepaths)
              if c is not None]
    print(f"\nLoaded {len(curves)} of {len(filepaths)} phial curves.")
    return curves


def main():
    cmp_mod = load_comparison_module()

    k_petg_analytical = analytical_estimate()

    disp_fem, force_fem = load_mode_a(cmp_mod)
    petg = measured_analysis(cmp_mod, float(force_fem.max()), k_petg_analytical)
    plot_petg_raw(petg, float(force_fem.max()))

    disp_norm, d_petg = series_normalise(disp_fem, force_fem, petg)
    k_fem, k_norm = report_normalisation(disp_fem, force_fem, disp_norm, d_petg)

    curves = load_experimental_curves(cmp_mod)
    plot_comparison(cmp_mod, curves, disp_fem, force_fem, disp_norm, k_fem, k_norm)


if __name__ == "__main__":
    main()
