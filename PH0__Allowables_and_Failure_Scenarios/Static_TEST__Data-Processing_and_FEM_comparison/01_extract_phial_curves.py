"""
Extract load-displacement curves from two MTS-style compression test export
files into one .xlsx per phial, named with batch/speed/breaking-mode info
pulled from a companion Excel summary sheet.

HOW THE BATCH SWITCH WORKS
--------------------------
Phials are numbered 1..76 continuously as they are encountered, in file 1
then file 2, regardless of how many specimens actually sit in each file.
A single running counter is carried from file 1 into file 2 (see
`start_counter=next_counter` below), so the "36th phial -> switch to batch B"
rule works as a plain threshold check and does not care where the file
boundary actually falls.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

from mss_parser import parse_mss_file, canonicalize_breaking_mode, format_speed

# ============================== CONFIGURATION ==============================
TEXT_FILE_1 = "raw_collection_1.txt"
TEXT_FILE_2 = "raw_collection_2.txt"
EXCEL_FILE = "failure_load_data.xlsx"

EXCEL_PHIAL_COL = "Phial Num."
EXCEL_SPEED_COL = "Velocità (mm/min)"
EXCEL_MODE_COL = "Modo rottura"

OUTPUT_DIR = "output_curves"

BATCH_A_LABEL = "A"
BATCH_B_LABEL = "B"
BATCH_SWITCH_AT = 37  # phial numbers >= this value belong to batch B

PLOT_EXEMPLARY_GRAPH = True
PLOT_PHIAL_NUMBER = 36          # global phial number (1-76) to plot
PLOT_OUTPUT_FILE = None        # None -> auto-named from that phial's tag
# ============================================================================


def load_phial_info(excel_path):
    """Returns {phial_number: {'speed': float|None, 'mode': 'ABDE'}}"""
    df_info = pd.read_excel(excel_path)
    df_info.columns = [str(c).strip() for c in df_info.columns]

    lookup = {}
    for _, row in df_info.iterrows():
        try:
            pnum = int(row[EXCEL_PHIAL_COL])
        except (ValueError, TypeError, KeyError):
            continue
        speed = row.get(EXCEL_SPEED_COL, None)
        mode_raw = row.get(EXCEL_MODE_COL, None)
        lookup[pnum] = {
            "speed": speed if pd.notna(speed) else None,
            "mode": canonicalize_breaking_mode(mode_raw),
        }
    return lookup


def build_filename(phial_number, batch, speed, mode):
    speed_str = format_speed(speed)
    return f"{phial_number:02d}{batch}_s{speed_str}_m{mode}.xlsx"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    info_lookup = load_phial_info(EXCEL_FILE)

    specimens_1, next_counter = parse_mss_file(TEXT_FILE_1, start_counter=1)
    specimens_2, next_counter = parse_mss_file(TEXT_FILE_2, start_counter=next_counter)
    all_specimens = specimens_1 + specimens_2

    print(f"Parsed {len(all_specimens)} specimens total (expected 76).")
    if len(all_specimens) != 76:
        print("  -> Count mismatch: check BeginSpecimen/EndSpecimen pairing "
              "in the source files before trusting the batch split.")

    for phial_number, thickness, width, df in all_specimens:
        batch = BATCH_A_LABEL if phial_number < BATCH_SWITCH_AT else BATCH_B_LABEL
        info = info_lookup.get(phial_number, {"speed": None, "mode": "NA"})
        speed, mode = info["speed"], info["mode"]

        filename = build_filename(phial_number, batch, speed, mode)
        filepath = os.path.join(OUTPUT_DIR, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            meta = pd.DataFrame({
                "Field": ["Phial number", "Batch", "Thickness_mm", "Width_mm",
                          "Speed_mm_per_min", "Breaking_mode"],
                "Value": [phial_number, batch, thickness, width, speed, mode],
            })
            meta.to_excel(writer, sheet_name="Metadata", index=False)

        if PLOT_EXEMPLARY_GRAPH and phial_number == PLOT_PHIAL_NUMBER:
            plt.figure(figsize=(7, 5))
            plt.plot(df["Extension_mm"], df["Load_N"])
            plt.xlabel("Displacement (mm)")
            plt.ylabel("Load (N)")
            plt.title(f"Load-Displacement — Phial {phial_number}{batch}")
            plt.grid(True)
            plot_name = PLOT_OUTPUT_FILE or filename.replace(".xlsx", ".png")
            plot_path = os.path.join(OUTPUT_DIR, plot_name)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved exemplary plot: {plot_path}")

    print(f"Done. {len(all_specimens)} files written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()