import re
import pandas as pd


def _to_float(int_part, dec_part):
    return float(f"{int_part}.{dec_part}")


def _parse_metadata_value(line):
    """Extract a European-decimal numeric value from a metadata line like:
    'Thickness, 3,175,  "mm"' -> 3.175
    """
    m = re.match(r'^\s*\w+,\s*(-?\d+),(\d+)\s*,', line)
    if m:
        return _to_float(m.group(1), m.group(2))
    return None


def parse_mss_file(filepath, start_counter=1):
    """
    Parses one MTS-style .mss/.txt export containing one or more
    BeginSpecimen...EndSpecimen blocks, each with a BeginData...EndData
    section of Time/Load/Extension rows using European decimal commas.

    Returns (specimens, next_counter) where specimens is a list of
    (phial_number, thickness_mm, width_mm, dataframe) tuples, and
    next_counter is the phial number the NEXT file should start counting from
    (so batch numbering stays continuous across files).
    """
    specimens = []
    counter = start_counter

    thickness = None
    width = None
    in_data = False
    header_lines_to_skip = 0
    data_rows = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("BeginSpecimen"):
                thickness = None
                width = None
                data_rows = []
                in_data = False
                continue

            if line.startswith("BeginData"):
                in_data = True
                header_lines_to_skip = 2  # column-name line + units line
                continue

            if line.startswith("EndData"):
                in_data = False
                continue

            if line.startswith("EndSpecimen"):
                df = pd.DataFrame(
                    data_rows, columns=["Time_s", "Load_N", "Extension_mm"]
                )
                specimens.append((counter, thickness, width, df))
                counter += 1
                continue

            if in_data:
                if header_lines_to_skip > 0:
                    header_lines_to_skip -= 1
                    continue
                fields = [x.strip() for x in line.split(", ")]
                if len(fields) != 3:
                    continue
                try:
                    row = [float(v.replace(",", ".")) for v in fields]
                except ValueError:
                    continue
                data_rows.append(row)

    return specimens, counter


def canonicalize_breaking_mode(raw):
    """
    Turns free-text breaking-mode cells ('A-B, D-E', 'D, E, rottura totale',
    '//') into a compact sorted-letter code ('ABDE', 'DE', 'NA').

    Splits on any non-letter character and keeps only single-letter tokens
    in A-E, so descriptive words like 'rottura totale' (which happen to
    contain the letters a/e) are correctly ignored rather than misread as
    mode codes.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "UNK"
    tokens = re.split(r"[^A-Za-z]+", str(raw))
    letters = sorted({t.upper() for t in tokens if len(t) == 1 and t.upper() in "ABCDE"})
    return "".join(letters) if letters else "UNK"


def format_speed(speed):
    """2.0 -> '2', 2.5 -> '2p5', None/NaN -> 'NA'"""
    if speed is None or (isinstance(speed, float) and pd.isna(speed)):
        return "UNK"
    return f"{speed:g}".replace(".", "p")