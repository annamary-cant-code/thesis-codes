"""
Statistical analysis of breaking mode vs. stiffness and failure load.

Merges phial_failure_summary.xlsx (breaking mode + failure load, more
complete) with phial_modulus_clean.xlsx (stiffness, smaller -- some phials
got excluded/flagged upstream and simply have no valid modulus). Every
per-mode statistic reports its own n honestly rather than pretending both
datasets are the same size.

Produces three files:
  1. breaking_mode_pie.png    -- pie chart (slice = mode/combo) + a companion
     stats table (n, stiffness range, failure-load range, batch mix) in the
     same figure, since cramming all of that as text directly on/around a
     pie slice becomes unreadable once there are more than a couple of
     small slices.
  2. failure_load_by_mode.png -- one row per breaking mode: raw points
     (jittered) + a box plot, NOT a fitted distribution -- some modes may
     only have a handful of phials, where a fitted curve would be
     statistically meaningless. A box plot degrades gracefully at any n.
  3. stiffness_by_mode.png    -- same idea, for stiffness.

Mode -> color mapping is identical across all three images, so a slice in
the pie chart and its row in the distribution plots are visually the same
color, without needing to re-read labels to connect them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch

# ============================== CONFIGURATION ==============================
FAILURE_SUMMARY_EXCEL = "phial_failure_summary.xlsx"
MODULUS_CLEAN_EXCEL = "phial_modulus_clean.xlsx"

OUTPUT_DIR = "mode_analysis"
PIE_CHART_FILE = "breaking_mode_pie.png"
PIE_ONLY_FILE = "breaking_mode_pie_only.png"
TABLE_ONLY_FILE = "breaking_mode_table_only.png"
FAILURE_LOAD_DIST_FILE = "failure_load_by_mode.png"
STIFFNESS_DIST_FILE = "stiffness_by_mode.png"
MODE_STATS_EXCEL = "breaking_mode_statistics.xlsx"

JITTER_WIDTH = 0.25
ROW_HEIGHT_INCHES = 1.1   # per mode, so the figure scales with however many modes are found

# Single-letter breaking-mode codes, translated from the original Italian
# "Modo rottura" / "Sigla" labels. A phial's mode can be a combination of
# these (e.g. "AD" = dome + base), and "UNK" means the mode wasn't identified.
MODE_LEGEND = {
    "A": "Dome",
    "B": "Bulging",
    "C": "Indentation",
    "D": "Base",
    "E": "Cylinder",
}
# ============================================================================


def mode_legend_caption():
    codes = "   ".join(f"{code} = {name}" for code, name in MODE_LEGEND.items())
    return (f"Mode codes: {codes}   |   combined letters (e.g. \"AD\") = multiple "
            f"modes on the same phial   |   UNK = mode not identified")


def load_and_merge():
    failure = pd.read_excel(FAILURE_SUMMARY_EXCEL)
    modulus = pd.read_excel(MODULUS_CLEAN_EXCEL)

    modulus_cols = ["Phial_Number", "Batch", "Tangent_Modulus_N_per_mm"]
    missing_in_modulus = [c for c in modulus_cols if c not in modulus.columns]
    if missing_in_modulus:
        raise ValueError(
            f"MODULUS_CLEAN_EXCEL ('{MODULUS_CLEAN_EXCEL}') is missing expected column(s) "
            f"{missing_in_modulus}. Its actual columns are: {list(modulus.columns)}. "
            f"Check that MODULUS_CLEAN_EXCEL points at phial_modulus_clean.xlsx, not "
            f"some other file."
        )

    # If FAILURE_SUMMARY_EXCEL already happens to have a same-named column
    # (e.g. the two config paths accidentally point at overlapping-schema
    # files), a plain merge() won't error -- it silently renames BOTH to
    # 'Tangent_Modulus_N_per_mm_x' / '_y' instead, which only surfaces later
    # as a confusing KeyError with no clue what actually went wrong. Catch
    # it here instead, with an actionable message.
    overlap = (set(modulus_cols) - {"Phial_Number", "Batch"}) & set(failure.columns)
    if overlap:
        print(f"WARNING: FAILURE_SUMMARY_EXCEL already contains column(s) {sorted(overlap)} "
              f"-- are FAILURE_SUMMARY_EXCEL and MODULUS_CLEAN_EXCEL pointing at the right, "
              f"distinct files? Dropping the pre-existing column(s) from the failure-summary "
              f"side and using MODULUS_CLEAN_EXCEL's values instead.")
        failure = failure.drop(columns=list(overlap))

    if "Breaking_Mode" not in failure.columns:
        raise ValueError(
            f"FAILURE_SUMMARY_EXCEL ('{FAILURE_SUMMARY_EXCEL}') has no 'Breaking_Mode' column. "
            f"Its actual columns are: {list(failure.columns)}. This script needs "
            f"phial_failure_summary.xlsx as produced by 02_process_failure_data.py."
        )

    return failure.merge(modulus[modulus_cols], on=["Phial_Number", "Batch"], how="left")


def compute_mode_stats(df):
    n_grand_total = len(df)
    # denominator for the "share of batch" column below: each batch's OWN
    # total phial count across every mode, not just this mode's group
    batch_totals = df["Batch"].value_counts().sort_index()
    all_batches = list(batch_totals.index)

    rows = []
    for mode, group in df.groupby("Breaking_Mode"):
        stiff = group["Tangent_Modulus_N_per_mm"].dropna()
        load = group["Failure_Load_N"].dropna()

        # "of THIS MODE's phials, what fraction came from each batch" --
        # rows sum to 100% (already existed)
        batch_counts = group["Batch"].value_counts(normalize=True).sort_index() * 100
        batch_str = ", ".join(f"{b}: {p:.0f}%" for b, p in batch_counts.items())

        # "of EACH BATCH's phials, what fraction broke via this mode" --
        # a DIFFERENT normalization (denominator is the batch's grand total,
        # not this mode's group), so e.g. two rows can't be compared by
        # eye against Batch_Composition -- a batch that's just bigger overall
        # would look inflated there but not here
        mode_batch_counts = group["Batch"].value_counts()
        batch_share_str = ", ".join(
            f"{b}: {100 * mode_batch_counts.get(b, 0) / batch_totals[b]:.0f}%" for b in all_batches
        )

        rows.append({
            "Breaking_Mode": mode,
            "N_Phials": len(group),
            "Pct_of_Total": 100 * len(group) / n_grand_total,
            "N_With_Stiffness": len(stiff),
            "Stiffness_Min_N_per_mm": stiff.min() if len(stiff) else np.nan,
            "Stiffness_Max_N_per_mm": stiff.max() if len(stiff) else np.nan,
            "Failure_Load_Min_N": load.min() if len(load) else np.nan,
            "Failure_Load_Max_N": load.max() if len(load) else np.nan,
            "Batch_Composition": batch_str,
            "Batch_Mode_Share": batch_share_str,
        })
    return pd.DataFrame(rows).sort_values("N_Phials", ascending=False).reset_index(drop=True)


def build_mode_colors(modes):
    cmap = plt.get_cmap("tab20" if len(modes) > 10 else "tab10")
    return {m: cmap(i / max(len(modes) - 1, 1)) for i, m in enumerate(modes)}


SMALL_SLICE_PCT_THRESHOLD = 3.0   # slices at/below this share get pulled into the column beside the pie


def draw_pie(ax_pie, stats_df, mode_colors, pie_fontsize=18, label_fontsize=16,
             col_x=1.2, row_gap=0.18, col_y_bottom=0.7):
    """Draws the breaking-mode pie (with the small-slice leader-line labels)
    onto ax_pie. Font sizes and label layout are parameters so the standalone
    pie-only figure can render everything larger than the combined figure
    has room for."""
    colors = [mode_colors[m] for m in stats_df["Breaking_Mode"]]

    # Slices at or below the threshold are pulled out of the pie's normal
    # labeling entirely (both the outer name label AND the inline %) and
    # combined into one "Mode — n%" leader-line label instead -- showing the
    # percentage in both places is what made this cluster doubly crowded
    # before (mode-name overlapping mode-name, AND number overlapping
    # number, right on top of each other).
    def autopct(pct):
        return f"{pct:.0f}%" if pct > SMALL_SLICE_PCT_THRESHOLD else ""

    wedges, mode_texts, _ = ax_pie.pie(
        stats_df["N_Phials"], labels=stats_df["Breaking_Mode"], colors=colors,
        autopct=autopct, startangle=90,
        wedgeprops=dict(edgecolor="white"),
        textprops={"fontsize": pie_fontsize},
    )

    # The small slices all land crammed together right next to each other
    # (they're the last ones placed, sweeping back around toward the start)
    # in a narrow arc at the TOP of the pie. Rather than spreading their
    # labels around the page, they're listed as one tidy LEFT-ALIGNED COLUMN
    # at a fixed x, starting top-right and reading straight down -- same
    # column edge for every label, like a legend.
    small_idx = [i for i, pct in enumerate(stats_df["Pct_of_Total"]) if pct <= SMALL_SLICE_PCT_THRESHOLD]
    if small_idx:
        anchors, angles_deg = [], []
        for i in small_idx:
            ang_deg = (wedges[i].theta1 + wedges[i].theta2) / 2
            ang_rad = np.deg2rad(ang_deg)
            anchors.append((np.cos(ang_rad), np.sin(ang_rad)))
            angles_deg.append(ang_deg)

        # Every connector is exactly TWO straight segments -- one radial,
        # one horizontal -- and then the text. The radial leg simply keeps
        # going out along the wedge's own angle until it reaches the height
        # of that label's row; the horizontal leg runs from there straight
        # to the text. No third (vertical) leg, so nothing runs alongside
        # anything else.
        #
        # This is why the whole label block sits ABOVE the pie: a ray leaving
        # a wedge in the upper half only ever climbs, so a row can only be
        # reached by the radial leg if it is higher than where that ray
        # starts. Rows below the pie's top would need the line to double
        # back, which is exactly what forced the extra vertical leg before.
        #
        # Row assignment is what keeps the fan from tangling: rows go in
        # ASCENDING angle order (shallowest wedge -> lowest row, steepest
        # wedge -> top row). Two rays out of the same centre can never cross
        # each other, and with this ordering a steeper ray always meets a
        # lower row's height further LEFT than that row's own elbow, so it
        # can never clip a horizontal leg either.
        order = sorted(range(len(small_idx)), key=lambda k: angles_deg[k])
        row_ys = [col_y_bottom + row_gap * rank for rank in range(len(order))]

        for rank, k in enumerate(order):
            i = small_idx[k]
            mode_texts[i].set_visible(False)
            row_y = row_ys[rank]

            # where this wedge's own radial ray reaches that row's height
            tan_a = np.tan(np.deg2rad(angles_deg[k]))
            elbow_x = row_y / tan_a if tan_a > 0 else col_x - 0.3
            elbow_x = min(elbow_x, col_x - 0.15)   # always leave a visible horizontal leg

            squared_path = Path(
                [anchors[k], (elbow_x, row_y), (col_x, row_y)],
                [Path.MOVETO, Path.LINETO, Path.LINETO],
            )
            ax_pie.add_patch(FancyArrowPatch(
                path=squared_path, arrowstyle="-|>", mutation_scale=12,
                color="gray", lw=1.2, zorder=1,
            ))

            label = f"{stats_df['Breaking_Mode'].iloc[i]} — {stats_df['Pct_of_Total'].iloc[i]:.0f}%"
            ax_pie.text(col_x + 0.12, row_y, label, fontsize=label_fontsize, ha="left", va="center")
        ax_pie.set_xlim(-1.3, col_x + 1.3)
        ax_pie.set_ylim(-1.35, row_ys[-1] + 0.3)

    ax_pie.set_title(f"Breaking mode distribution (n={int(stats_df['N_Phials'].sum())} phials)",
                      fontsize=pie_fontsize + 2, pad=20)


def draw_stats_table(ax_table, stats_df, mode_colors, fontsize=12):
    """Draws the per-mode stats table onto ax_table."""
    ax_table.axis("off")
    col_labels = ["Mode", "n", "Stiffness range (N/mm)", "Failure load range (N)",
                  "Batch mix (of this mode)", "% of batch with this mode"]
    cell_text = []
    for _, row in stats_df.iterrows():
        if pd.notna(row["Stiffness_Min_N_per_mm"]):
            stiff_range = f"{row['Stiffness_Min_N_per_mm']:.0f}-{row['Stiffness_Max_N_per_mm']:.0f} (n={int(row['N_With_Stiffness'])})"
        else:
            stiff_range = "n/a"
        load_range = f"{row['Failure_Load_Min_N']:.0f}-{row['Failure_Load_Max_N']:.0f}"
        cell_text.append([row["Breaking_Mode"], int(row["N_Phials"]), stiff_range, load_range,
                           row["Batch_Composition"], row["Batch_Mode_Share"]])

    # bbox=[0,0,1,1] stretches the table to fill the ENTIRE axis, rather
    # than loc="center" auto-sizing to its own content and leaving whatever
    # blank margin is left in the axis around it -- that leftover margin is
    # exactly what showed up as unwanted empty space bordering the table.
    table = ax_table.table(cellText=cell_text, colLabels=col_labels, bbox=[0, 0, 1, 1], cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # bold + enlarge the header row
    for col_idx in range(len(col_labels)):
        table[0, col_idx].set_text_props(weight="bold", fontsize=fontsize)

    # color the "Mode" column cell to match its pie slice
    for row_idx, mode in enumerate(stats_df["Breaking_Mode"], start=1):  # +1 to skip header row
        cell = table[row_idx, 0]
        cell.set_facecolor(mode_colors[mode])
        cell.set_text_props(weight="bold", fontsize=fontsize)

    return table


def plot_pie_with_stats(stats_df, mode_colors, out_path):
    n_modes = len(stats_df)
    # Height is whichever needs more room: the table (one row per mode at its
    # font size) or a floor big enough for the pie to render at a readable
    # size. Matplotlib includes each axes' FULL allocated box in the tight
    # bbox regardless of how much of it its content actually uses (confirmed
    # earlier in this file's history) -- so going any taller than this max
    # brings back the huge blank margin above/below both panels that was
    # fixed before; going shorter than the pie's floor just shrinks the pie
    # again, since an aspect-locked circle is capped by whichever of its
    # axes box's width/height is smaller.
    table_height = 0.42 * n_modes + 2.2
    pie_height_floor = 9.5
    fig_height = max(table_height, pie_height_floor)
    # Wide, and with the pie given a bigger absolute width share, so the pie
    # circle renders large -- it previously looked tiny because the small-
    # slice label row forces a wide x/y data range, and on an aspect-locked
    # pie that data range (not the subplot's pixel size) is what determines
    # how big the circle actually draws.
    fig, (ax_pie, ax_table) = plt.subplots(
        1, 2, figsize=(26, fig_height), gridspec_kw={"width_ratios": [1.3, 2.0]}
    )

    draw_pie(ax_pie, stats_df, mode_colors)
    draw_stats_table(ax_table, stats_df, mode_colors, fontsize=15)

    # negative y leaves a clear gap between the bottom of the table and this
    # caption -- tight bbox then grows to include it, keeping the gap
    fig.text(0.5, -0.05, mode_legend_caption(), ha="center", va="bottom", fontsize=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pie_only(stats_df, mode_colors, out_path):
    fig, ax_pie = plt.subplots(figsize=(11, 10))
    draw_pie(ax_pie, stats_df, mode_colors, pie_fontsize=19, label_fontsize=16,
             col_x=1.72, row_gap=0.19, col_y_bottom=1.05)
    fig.text(0.5, -0.03, mode_legend_caption(), ha="center", va="bottom", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_table_only(stats_df, mode_colors, out_path):
    n_modes = len(stats_df)
    fig, ax_table = plt.subplots(figsize=(16, 0.5 * n_modes + 1.5))
    draw_stats_table(ax_table, stats_df, mode_colors, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distribution_by_mode(df, value_col, xlabel, title, stats_df, mode_colors, out_path, seed=0):
    rng = np.random.default_rng(seed)
    # order rows to match the pie/table (largest n first), top-to-bottom
    modes_ordered = list(stats_df["Breaking_Mode"])
    n_modes = len(modes_ordered)

    fig, ax = plt.subplots(figsize=(9, ROW_HEIGHT_INCHES * n_modes + 1))

    counts = {}
    for i, mode in enumerate(modes_ordered):
        y = n_modes - 1 - i  # first mode in the list plotted at the TOP
        vals = df.loc[df["Breaking_Mode"] == mode, value_col].dropna().values
        counts[mode] = len(vals)
        color = mode_colors[mode]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(vals))
        ax.scatter(vals, np.full(len(vals), y) + jitter, color=color, alpha=0.75,
                    s=28, edgecolor="black", linewidth=0.4, zorder=3)
        if len(vals) >= 2:
            ax.boxplot(vals, positions=[y], orientation="horizontal", widths=0.5, manage_ticks=False,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.5),
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color=color), capprops=dict(color=color),
                        zorder=2)

    ax.set_yticks(range(n_modes))
    ax.set_yticklabels([f"{modes_ordered[n_modes-1-i]} (n={counts[modes_ordered[n_modes-1-i]]})"
                          for i in range(n_modes)])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # mirror the x-axis ticks/labels at the top too, so the values are
    # readable without scrolling down past a tall (many-mode) figure
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlabel(xlabel)

    fig.text(0.5, 0.0, mode_legend_caption(), ha="center", va="bottom", fontsize=10.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_merge()
    stats_df = compute_mode_stats(df)

    stats_df.to_excel(os.path.join(OUTPUT_DIR, MODE_STATS_EXCEL), index=False)
    print(stats_df.to_string(index=False))

    modes_by_size = list(stats_df["Breaking_Mode"])
    mode_colors = build_mode_colors(modes_by_size)

    plot_pie_with_stats(stats_df, mode_colors, os.path.join(OUTPUT_DIR, PIE_CHART_FILE))
    print(f"Saved {PIE_CHART_FILE}")

    plot_pie_only(stats_df, mode_colors, os.path.join(OUTPUT_DIR, PIE_ONLY_FILE))
    print(f"Saved {PIE_ONLY_FILE}")

    plot_table_only(stats_df, mode_colors, os.path.join(OUTPUT_DIR, TABLE_ONLY_FILE))
    print(f"Saved {TABLE_ONLY_FILE}")

    load_stats = stats_df  # already sorted by N_Phials -- same order as the pie chart
    plot_distribution_by_mode(
        df, "Failure_Load_N", "Failure load (N)", "Failure load by breaking mode",
        load_stats, mode_colors, os.path.join(OUTPUT_DIR, FAILURE_LOAD_DIST_FILE),
    )
    print(f"Saved {FAILURE_LOAD_DIST_FILE}")

    # SAME row order as above (not re-sorted by its own n) -- so a given mode
    # sits in the same row position in both images, making it possible to
    # read off "this mode's load AND stiffness" from one row at a glance.
    plot_distribution_by_mode(
        df, "Tangent_Modulus_N_per_mm", "Tangent modulus / stiffness (N/mm)", "Stiffness by breaking mode",
        stats_df, mode_colors, os.path.join(OUTPUT_DIR, STIFFNESS_DIST_FILE),
    )
    print(f"Saved {STIFFNESS_DIST_FILE}")


if __name__ == "__main__":
    main()