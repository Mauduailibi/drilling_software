
# post2.py
# -------------------------------------------------------------------------
# Comparative scatter plot for total operational time as a function of L1
# using:
#
#   ΔL1 = 10 m
#   ΔL1 = 1 m
#
# The script reads the CSV files already generated inside:
#
#   post_outputs/
#
# and creates:
#
#   post2_results/
#       └── total_time_vs_l1_comparison.png
#
# No title is added to the figure.
# -------------------------------------------------------------------------

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
POST_OUTPUTS = ROOT / "outputs" / "post_outputs"


# -------------------------------------------------------------------------
# INPUT FILES
# -------------------------------------------------------------------------

FILE_1 = (
    str(POST_OUTPUTS) + "/"
    "l1_sensitivity_fixed_Rstep_50_Ltraj_10_"
    "full_candidates_step_1p0.csv"
)

FILE_10 = (
    str(POST_OUTPUTS) + "/"
    "l1_sensitivity_fixed_Rstep_50_Ltraj_10_"
    "full_candidates_step_10p0.csv"
)


# -------------------------------------------------------------------------
# OUTPUT DIRECTORY
# -------------------------------------------------------------------------

OUTPUT_DIR = ROOT / "outputs" / "post2_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# LOAD BEST RESPONSE FOR EACH L1
# -------------------------------------------------------------------------

def load_best_per_l1(csv_path):

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Detect total-time column automatically
    # ------------------------------------------------------------

    total_col = None

    for col in df.columns:

        low = col.lower()

        if (
            "total" in low
            and "time" in low
        ):
            total_col = col
            break

    if total_col is None:

        raise ValueError(
            f"Total-time column not found in {csv_path}\n"
            f"Columns found:\n{list(df.columns)}"
        )

    # ------------------------------------------------------------
    # Detect L1 column automatically
    # ------------------------------------------------------------

    l1_col = None

    for col in df.columns:

        low = col.lower().replace(" ", "")

        if "l1" in low:
            l1_col = col
            break

    if l1_col is None:

        raise ValueError(
            f"L1 column not found in {csv_path}\n"
            f"Columns found:\n{list(df.columns)}"
        )

    # ------------------------------------------------------------
    # Select best total time for each L1
    # ------------------------------------------------------------

    idx = df.groupby(l1_col)[total_col].idxmin()

    best = (
        df.loc[idx, [l1_col, total_col]]
        .sort_values(l1_col)
        .reset_index(drop=True)
    )

    return best, l1_col, total_col


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():

    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    best_1, l1_col, total_col = load_best_per_l1(FILE_1)

    best_10, _, _ = load_best_per_l1(FILE_10)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------
    # ΔL1 = 1 m (background)
    # ------------------------------------------------------------


# ------------------------------------------------------------
# ΔL1 = 1 m (background)
# ------------------------------------------------------------

    ax.scatter(
        best_1[l1_col],
        best_1[total_col],
        s=28,
        marker="o",
        alpha=0.75,
        label=r"$\Delta L_1 = 1~\mathrm{m}$",
        zorder=1,
    )

    # ------------------------------------------------------------
    # ΔL1 = 10 m (foreground)
    # ------------------------------------------------------------

    ax.scatter(
        best_10[l1_col],
        best_10[total_col],
        s=38,
        marker="x",
        linewidths=1.8,
        label=r"$\Delta L_1 = 10~\mathrm{m}$",
        zorder=3,
    )



    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel(
        r"$L_1~(\mathrm{m})$",
        fontsize=18,
    )

    ax.set_ylabel(
        "Total operational time (h)",
        fontsize=18,
    )

    # ------------------------------------------------------------
    # Tick size
    # ------------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=14,
    )

    # ------------------------------------------------------------
    # Grid and legend
    # ------------------------------------------------------------

    ax.grid(True, alpha=0.3)

    ax.legend(
        fontsize=14,
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    plt.tight_layout()

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "total_time_vs_l1_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFigure saved to:\n")

    print(output_path.resolve())


# -------------------------------------------------------------------------
# EXECUTION
# -------------------------------------------------------------------------

if __name__ == "__main__":

    main()

