import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main(args):

    df = pd.read_csv(args.file_path)

    property_name = ["Energy score", "Exicton size", "T2 - T1"]
    for i, property in enumerate(args.property):
        grouped = df.groupby("step").agg({property: ["mean", "std"]}).reset_index()

        grouped.columns = ["step", f"{property}_mean", f"{property}_std"]

        # Function to plot with similar style
        def plot_with_style(grouped, y_col_mean, y_col_std, y_label, title):

            label_size = 18
            tick_size = 16
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                grouped["step"],
                grouped[y_col_mean],
                yerr=grouped[y_col_std],
                fmt="-",
                color="black",
                ecolor="lightgray",
                elinewidth=1,
                capsize=2,
            )
            plt.xlabel("Epoch number", fontsize=label_size)
            plt.ylabel(y_label, fontsize=label_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.tight_layout()
            plt.savefig(f"{title}.png", dpi=400)
            plt.show()

        plot_with_style(
            grouped, f"{property}_mean", f"{property}_std", property_name[i], f"Step vs {property}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot optimization curves from RL iteration data.")
    parser.add_argument(
        "-f",
        "--f",
        "--f",
        dest="file_path",
        type=str,
        default="staged_learning_2.csv",
        help="Path to the CSV file containing the RL data.",
    )
    parser.add_argument(
        "-tn",
        "--target_names",
        dest="property",
        nargs="+",
        type=str,
        default=[
            "energy_score (raw)",
            "exciton_size (raw)",
            "T2-T1 (raw)",
            # "SA score (raw)",
        ],  # "SA score (raw)"
        help="List of target names (default=[energy_score (raw) or excit" "on_size (raw)])",
    )

    args = parser.parse_args()
    main(args)
