import os

import numpy as np
import pandas as pd
from critdd import Diagram
from plotnine import (
    aes,
    geom_boxplot,
    geom_histogram,
    geom_line,
    geom_point,
    ggplot,
    theme_matplotlib,
    xlab,
    ylab,
    ylim,
)
from scipy.stats import rankdata


def plot_histogram_of_significant_performances(df):
    df = df.copy()

    plot_df = (
        df.groupby(["trial", "snapshot", "plate"])["corrected_auroc_significant"]
        .sum()
        .reset_index()
    )
    num_algs = df["algorithm"].nunique()
    num_tables = len(plot_df)
    p = (
        ggplot(plot_df)
        + geom_histogram(aes(x="corrected_auroc_significant"))
        + xlab(
            "Number of algorithms with Bonferroni-corrected significant performance "
            f"(/{num_algs})"
        )
        + ylab(f"Number of tables (/{num_tables})")
    )

    return p


def plot_critical_difference_diagram(df):
    df = df.copy()

    plot_df = df.pivot(
        index=("trial", "snapshot", "plate"), columns="algorithm", values="auroc"
    )
    plot_df = plot_df.dropna(how="all").fillna(0.0)  # dropna(how="any", axis=1)
    diagram = Diagram(
        plot_df.to_numpy(), treatment_names=plot_df.columns, maximize_outcome=True
    )

    print("Algorithms: ", plot_df.columns)
    print("Average ranks: ", diagram.average_ranks)
    print("Groups: ", diagram.get_groups(alpha=0.05, adjustment="holm"))

    print("Mean performance:")
    print(plot_df.mean())

    return diagram, plot_df.columns


def print_info_on_best_algorithm(df, average_ranks, algorithm_names):
    best_algorithm = algorithm_names[np.argmin(average_ranks)]

    df_sub = df[df["algorithm"] == best_algorithm].copy()
    num_significant_tables = df_sub["corrected_auroc_significant"].sum()
    frac_significant_tables = num_significant_tables / len(df_sub)

    num_significant_algorithms = df.groupby(["trial", "snapshot", "plate"])[
        "corrected_auroc_significant"
    ].sum()
    iforest_significant = df_sub.groupby(["trial", "snapshot", "plate"])[
        "corrected_auroc_significant"
    ].sum()
    _df = df.pivot(
        index=("trial", "snapshot", "plate"), columns="algorithm", values="auroc"
    ).fillna(0.0)
    num_insignificant_iforest_loses_tables = (
        (num_significant_algorithms > 0) & (iforest_significant == 0)
    ).sum()

    print("Best algorithm: ", best_algorithm)
    print(
        "% of tables where best is significant (#): ",
        frac_significant_tables,
        num_significant_tables,
    )
    print(
        (
            "# of tables where best algorithm is insignificant and loses to a "
            "significant performer: "
        ),
        num_insignificant_iforest_loses_tables,
    )
    print("Winner breakdown: ", _df.idxmax(axis=1).value_counts() / len(_df))


def trial_by_snapshot_success(df):
    plot_df = (
        (
            df.groupby(["trial", "snapshot_number", "plate"])[
                "corrected_auroc_significant"
            ].sum()
            > 0
        )
        .reset_index()
        .groupby(["trial", "snapshot_number"])["corrected_auroc_significant"]
        .mean()
        .reset_index()
    )

    plt = (
        ggplot(
            plot_df,
            aes(
                x="snapshot_number",
                y="corrected_auroc_significant",
                color="trial",
                group="trial",
            ),
        )
        + geom_line()
        + geom_point()
        + ylim([0, 1])
        + xlab("snapshot #")
        + ylab("fraction of datasets successful")
        + theme_matplotlib()
    )

    return plt


def trial_by_snapshot_irregularity_prop(df):
    plot_df = df[
        ["trial", "plate", "snapshot_number", "anomaly_proportion"]
    ].drop_duplicates()
    plot_df["snapshot_number"] = pd.Categorical(plot_df["snapshot_number"])
    plt = ggplot(plot_df) + geom_boxplot(
        aes(x="snapshot_number", y="anomaly_proportion", color="trial")
    )

    return plt


def comparison_tables(df):
    _df = (
        df.pivot(
            index=("trial", "snapshot_number", "plate"),
            columns="algorithm",
            values="auroc",
        )
        .dropna(how="all")
        .fillna(0.0)
    )
    best_idx = np.argmax(rankdata(_df, axis=1).mean(axis=0))
    best_algo = _df.columns[best_idx]

    df_sub = df[df["algorithm"] == best_algo]
    gb_performance_table = (
        df_sub.groupby(["trial", "snapshot_number"])["auroc"]
        .mean()
        .reset_index()
        .pivot(index="trial", columns="snapshot_number", values="auroc")
    )

    extra_trial_data = pd.DataFrame(
        [
            ("poise", 8351, 190, 23),
            ("rely", 18113, 951, 44),
            ("hope3", 12705, 228, 21),
            ("tips3", 5713, 86, 9),
            ("compass", 27395, 602, 33),
            ("hipattack", 2970, 69, 17),
            ("manage", 1754, 84, 19),
        ],
        columns=["trial", "\\# participants", "\\# centres", "\\# countries"],
    ).set_index("trial")
    mapping = {
        "plate": "count",
        "num_samples": "mean",
        "num_columns": "mean",
        "anomaly_proportion": "mean",
    }
    summary_table = df_sub.groupby("trial").agg(mapping)
    overall = df_sub.agg(mapping).to_frame().T
    overall.index = ["\\textbf{total}"]
    summary_table = pd.concat((summary_table, overall), axis=0)
    summary_table.columns = [
        "\\# datasets",
        "\\# instances (avg.)",
        "\\# features (avg.)",
        "\\% irregular (avg.)",
    ]
    summary_table["\\% irregular (avg.)"] = summary_table["\\% irregular (avg.)"] * 100
    summary_table = extra_trial_data.join(summary_table, how="outer")

    return (
        gb_performance_table,
        summary_table,
        _df.groupby("trial").mean().iloc[:, [3, 0, 5]],
    )


def print_info_on_all_algorithms(df):
    at_least_one_significant = (
        df.groupby(["trial", "snapshot", "plate"])["corrected_auroc_significant"].sum()
        > 0
    )
    total_num_plates = len(at_least_one_significant)

    print(
        "# of plates where at least one algorithm is significant: ",
        at_least_one_significant.sum(),
    )
    print("Total # plates: ", total_num_plates)


def main(args):
    os.makedirs(args.output_directory, exist_ok=True)

    trial_summary_fns = os.listdir(args.input_directory)
    summary_dfs = []
    for trial_summary_fn in trial_summary_fns:
        trial_name = trial_summary_fn.split(".")[0]
        trial_summary_path = os.path.join(args.input_directory, trial_summary_fn)
        summary_df = pd.read_csv(trial_summary_path)
        summary_df["trial"] = trial_name
        summary_dfs.append(summary_df)
    df = pd.concat(summary_dfs, axis=0)
    df["corrected_auroc_significant"] = df["auroc.lower"] > 0.5

    # Add snapshot number for cross-trial comparison.
    trial_dfs = []
    for trial_name in df["trial"].unique():
        trial_df = df[df["trial"] == trial_name].copy()
        snapshot_names = trial_df["snapshot"].unique()
        snapshot_idxs = snapshot_names.argsort()
        if max(snapshot_idxs) < 4:
            snapshot_idxs += 4 - max(snapshot_idxs)
        snapshot_mapper = dict(zip(snapshot_names, snapshot_idxs))
        trial_df["snapshot_number"] = trial_df["snapshot"].map(snapshot_mapper)
        trial_dfs.append(trial_df)
    df = pd.concat(trial_dfs, axis=0)

    print_info_on_all_algorithms(df)

    df.loc[df["algorithm"].str.startswith("IForest"), "algorithm"] = "IForest"

    output_path = os.path.join(args.output_directory, "significant_performances.jpg")
    plt = plot_histogram_of_significant_performances(df)
    plt.save(output_path)

    output_path = os.path.join(args.output_directory, "cd_diagram.tex")
    diagram, algorithm_names = plot_critical_difference_diagram(df)
    diagram.to_file(
        output_path,
        alpha=0.05,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": "Algorithm performance across all trials"},
    )

    print_info_on_best_algorithm(df, diagram.average_ranks, algorithm_names)

    gb_performance_table, summary_table, average_performance = comparison_tables(df)

    output_path = os.path.join(
        args.output_directory, "global_best_performance_table.tex"
    )
    gb_performance_table.to_latex(output_path, na_rep="-", float_format="%.2f")
    output_path = os.path.join(args.output_directory, "global_best_summary_table.tex")
    summary_table.to_latex(output_path, na_rep="-", float_format="%.1f")
    output_path = os.path.join(args.output_directory, "trial_average_performances.tex")
    average_performance.to_latex(output_path, na_rep="-", float_format="%.3f")

    output_path = os.path.join(args.output_directory, "trial_by_snapshot_success.jpg")
    plt = trial_by_snapshot_success(df)
    plt.save(output_path)

    output_path = os.path.join(
        args.output_directory, "trial_by_snapshot_irregularity_prop.jpg"
    )
    plt = trial_by_snapshot_irregularity_prop(df)
    plt.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    args = parser.parse_args()

    main(args)
