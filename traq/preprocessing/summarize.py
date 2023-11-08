import os
import pickle
from functools import reduce

import numpy as np
import pandas as pd
import toml


def summarize(config_filename, output_root):
    config = toml.load(open(config_filename, "r"))
    trial_name = config["name"]
    root_dir = os.path.join(output_root, "diffs", trial_name)
    output_directory = os.path.join(output_root, "summaries")
    os.makedirs(output_directory, exist_ok=True)
    output_filename = os.path.join(output_directory, f"{trial_name}.csv")
    table_names = load_metadata(config["root"], config["metadata"])

    print(f"{trial_name}: ")
    print_and_write_summary(root_dir, trial_name, output_filename, table_names)


def load_metadata(root_dir, metadata_config):
    metadata_filename = os.path.join(root_dir, metadata_config.pop("filename"))
    table_name_colname = metadata_config.pop("column_name")
    metadata = pd.read_excel(metadata_filename, **metadata_config)
    table_names = list(metadata[table_name_colname].unique())

    return table_names


def print_and_write_summary(root_dir, trial_name, output_filename, table_names):
    summary_dfs = []
    for diff_path in os.listdir(root_dir):
        total_diff_path = os.path.join(root_dir, diff_path, "total_diff.pkl")
        total_diff = pickle.load(open(total_diff_path, "rb"))
        summary = []
        for table_name, table_info in total_diff.items():
            if table_info["success"]:
                num_diffs = len(table_info["diff"])
                num_cells = np.prod(table_info["input_data"].shape)
                additional_idxers = set(table_info["index_names"])

                summary.append(
                    {
                        "table_name": table_name,
                        "num_diffs": num_cells,
                        "diff_frac": num_diffs / num_cells,
                        "additional_idxers": additional_idxers,
                        "success": True,
                    }
                )
                # import pdb; pdb.set_trace()
            else:
                summary.append(
                    {
                        "table_name": table_name,
                        "success": False,
                        "error": table_info["error"],
                    }
                )
        summary_df = pd.DataFrame(summary)
        summary_df["diff_name"] = diff_path
        summary_dfs.append(summary_df)
    summary_df_long = pd.concat(summary_dfs, axis=0)

    # summary_df = summary_df_long.pivot(index="table_name", columns="diff_name")
    # summary_df.columns = summary_df.columns.swaplevel(0, 1)
    # summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)

    additional_idxers = summary_df_long.groupby("table_name")["additional_idxers"].agg(
        lambda x: reduce(set.union, (y for y in x if isinstance(y, set)), set())
    )

    if "diff_frac" not in summary_df_long.columns:
        summary_df_long["diff_frac"] = np.nan

    summary_df = summary_df_long.pivot(
        index="table_name", columns="diff_name", values="diff_frac"
    )
    summary_df = summary_df.fillna(
        summary_df_long.pivot(
            index="table_name", columns="diff_name", values="error"
        ).loc[summary_df.index]
    ).fillna("TableNotPresent")
    summary_df["num_successes"] = summary_df_long.groupby("table_name")["success"].sum()
    summary_df = summary_df.join(additional_idxers)

    num_successful = summary_df["num_successes"].sum()
    num_partially_successful_tables = (summary_df["num_successes"] > 0).sum()
    summary_df = summary_df.sort_values(by="num_successes", ascending=False).drop(
        "num_successes", axis=1
    )
    total_tables = len(summary_df)
    num_snapshots = summary_df.shape[1]
    total_attempts = total_tables * num_snapshots
    pct_success = num_successful / total_attempts * 100
    pct_partial_success = num_partially_successful_tables / total_tables * 100
    additional_idxers_all = reduce(set.union, summary_df["additional_idxers"])

    summary_df.to_csv(output_filename)

    # Look at tables of concern.
    levelwise_summary = tables_of_concern(summary_df_long)

    print(
        "\tNumber of tables successfully processed at least once: "
        f"{num_partially_successful_tables} / {total_tables} "
        f"({pct_partial_success:.2f}%)"
    )
    print(
        f"\tNumber of table snapshots successfully processed: {num_successful} / "
        f"{total_attempts} ({pct_success:.2f}%)"
    )
    print(f"\tAdditional indexers: {additional_idxers_all}")
    print("Levelwise summary:")
    print(levelwise_summary)


def tables_of_concern(summary_df):
    summary_df_long = summary_df.explode("additional_idxers")
    summary_df_long["uses_nonstandard_index"] = summary_df_long[
        "success"
    ] & ~summary_df_long["additional_idxers"].isin(["centre", "id"])
    summary_df_long["uses_highly_nonstandard_index"] = (
        summary_df_long["success"]
        & ~summary_df_long["additional_idxers"].isin(["centre", "id"])
        & ~summary_df_long["additional_idxers"].str.match(r".+r(pt)?n$").fillna(False)
    )
    success = (
        summary_df_long[["diff_name", "table_name", "success"]]
        .drop_duplicates()
        .set_index(["diff_name", "table_name"])
    )
    uni = summary_df_long.groupby(["diff_name", "table_name"])[
        "uses_nonstandard_index"
    ].max()
    uhni = summary_df_long.groupby(["diff_name", "table_name"])[
        "uses_highly_nonstandard_index"
    ].max()

    per_snapshot_summary = pd.concat((success, uni, uhni), axis=1)
    print(per_snapshot_summary)

    success_rate = summary_df_long.groupby("table_name")["success"].mean()
    level_0_tables = set(success_rate[success_rate == 1.0].index)
    summary_df_long = summary_df_long[
        ~summary_df_long["table_name"].isin(level_0_tables)
    ]
