import os
import pickle

import numpy as np
import pandas as pd
import toml

from . import config
from .exceptions import MetadataUnavailable


def derive_labels(config_filename, output_directory):
    config = toml.load(open(config_filename, "r"))
    trial_name = config["name"]
    diffs_root = os.path.join(output_directory, "diffs", config["name"])
    snapshot_names = config["snapshots"]["preliminary"]
    output_root = os.path.join(output_directory, "derived")
    trial_output_path = os.path.join(output_root, trial_name)
    os.makedirs(trial_output_path, exist_ok=True)
    metadata_config = config.get("metadata", {})

    for snapshot_name in snapshot_names:
        diff_name = f"{snapshot_name}_diff"
        diff_path = os.path.join(diffs_root, diff_name)
        derived = derive_diff_labels(
            snapshot_name,
            diff_path,
            config.get("label_derivation", {}),
            metadata_config,
            config["root"],
        )
        snapshot_output_path = os.path.join(trial_output_path, f"{snapshot_name}.pkl")
        pickle.dump(derived, open(snapshot_output_path, "wb"))


def parse_out_idatafax_tables(root_path, metadata_config):
    metadata_config = metadata_config.copy()

    if "unavailable" in metadata_config and metadata_config["unavailable"]:
        raise MetadataUnavailable

    filename = metadata_config.pop("filename", "Documents/CRF-DataFax-Setup.xlsx")
    filepath = os.path.join(root_path, filename)
    column_name = metadata_config.pop("column_name", "SAS libraries")
    if "header" not in metadata_config:
        metadata_config["header"] = 1

    metadata = pd.read_excel(filepath, **metadata_config)
    valid_tables = list(metadata[column_name].unique())

    return valid_tables


def _subset_diff(table_diff):
    is_change = table_diff["diff"] == "C"
    is_fill_in = table_diff["left"].isin([np.nan, None])
    sel = is_change & ~is_fill_in
    table_diff = table_diff[sel].copy()

    return table_diff


def _check_excluded_columns(table_diff, df):
    column_change_counts = table_diff.groupby("column")["diff"].count()
    column_change_fracs = column_change_counts / len(df)
    if config.EXCLUDE_COLUMN_CHANGE_FRAC > 0:
        exclude_columns = list(
            column_change_fracs[
                column_change_fracs > config.EXCLUDE_COLUMN_CHANGE_FRAC
            ].index
        )
    else:
        exclude_columns = []

    return exclude_columns


def _derive_labels(df, change_counts):
    width = len(df.columns)
    labels_short = change_counts.to_frame().rename({"column": "change_count"}, axis=1)
    labels_short["change_frac"] = labels_short["change_count"] / width
    labels_short["record_change"] = (
        labels_short["change_frac"] > config.RECORD_CHANGE_FRACTION
    )
    labels_short["any_change"] = labels_short["change_count"] > 0
    numeric_labels = set(["change_frac", "change_count"])
    boolean_labels = set(["record_change", "any_change"])

    return labels_short, numeric_labels, boolean_labels


def _pair_labels(df, labels_short, numeric_labels, boolean_labels):
    df_labelled = df.join(labels_short)
    for numeric_label_col in numeric_labels:
        df_labelled[numeric_label_col] = df_labelled[numeric_label_col].fillna(0.0)
    for boolean_label_col in boolean_labels:
        df_labelled[boolean_label_col] = df_labelled[boolean_label_col].fillna(False)
    df = df_labelled.drop(
        ["change_count", "change_frac", "record_change", "any_change"], axis=1
    )
    labels = df_labelled.drop(df.columns, axis=1)

    return df, labels


def derive_diff_labels(
    snapshot_name, diff_path, label_derivation_config, metadata_config, root_path
):
    labelled_dataframes = {}

    total_diff_path = os.path.join(diff_path, "total_diff.pkl")
    total_diff = pickle.load(open(total_diff_path, "rb"))
    try:
        include_tables = parse_out_idatafax_tables(root_path, metadata_config)
    except MetadataUnavailable:
        include_tables = list(total_diff.keys())
    exclude_tables = label_derivation_config.get("exclude_tables", [])
    exclude_columns = label_derivation_config.get("exclude_fields", [])
    for table_name, table in total_diff.items():
        if not table["success"]:
            continue
        if table_name in exclude_tables:
            continue
        if table_name not in include_tables:
            continue

        # The data, and the diff.
        df = table["input_data"]
        table_diff = table["diff"]

        # Process diff.
        table_diff = _subset_diff(table_diff)

        # Produce column-level change counts and check for columns that change too much
        # to be reliable irregularity indicators.
        exclude_columns += _check_excluded_columns(table_diff, df)

        # Produce record-level change counts. For record-level changes, exclude invalid
        # columns.
        is_valid_column = ~table_diff["column"].isin(exclude_columns)
        sel = is_valid_column
        table_diff = table_diff[sel].copy()
        change_counts = table_diff.groupby(table["index_names"])["column"].count()

        # Exclude columns that might result in over-optimistic estimates of performance.
        if config.EXCLUDE_REVISION_INDICATOR_COLUMNS:
            exclude_columns = list(set(df.columns) & set(exclude_columns))
            df = df.drop(exclude_columns, axis=1)

        # Derive labels.
        labels_short, numeric_labels, boolean_labels = _derive_labels(df, change_counts)

        # Pair input dataframe with labels.
        labelled_dataframes[table_name] = _pair_labels(
            df, labels_short, numeric_labels, boolean_labels
        )

    return labelled_dataframes
