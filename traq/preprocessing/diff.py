"""Compute anomaly labels for a clinical trial."""

import glob
import os
import pickle
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from warnings import warn

import pandas as pd
import toml
from py4j.protocol import Py4JJavaError
from pyspark.errors.exceptions.captured import (
    AnalysisException,
    IllegalArgumentException,
)
from tqdm import tqdm

from . import config
from .exceptions import (
    DataFrameLoadingException,
    DataFrameTooLarge,
    DiffIdentifiersInsufficient,
    DuplicatesUncountable,
    EmptyTableSnapshot,
    ExcludedTable,
    GoldStandardSnapshotTooNarrow,
    GoldStandardSnapshotTooWide,
    GoldStandardTableNotFound,
    InputFileNotFound,
    NoMeaningfulData,
    ParticipantIdentifierNotAvailable,
    PreliminarySnapshotTooNarrow,
    PreliminarySnapshotTooWide,
    SnapshotSchemaChange,
    SufficientIdentifiersNotAvailable,
    TRAQPreprocessingException,
)
from .spark import get_spark
from .utils import DummyExecutor

spark = None


def load_sas_spark(filename):
    """
    Load a SAS file.

    Params:
        filename: The SAS filename to load.

    Returns:
        df: a Spark dataframe.
    """
    global spark
    if spark is None:
        spark = get_spark()

    df = spark.read.format("com.github.saurfang.sas.spark").load(
        filename, forceLowercaseNames=True, inferLong=True
    )
    return df


def load_idatafax_frame(filename, force_keep_columns=None):
    """
    Load an iDataFax dataframe.

    Params:
        filename: The SAS file to load.
        force_keep_columns: The metadata columns not to drop.

    Returns:
        df: The loaded dataframe.
    """
    if filename is None:
        raise DataFrameLoadingException

    if force_keep_columns is None:
        force_keep_columns = []
    try:
        df = load_sas_spark(filename)
    except (ValueError, Py4JJavaError):
        raise DataFrameLoadingException
    cols_prune = [
        col
        for col in df.schema.names
        if col.startswith("df") and col not in force_keep_columns
    ]
    df = df.drop(*cols_prune)

    # Type columns.
    for colname, coltype in config.TYPED_COLS.items():
        if colname in df.schema.names:
            df = df.withColumn(colname, df[colname].cast(coltype))

    return df


def _dup_count(df, colset):
    """
    Count the max number of duplicates when using a columns `colset` as an index.

    Params:
        df: The dataframe.
        colset: The set of columns to use as an index.

    Returns:
        max: The maximum number of duplicates for a given record when using `colset` to
             index records.
    """
    try:
        dup_count = df.groupby(colset).count().agg({"count": "max"}).collect()[0][0]
    except AnalysisException:
        dup_count = None

    if not dup_count:
        raise DuplicatesUncountable

    return dup_count


def _greedily_identify_index_columns(pre_df, post_df, current_columns):
    """
    Identify a valid set of columns to uniquely identify records.

    Params:
        pre_df: The preliminary snapshot dataframe.
        post_df: The "gold standard" (final) snapshot.
        current_columns: The set of columns to check and/or build on.

    Returns:
        suitable_columns: A set of columns that is suitable for indexing records.

    Raises:
        SufficientIdentifiersNotAvailable: There is no set of identifying columns in the
                                           dataframe that meet the criteria.
    """
    if len(current_columns) > config.MAX_NUM_ID_COLS:
        raise SufficientIdentifiersNotAvailable

    # Check if the current set of columns is sufficient to uniquely identify a
    # participant.
    pre_dup_count = _dup_count(pre_df, current_columns)
    pre_dup = pre_dup_count > 1
    post_dup_count = _dup_count(post_df, current_columns)
    post_dup = post_dup_count > 1
    if not (pre_dup or post_dup):
        return current_columns

    # Identify potential columns to add to the set of index columns.
    joint_columns = set(pre_df.schema.names) & set(post_df.schema.names) - set(
        current_columns
    )
    if not joint_columns:
        raise SufficientIdentifiersNotAvailable

    # Identify the best column to add to the set of index columns.
    potential_dup_counts = {}
    for colname in joint_columns:
        potential_colset = list(set(current_columns) | set([colname]))
        potential_dup_counts[colname] = _dup_count(post_df, potential_colset)
    best_column = min(potential_dup_counts, key=potential_dup_counts.get)
    best_column_dup_count = potential_dup_counts[best_column]
    best_columns = [
        colname
        for colname in potential_dup_counts
        if potential_dup_counts[colname] == best_column_dup_count
    ]
    best_columns_filt = [
        colname
        for colname in best_columns
        if any(re.match(r, colname) for r in config.POTENTIAL_ID_COL_REGEXS)
    ]
    if best_columns_filt:
        best_columns = best_columns_filt
        best_column = best_columns[0]
    if len(best_columns) > 1:
        warn(
            "There is more than one good option for more uniquely identifying rows: "
            f"{best_columns}, selecting {best_column}"
        )

    return _greedily_identify_index_columns(
        pre_df, post_df, list(set(current_columns) | set([best_column]))
    )


def select_index_columns(pre_df, post_df, additional_identifiers):
    id_cols = config.DEFAULT_ID_COLS.copy() + additional_identifiers

    # Confirm that the pre snapshot is non-empty.
    if pre_df.count() == 0:
        raise EmptyTableSnapshot

    # Confirm that the absolutely required columns are available.
    if not all(col in pre_df.schema.names for col in id_cols):
        raise ParticipantIdentifierNotAvailable

    # Sometimes, we can identify a subset of columns which uniquely identify a patient
    # using a custom, greedy search. Try it first.
    id_cols = _greedily_identify_index_columns(pre_df, post_df, current_columns=id_cols)

    return id_cols


def _ensure_sizes(pre_df, post_df):
    if len(pre_df.columns) < config.MIN_COLUMNS:
        raise PreliminarySnapshotTooNarrow
    if len(pre_df.columns) > config.MAX_COLUMNS:
        raise PreliminarySnapshotTooWide
    if len(post_df.columns) < config.MIN_COLUMNS:
        raise GoldStandardSnapshotTooNarrow
    if len(post_df.columns) > config.MAX_COLUMNS:
        raise GoldStandardSnapshotTooWide


def _align_schemas(pre_df, post_df):
    # Check for column discrepancies (columns being added, removed, or renamed).
    if len(post_df.schema.names) != len(pre_df.schema.names):
        added = set(post_df.schema.names) - set(pre_df.schema.names)
        post_df = post_df.drop(*added)
        removed = set(pre_df.schema.names) - set(post_df.schema.names)
        pre_df = pre_df.drop(*removed)

    return pre_df, post_df


def _compute_changes(pre_df, post_df, id_cols):
    try:
        diff = pre_df.diff(post_df, *id_cols)
    except IllegalArgumentException:
        raise SnapshotSchemaChange

    # Filter to changes and deletions and convert to Pandas
    try:
        changes = diff.filter((diff["diff"] == "C") | (diff["diff"] == "D")).toPandas()
    except Py4JJavaError as exc:
        raise DataFrameTooLarge(original_exception=exc)
    changes = changes.set_index(["diff"] + id_cols)

    return changes


def compute_table_diff(
    preliminary_table_path, final_table_path, additional_identifiers
):
    """
    Compute the diff between a preliminary table snapshot and the gold standard table.

    Params:
        preliminary_table_path: The path to the preliminary snapshot SAS file.
        final_table_path: The path to the gold-standard SAS file.
        additional_identifiers: Additional columns to use as an initial set of
                                identifiers for the table.

    Returns:
        pre_df_pd: The Pandas dataframe representing the preliminary snapshot data,
                   indexed by the discovered identifiers.
        changes_salient: The salient changes between the preliminary and gold-standard
                         snapshots.
    """
    # Load the pre and post snapshots.
    pre_df = load_idatafax_frame(
        preliminary_table_path, force_keep_columns=config.FORCE_KEEP_COLS
    )
    try:
        post_df = load_idatafax_frame(
            final_table_path, force_keep_columns=config.FORCE_KEEP_COLS
        )
    except DataFrameLoadingException:
        raise GoldStandardTableNotFound

    # Ensure not too large for the pipeline.
    _ensure_sizes(pre_df, post_df)

    # Construct a list of columns that uniquely identifies a row.
    id_cols = select_index_columns(pre_df, post_df, additional_identifiers)

    pre_df, post_df = _align_schemas(pre_df, post_df)

    # Compute the diff and then the relevant changeset.
    changes = _compute_changes(pre_df, post_df, id_cols)

    # Pivot to useful format.
    if len(changes.columns) == 0 or len(changes) == 0:
        raise NoMeaningfulData

    changes.columns = pd.MultiIndex.from_tuples(
        list(tuple(col.split("_")) for col in changes.columns)
    )
    changes_long = changes.reset_index().melt(id_vars=(["diff"] + id_cols))
    try:
        changes_pivot = changes_long.pivot(
            index=(["diff"] + id_cols + ["variable_1"]),
            columns="variable_0",
            values="value",
        )
    except ValueError:
        raise DiffIdentifiersInsufficient

    # Filter to salient changes.
    changes_salient = changes_pivot[
        (changes_pivot["left"] != changes_pivot["right"])
        & ~(changes_pivot["left"].isnull() & changes_pivot["right"].isnull())
    ].copy()
    changes_salient.index = changes_salient.index.rename({"variable_1": "column"})
    changes_salient = changes_salient.reset_index()

    # Produce versions of the raw data suitable for analysis (e.g. to serve as X for
    # the ML algorithms).
    pre_df_pd = pre_df.toPandas()

    return pre_df_pd.set_index(id_cols), changes_salient


def _generate_failure_data(preliminary_root, final_root, table, exc):
    """Format the output of a table diff failure."""
    preliminary_table_path = os.path.join(preliminary_root, table)
    final_table_path = os.path.join(final_root, table)

    return {
        "success": False,
        "error": exc.__class__.__name__,
        "error_message": exc.message,
        "preliminary_table_path": preliminary_table_path,
        "final_table_path": final_table_path,
    }


def process_table(preliminary_root, final_root, table, exclude, additional_identifiers):
    """
    Process a preliminary table snapshot (paired with its gold-standard snapshot).

    Params:
        preliminary_root: The preliminary snapshot root.
        final_root: The gold-standard snapshot root.
        table: The table filename.
        exclude: A list of table names to exclude.
        additional_identifiers: Columns to use as additional record identifiers for
                                each table.

    Returns:
        table_name: The name of the table.
        table_diff_result: The information from the resulting diff in the event of a
                           success, or failure information if it failed.
    """
    # Input and output filenames.
    table_name = table.split(".")[0]
    if table_name in exclude:
        return table_name, _generate_failure_data(
            preliminary_root, final_root, table, ExcludedTable()
        )

    # Try to identify the final table path. Might not exist.
    preliminary_table_path = os.path.join(preliminary_root, table)
    final_table_path = None
    possible_filenames = [f"{table_name}.sas7bdat", f"{table_name}.sas7bdat.gz"]
    for possible_filename in possible_filenames:
        possible_path = os.path.join(final_root, possible_filename)
        if os.path.exists(possible_path):
            final_table_path = possible_path
            break
    try:
        pre_df_pd, changes_salient = compute_table_diff(
            preliminary_table_path,
            final_table_path,
            additional_identifiers[table_name]
            if table_name in additional_identifiers
            else [],
        )

        # Output.
        table_diff_result = {
            "success": True,
            "input_data": pre_df_pd,
            "index_names": list(pre_df_pd.index.names),
            "diff": changes_salient,
        }
    except TRAQPreprocessingException as exc:
        table_diff_result = _generate_failure_data(
            preliminary_root, final_root, table, exc
        )

    return table_name, table_diff_result


def process_tables_in_executor(
    preliminary_root, final_root, tables, exclude, additional_identifiers, executor
):
    """
    For each table, submit a background task to compute the diff.

    Params:
        preliminary_root: The root path of all pre- table snapshots.
        final_root: The root path of all post- table snapshots.
        tables: The list of table filenames to process.
        exclude: A list of table names to exclude.
        additional_identifiers: The trial-level configurable list of additional
                                identifiers to use for each table.
        executor: An `Executor` instance that is used to process tables.

    Returns:
        total_diff: The diff between each pair of tables.
    """
    total_diff = {}

    with executor:
        future_to_table = {
            executor.submit(
                process_table,
                preliminary_root,
                final_root,
                table,
                exclude,
                additional_identifiers,
            ): table
            for table in tables
        }
        try:
            for future in tqdm(
                as_completed(future_to_table), total=len(future_to_table)
            ):
                table = future_to_table[future]
                try:
                    table_name, table_diff = future.result()
                except Exception:
                    warn(f"Table {table_name} raised exception:")
                    warn(traceback.format_exc())
                else:
                    total_diff[table_name] = table_diff
                finally:
                    del future_to_table[future]
        # TODO this clause is not reached in current version of the code, but could be
        # useful so leaving it for illustration purposes
        except TimeoutError as exc:
            for _, table_name in future_to_table:
                total_diff[table_name] = _generate_failure_data(
                    preliminary_root, final_root, table, exc
                )

    return total_diff


def compute_diff(
    study_root,
    preliminary,
    final,
    overrides,
    exclude,
    additional_identifiers,
    num_workers,
):
    """
    Compute the total diff between a preliminary snapshot and a gold-standard snapshot.

    Params:
        study_root: The root of all snapshots, including the preliminary snapshot and
                    the gold-standard snapshot.
        preliminary: The name of the preliminary snapshot to process.
        final: The name of the gold-standard snapshot to refer to.
        overrides: Global config overrides specific to this trial.
        exclude: A list of table names to exclude.
        additional_identifiers: Additional indexers to use for each table in the study.
        num_workers: The number of workers to use for `Executor`s.

    Returns:
        total_diff: The diff between each preliminary table and its corresponding gold-
                    standard table.
    """

    preliminary_root = os.path.join(study_root, preliminary)
    final_root = os.path.join(study_root, final)

    # Check that inputs exist.
    if not os.path.exists(preliminary_root):
        raise InputFileNotFound(f"Preliminary snapshot {preliminary} not found")
    if not os.path.exists(final_root):
        raise InputFileNotFound(f"Gold-standard snapshot {final} not found")

    # Set config overrides.
    if "max_num_id_cols" in overrides:
        config.MAX_NUM_ID_COLS = overrides["max_num_id_cols"]
    if "default_id_cols" in overrides:
        config.DEFAULT_ID_COLS = overrides["default_id_cols"]

    # The bottleneck is assigning jobs to the Spark cluster. To avoid this, we can run
    # the jobs which do the assigning in background threads. This is essentially IO-
    # bound code (talking to the Spark cluster which is running in Java in the
    # background).
    tables = glob.glob("*.sas7bdat*", root_dir=preliminary_root)
    if num_workers < 0:
        executor = DummyExecutor()
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
    total_diff = process_tables_in_executor(
        preliminary_root, final_root, tables, exclude, additional_identifiers, executor
    )

    return total_diff


def write_diff(diff, output_path):
    """
    Write out the study diff.

    Params:
        diff: The total study diff.
        output_path: The root directory for this snapshot diff, will be created if it
                     doesn't exist.
    """
    # Make output directory.
    os.makedirs(output_path, exist_ok=True)

    # Write total diff to a pickle file for completeness.
    total_diff_filename = os.path.join(output_path, "total_diff.pkl")
    pickle.dump(diff, open(total_diff_filename, "wb"))

    # Write each individual table as feather files for fast access.
    for table_name in diff:
        if not diff[table_name]["success"]:
            continue

        table_root_path = os.path.join(output_path, table_name)
        os.makedirs(table_root_path, exist_ok=True)

        data_path = os.path.join(table_root_path, "data.feather")
        diff_path = os.path.join(table_root_path, "diff.pkl")
        input_data = diff[table_name]["input_data"]
        if isinstance(input_data.index, (pd.Index, pd.MultiIndex)):
            input_data = input_data.reset_index()
        input_data.to_feather(data_path)
        pickle.dump(diff[table_name]["diff"], open(diff_path, "wb"))


def compute_and_write_diff(config, snapshot_name, num_workers, output_root):
    """
    Compute the study diff and write it to disk.

    Params:
        config: The configuration specified for a particular study.
        snapshot_name: The snapshot to compute the diff for.
        num_workers: The number of workers to use for `Executor`s.
        output_root: The root directory for all output related to a particular study.
    """
    exclude = (
        config["tables"]["exclude"]
        if "tables" in config and "exclude" in config["tables"]
        else []
    )
    additional_identifiers = (
        config["tables"]["additional_identifiers"]
        if "tables" in config and "additional_identifiers" in config["tables"]
        else {}
    )
    snapshots_root = os.path.join(config["root"], config["snapshots"]["root_subdir"])

    diff = compute_diff(
        snapshots_root,
        snapshot_name,
        config["snapshots"]["gold_standard"],
        config["overrides"] if "overrides" in config else {},
        exclude,
        additional_identifiers,
        num_workers,
    )
    output_path = os.path.join(output_root, config["name"], f"{snapshot_name}_diff")
    write_diff(diff, output_path)


def diff(config_filename, output_root, num_workers):
    """
    Compute the diffs for a particular study.

    For each preliminary snapshot, we:

    1. Iterate over each table in the preliminary snapshot and attempt to match it with
       a gold-standard table.
    2. Check whether ["centre", "id"] is a sufficient set of columns to uniquely
       identify rows in both the preliminary and gold-standard table.
    3. If it's not a sufficient set of columns, iterate over all columns `x` in the
       table and check whether ["centre", "id", x] is closer to sufficient. Identify the
       optimal `x`, and repeat the process until sufficient or the number of columns in
       the colset exceeds a configurable value.
    4. Compute the diff using the selected set of indexers and subset to
       non-double-nulls (salient changes).
    5. Output the diff for all tables, successful or not, in the correct format (either
       a successful diff paired with a suitable input dataframe, or an interpretable
       error message).

    Params:
        config_filename: The path to the config file (TOML format).
        output_root: The output root where the diffs directory will be placed.
        num_workers: The nubmer of workers for each `Executor`.
    """
    # Load config.
    config = toml.load(open(config_filename, "r"))
    trials_root = os.path.join(output_root, "diffs")

    # Look at each pair of adjacent snapshots.
    if num_workers < 0:
        executor = DummyExecutor()
    else:
        executor = ThreadPoolExecutor(max_workers=10)
    with executor:
        preliminary_snapshot_names = config["snapshots"]["preliminary"]
        futures = {
            executor.submit(
                compute_and_write_diff, config, snapshot_name, num_workers, trials_root
            ): snapshot_name
            for snapshot_name in preliminary_snapshot_names
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            snapshot_name = futures[future]
            try:
                future.result()
            except TRAQPreprocessingException:
                warn(f"{snapshot_name} generated exception:")
                warn(traceback.format_exc())
