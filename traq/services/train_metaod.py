import os
import pickle

import numpy as np
import pandas as pd
from metaod.models.core import MetaODClass
from sklearn.preprocessing import MinMaxScaler

from traq.grid import cash


def fix_nan(X):
    """MetaOD utility function."""
    # TODO: should store the mean of the meta features to be used for test_meta
    # replace by 0 for now
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    return X


def train_metaod(training_set):
    algorithms = [algo.__name__ for algo in cash]

    trial_idxs = []
    performances = []
    metafeatures = []
    for trial_idx, (metafeatures_dict, performance_table) in enumerate(training_set):
        performance_table = (
            performance_table.pivot(
                index=("snapshot", "plate"),
                columns="algorithm",
                values="auroc",
            )
            .dropna(how="all")
            .fillna(0.0)
            .loc[:, algorithms]
        )
        for snapshot_name in metafeatures_dict:
            for plate in metafeatures_dict[snapshot_name]:
                if (snapshot_name, plate) not in performance_table.index:
                    continue

                mf = metafeatures_dict[snapshot_name][plate][0]
                perf = np.array(performance_table.loc[(snapshot_name, plate)])
                metafeatures.append(mf)
                performances.append(perf)
                trial_idxs.append(trial_idx)
    trial_idxs = np.array(trial_idxs)
    meta_mat = np.vstack(metafeatures)
    performances = np.vstack(performances)

    nonerror_sel = np.isnan(meta_mat).sum(axis=1) < meta_mat.shape[1]
    trial_idx = trial_idxs[nonerror_sel]
    meta_mat = meta_mat[nonerror_sel, :]
    performances = performances[nonerror_sel, :]

    meta_mat[np.isinf(meta_mat)] = np.nan
    _scaler = MinMaxScaler()
    meta_mat_transformed = fix_nan(_scaler.fit_transform(meta_mat))
    meta_mat_transformed[np.isnan(meta_mat_transformed)] = 0.0
    performances[np.isnan(performances)] = 0.0  # otherwise optimistic

    train_sel = trial_idx != 0  # hold out the first trial for tuning
    val_sel = trial_idx == 0  # tuning set

    _clf = MetaODClass(
        performances[train_sel, :],
        performances[val_sel, :],
        learning="sgd",
    )
    _clf.train(
        meta_features=meta_mat_transformed[train_sel, :],
        valid_meta=meta_mat_transformed[val_sel, :],
    )

    return _scaler, _clf


def main(args):
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    training_set = []
    for performance_filename in os.listdir(args.performance_directory):
        trial_name = performance_filename.split(".")[0]
        if trial_name == args.exclude_from_training:
            print(f"Excluding {trial_name} from meta-training")
            continue
        metafeatures_filename = os.path.join(
            args.metafeatures_directory, f"{trial_name}.pkl"
        )
        performance_filename = os.path.join(
            args.performance_directory, performance_filename
        )
        metafeatures = pickle.load(open(metafeatures_filename, "rb"))
        performance = pd.read_csv(performance_filename)

        training_set.append((metafeatures, performance))

    trained = train_metaod(training_set)

    pickle.dump(trained, open(args.output_filename, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metafeatures_directory", type=str, required=True)
    parser.add_argument("--performance_directory", type=str, required=True)
    parser.add_argument("--exclude_from_training", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    args = parser.parse_args()

    main(args)
