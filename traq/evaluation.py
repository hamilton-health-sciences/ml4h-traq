import multiprocessing as mp
from functools import partial
from math import sqrt

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from traq.utils import preprocess_plate


def _build_hyperparameters(grid):
    # TODO implement - ok for now if just using default hyps
    # TODO when implementing - use OrderedD]ct
    return [{}]


def roc_auc_ci(y_true, y_score, alpha=0.05, positive=1):
    """
    Normal approximation to get a confidence interval.

    From: https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
    """
    z = sp.stats.norm.ppf(1 - alpha / 2)

    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC**2) + (N2 - 1) * (Q2 - AUC**2))
        / (N1 * N2)
    )
    lower = AUC - z * SE_AUC
    upper = AUC + z * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1

    return (lower, upper)


def _evaluation_metrics(yhat, y, correction_factor=1):
    if len(np.unique(y)) < 2:
        return {"error": "no meaningful labels"}
    if (~np.isinf(yhat)).sum() == 0:
        return {"error": "all infinity predictions"}
    yhat[np.isinf(yhat) & (yhat > 0)] = np.max(yhat[~np.isinf(yhat)])
    yhat[np.isinf(yhat) & (yhat < 0)] = np.min(yhat[~np.isinf(yhat)])
    if np.isnan(yhat).sum() > 0:
        if np.isnan(yhat).sum() == len(yhat):
            return {"error": "missing predictions"}
        yhat[np.isnan(yhat)] = np.nanmean(yhat)

    auroc_lower, auroc_upper = roc_auc_ci(y, yhat)
    auroc_lower_corrected, auroc_upper_corrected = roc_auc_ci(
        y, yhat, alpha=0.05 / correction_factor
    )

    return {
        "anomaly_proportion": y.mean(),
        "auroc": roc_auc_score(y, yhat),
        "auroc.lower": auroc_lower,
        "auroc.upper": auroc_upper,
        "auroc.lower.corrected": auroc_lower_corrected,
        "auroc.upper.corrected": auroc_upper_corrected,
        "aupr": average_precision_score(y, yhat),
    }


def evaluate_pyod_models_plate(plate, cash, trial_name, snapshot_name):
    plate_results = []

    preprocessed = preprocess_plate(plate)
    if not preprocessed:
        return
    X, y = preprocessed

    num_algorithms = 0
    for _, hyperparameter_grid in cash.items():
        num_algorithms += len(_build_hyperparameters(hyperparameter_grid))

    for algorithm, hyperparameter_grid in cash.items():
        hyperparameter_combinations = _build_hyperparameters(hyperparameter_grid)
        for hyperparameter_combination in hyperparameter_combinations:
            model_instance = algorithm(**hyperparameter_grid)
            model_instance.fit(X)
            predictions = model_instance.decision_scores_
            metrics = _evaluation_metrics(
                predictions, y, correction_factor=num_algorithms
            )

            plate_results.append(
                {
                    "plate": plate.name,
                    "num_samples": X.shape[0],
                    "num_columns": X.shape[1],
                    "num_anomalies": y.sum(),
                    "anomaly_proportion": y.mean(),
                    "algorithm": algorithm.__name__,
                    "hyperparameters": hyperparameter_combination,
                    **metrics,
                }
            )

    return plate_results


def evaluate_pyod_models(dataset, cash, trial_name, num_workers=48):
    """
    Params:
        dataset: The dataset to evaluate PyOD models.
        cash: The CASH space specifying the models.
    """
    results = []
    with mp.Pool(num_workers) as pool:
        f = partial(
            evaluate_pyod_models_plate,
            cash=cash,
            trial_name=trial_name,
            snapshot_name=dataset._name,
        )
        plates = dataset.plates()
        for plate_results in tqdm(pool.imap_unordered(f, plates), total=len(plates)):
            if plate_results is not None:
                results += plate_results
    results_df = pd.DataFrame(results)

    return results_df
