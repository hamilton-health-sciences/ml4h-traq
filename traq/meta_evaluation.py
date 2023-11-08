import multiprocessing as mp
from functools import partial
from itertools import product

import pandas as pd
from tqdm import tqdm

from traq.evaluation import _evaluation_metrics
from traq.utils import preprocess_plate


def _build_hyperparameters(hyperparameter_grid):
    combinations = list(
        product(*(hyp_settings for _, hyp_settings in hyperparameter_grid.items()))
    )
    hyps = [
        dict(zip(hyperparameter_grid.keys(), combination))
        for combination in combinations
    ]

    return hyps


def evaluate_model_selection_plate(plate, cash, trial_name, snapshot_name):
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
            model_instance = algorithm(**hyperparameter_combination)
            plate_name = plate.name
            model_instance.fit(X, (trial_name, snapshot_name, plate_name))
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
                    "algorithm": str(model_instance),
                    "hyperparameters": hyperparameter_combination,
                    **metrics,
                }
            )

    return plate_results


def evaluate_model_selection(dataset, cash, trial_name, num_workers=48):
    """
    Params:
        dataset: The dataset to evaluate PyOD models.
        cash: The CASH space specifying the models.
    """
    results = []
    with mp.Pool(num_workers) as pool:
        f = partial(
            evaluate_model_selection_plate,
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
