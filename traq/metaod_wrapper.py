"""Wrapper around the public implementation of MetaOD."""

import json
import os
from itertools import product
from typing import Optional, Tuple

import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler

from traq.grid import cash
from traq.services.utils import decode_array, encode_array

METAOD_SERVICE_URL = os.environ.get("METAOD_SERVICE_URL", "http://localhost:8000")


def _build_hyperparameters(hyperparameter_grid):
    combinations = list(
        product(*(hyp_settings for _, hyp_settings in hyperparameter_grid.items()))
    )
    hyps = [
        dict(zip(hyperparameter_grid.keys(), combination))
        for combination in combinations
    ]

    return hyps


retrained_model_list = [
    (algo, config)
    for algo, hyp_grid in cash.items()
    for config in _build_hyperparameters(hyp_grid)
]


def submit_metaod_request(
    model_name: str,
    X: np.ndarray,
    dataset_identifiers: Optional[Tuple[str]] = None,
    k: int = 1,
):
    url = METAOD_SERVICE_URL + "/metaod"
    X_enc, X_shape = encode_array(X)
    payload = {
        "model_name": model_name,
        "input_array_encoded": X_enc,
        "shape": X_shape,
        "trial_name": dataset_identifiers[0],
        "snapshot_name": dataset_identifiers[1],
        "plate_name": dataset_identifiers[2],
        "k": k,
    }
    r = requests.post(url, data=json.dumps(payload))
    assert r.status_code == 200, r.json()
    response = r.json()
    est_class, est_params = response["est_class"], response["est_params"]
    decision_function = decode_array(*response["decision_function"])

    return est_class, est_params, decision_function


class MetaODPE:
    def __init__(self, model_name: str = "default", k=3) -> None:
        self._model_name = model_name
        self._k = k

    def fit(self, X, dataset_identifiers=None, y=None):
        self._X = X
        (
            self._est_class,
            self._est_params,
            self.decision_scores_,
        ) = submit_metaod_request(self._model_name, X, dataset_identifiers, k=self._k)

        return self

    def __str__(self):
        if self._model_name == "default":
            model_name = "pretrained"
        else:
            model_name = "retrained"

        return f"MetaODPE({model_name}, k={self._k})"


def prune_inf(x):
    if (~np.isinf(x)).sum() > 0:
        return np.clip(x, np.min(x[~np.isinf(x)]), np.max(x[~np.isinf(x)]))

    return np.zeros(len(x))


class Ensemble:
    def __init__(self, transform: Optional[str] = None) -> None:
        self._transform = transform
        if transform is None:
            self._fxn = lambda est, X: prune_inf(est.decision_scores_)
        elif transform == "probabilistic":
            self._fxn = (
                lambda est, X: MinMaxScaler()
                .fit_transform(prune_inf(est.decision_scores_).reshape(-1, 1))
                .reshape(-1)
            )

    def fit(self, X, dataset_identifiers=None, y=None):
        self._X = X
        self._estimators = []
        for algo, config in retrained_model_list:
            est = algo(**config)
            est.fit(self._X)
            self._estimators.append(est)

        scores = []
        for est in self._estimators:
            scores.append(self._fxn(est, X))
        self.decision_scores_ = np.nanmean(np.vstack(scores), axis=0)

        return self

    def __str__(self):
        if self._transform is None:
            return "NaiveEnsemble"

        if self._transform == "probabilistic":
            return "ProbabilisticEnsemble"


class MetaODWrapper:
    def __init__(self, model_name: str = "default") -> None:
        self._model_name = model_name

    def fit(self, X, dataset_identifiers=None, y=None):
        self._X = X
        (
            self._est_class,
            self._est_params,
            self.decision_scores_,
        ) = submit_metaod_request(self._model_name, X, dataset_identifiers)

        return self

    def __str__(self):
        if self._model_name == "default":
            model_name = "pretrained"
        else:
            model_name = "retrained"

        return f"MetaOD({model_name})"
