import os
import pickle
from itertools import product
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from joblib import load
from metaod.models.gen_meta_features import generate_meta_features
from metaod.models.predict_metaod import get_top_models
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

from traq.grid import cash

from .utils import PRETRAINED_BASE_ESTIMATORS, decode_array, encode_array


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


class MetaODInput(BaseModel):
    model_name: Optional[str] = "default"

    trial_name: Optional[str] = None
    snapshot_name: Optional[str] = None
    plate_name: Optional[str] = None

    input_array_encoded: str
    shape: List[int]

    # For ensembling.
    k: Optional[int] = 1


app = FastAPI()

trained_model_location = "models/metaod_pretrained"
retrained_models_location = "models/metaod"

meta_scalar = load(os.path.join(trained_model_location, "meta_scalar.joblib"))
trained_models = [
    "train_0.joblib",
    "train_2.joblib",
    # "train_42.joblib"
]
# # load trained models
model_lists = list(load(os.path.join(trained_model_location, "model_list.joblib")))
clfs = [load(os.path.join(trained_model_location, model)) for model in trained_models]


def load_cached_metafeatures(trial, snapshot, plate):
    pkl_path = os.path.join("data", "metafeatures", f"{trial}.pkl")
    if not os.path.exists(pkl_path):
        return
    data = pickle.load(open(pkl_path, "rb"))
    plate_metafeatures = data[snapshot][plate]

    return plate_metafeatures


def select_model(X, metafeatures=None, k=1):
    """For speedups."""
    # print(os.path.realpath(__file__))
    # unzip trained models
    # with ZipFile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                           'trained_models.zip'), 'r') as zip:
    #     # # printing all the contents of the zip file
    #     # zip.printdir()

    #     # extracting all the files
    #     print('Extracting trained models now...')
    #     zip.extractall(path='trained_models')
    #     print('Finish extracting models')

    # load PCA scalar
    # generate meta features
    if metafeatures is None:
        meta_X, _ = generate_meta_features(X)
    else:
        meta_X, _ = metafeatures
    meta_X = np.nan_to_num(meta_X, nan=0)
    # replace nan by 0 for now
    # todo: replace by mean is better as fix_nan
    meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1)).astype(float)
    meta_X = np.clip(meta_X, 0, 1)  # original values lie in [0, 1]

    # use all trained models for ensemble
    predict_scores = np.zeros([len(trained_models), len(model_lists)])

    for i, model in enumerate(trained_models):
        clf = clfs[i]
        # w = load (model)
        predict_scores[i,] = clf.predict(meta_X)
        # predicted_scores_max = np.nanargmax(predict_scores[i,])
        # print('top model', model_lists[predicted_scores_max])
    combined_predict = np.average(predict_scores, axis=0)

    predicted_scores_sorted = get_top_models(combined_predict, k)
    # predicted_scores_max = np.nanargmax(combined_predict)

    model_name = np.asarray(model_lists)[predicted_scores_sorted]
    if k == 1:
        model = model_name_to_model[model_name[0]]
        est_class, est_params = type(model), model.get_params()

        return est_class, est_params
    else:
        models = [model_name_to_model[name] for name in model_name]
        est_specs = [(type(model), model.get_params()) for model in models]

        return _Ensemble, {"est_specs": est_specs}


def infer_retrained_model(model_name, X, metafeatures=None, k=1):
    """For speedups."""
    scalar, clf = pickle.load(
        open(os.path.join(retrained_models_location, model_name), "rb")
    )

    # generate meta features
    if metafeatures is None:
        meta_X, _ = generate_meta_features(X)
    else:
        meta_X, _ = metafeatures
    meta_X = np.nan_to_num(meta_X, nan=0)
    # replace nan by 0 for now
    # todo: replace by mean is better as fix_nan
    meta_X = scalar.transform(np.asarray(meta_X).reshape(1, -1)).astype(float)
    meta_X = np.nan_to_num(meta_X, nan=0)  # always nan metafeatures bring nans back
    meta_X = np.clip(meta_X, 0, 1)  # original values lie in [0, 1]

    predict_scores = clf.predict(meta_X)

    if k == 1:
        return retrained_model_list[np.nanargmax(predict_scores)]
    else:
        idxs = get_top_models(predict_scores[0], k)
        est_specs = [retrained_model_list[idx] for idx in idxs]

        return _Ensemble, {"est_specs": est_specs}


def prune_inf(x):
    if (~np.isinf(x) & ~np.isnan(x)).sum() > 0:
        m = np.min(x[~np.isinf(x) & ~np.isnan(x)])
        M = np.max(x[~np.isinf(x) & ~np.isnan(x)])
        return np.clip(x, m, M)

    return np.zeros(len(x))


class _Ensemble:
    def __init__(self, est_specs):
        self._est_specs = est_specs

    def fit(self, X):
        self._estimators = [algo(**params).fit(X) for algo, params in self._est_specs]

        # probabilistic
        scores = [
            MinMaxScaler()
            .fit_transform(prune_inf(est.decision_scores_).reshape(-1, 1))
            .reshape(-1)
            for est in self._estimators
        ]
        self.decision_scores_ = np.nanmean(np.vstack(scores), axis=0)

        return self

    def __str__(self):
        return "ProbabilisticEnsemble"


@app.post("/metaod")
def metaod(metaod_input: MetaODInput):
    metafeatures = None
    if metaod_input.trial_name:
        try:
            metafeatures = load_cached_metafeatures(
                metaod_input.trial_name,
                metaod_input.snapshot_name,
                metaod_input.plate_name,
            )
        except Exception:
            raise

    input_array = decode_array(metaod_input.input_array_encoded, metaod_input.shape)

    if metaod_input.model_name == "default":
        est_class, est_params = select_model(input_array, metafeatures, metaod_input.k)
    else:
        est_class, est_params = infer_retrained_model(
            metaod_input.model_name, input_array, metafeatures, metaod_input.k
        )

    estimator = est_class(**est_params)
    estimator.fit(input_array)
    decision_function = estimator.decision_scores_

    return {
        "est_class": str(est_class),
        "est_params": {} if isinstance(estimator, _Ensemble) else est_params,
        "decision_function": encode_array(decision_function),
    }


model_names = joblib.load(
    open(os.path.join(trained_model_location, "model_list.joblib"), "rb")
)
model_name_to_model = dict(zip(model_names, PRETRAINED_BASE_ESTIMATORS))
