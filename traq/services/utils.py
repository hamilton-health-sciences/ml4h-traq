"""Utilities for working with difficult-to-integrate libraries."""

from base64 import b64decode, b64encode
from typing import Sequence, Tuple

import numpy as np
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM


def decode_array(input_string: str, shape: Sequence[int]) -> np.ndarray:
    b = b64decode(input_string.encode("utf-8"))
    ary = np.frombuffer(b).reshape(shape).copy()

    return ary


def encode_array(input_array: np.ndarray) -> Tuple[str, Sequence[int]]:
    b = input_array.tobytes()
    s = b64encode(b).decode("utf-8")

    return s, list(input_array.shape)


PRETRAINED_BASE_ESTIMATORS = [
    LODA(n_bins=5, n_random_cuts=10),
    LODA(n_bins=5, n_random_cuts=20),
    LODA(n_bins=5, n_random_cuts=30),
    LODA(n_bins=5, n_random_cuts=40),
    LODA(n_bins=5, n_random_cuts=50),
    LODA(n_bins=5, n_random_cuts=75),
    LODA(n_bins=5, n_random_cuts=100),
    LODA(n_bins=5, n_random_cuts=150),
    LODA(n_bins=5, n_random_cuts=200),
    LODA(n_bins=10, n_random_cuts=10),
    LODA(n_bins=10, n_random_cuts=20),
    LODA(n_bins=10, n_random_cuts=30),
    LODA(n_bins=10, n_random_cuts=40),
    LODA(n_bins=10, n_random_cuts=50),
    LODA(n_bins=10, n_random_cuts=75),
    LODA(n_bins=10, n_random_cuts=100),
    LODA(n_bins=10, n_random_cuts=150),
    LODA(n_bins=10, n_random_cuts=200),
    LODA(n_bins=15, n_random_cuts=10),
    LODA(n_bins=15, n_random_cuts=20),
    LODA(n_bins=15, n_random_cuts=30),
    LODA(n_bins=15, n_random_cuts=40),
    LODA(n_bins=15, n_random_cuts=50),
    LODA(n_bins=15, n_random_cuts=75),
    LODA(n_bins=15, n_random_cuts=100),
    LODA(n_bins=15, n_random_cuts=150),
    LODA(n_bins=15, n_random_cuts=200),
    LODA(n_bins=20, n_random_cuts=10),
    LODA(n_bins=20, n_random_cuts=20),
    LODA(n_bins=20, n_random_cuts=30),
    LODA(n_bins=20, n_random_cuts=40),
    LODA(n_bins=20, n_random_cuts=50),
    LODA(n_bins=20, n_random_cuts=75),
    LODA(n_bins=20, n_random_cuts=100),
    LODA(n_bins=20, n_random_cuts=150),
    LODA(n_bins=20, n_random_cuts=200),
    LODA(n_bins=25, n_random_cuts=10),
    LODA(n_bins=25, n_random_cuts=20),
    LODA(n_bins=25, n_random_cuts=30),
    LODA(n_bins=25, n_random_cuts=40),
    LODA(n_bins=25, n_random_cuts=50),
    LODA(n_bins=25, n_random_cuts=75),
    LODA(n_bins=25, n_random_cuts=100),
    LODA(n_bins=25, n_random_cuts=150),
    LODA(n_bins=25, n_random_cuts=200),
    LODA(n_bins=30, n_random_cuts=10),
    LODA(n_bins=30, n_random_cuts=20),
    LODA(n_bins=30, n_random_cuts=30),
    LODA(n_bins=30, n_random_cuts=40),
    LODA(n_bins=30, n_random_cuts=50),
    LODA(n_bins=30, n_random_cuts=75),
    LODA(n_bins=30, n_random_cuts=100),
    LODA(n_bins=30, n_random_cuts=150),
    LODA(n_bins=30, n_random_cuts=200),
    ABOD(n_neighbors=3),
    ABOD(n_neighbors=5),
    ABOD(n_neighbors=10),
    ABOD(n_neighbors=15),
    ABOD(n_neighbors=20),
    ABOD(n_neighbors=25),
    ABOD(n_neighbors=50),
    ABOD(n_neighbors=60),
    ABOD(n_neighbors=75),
    ABOD(n_neighbors=80),
    ABOD(n_neighbors=90),
    ABOD(n_neighbors=100),
    IForest(n_estimators=10, max_features=0.1),
    IForest(n_estimators=10, max_features=0.2),
    IForest(n_estimators=10, max_features=0.3),
    IForest(n_estimators=10, max_features=0.4),
    IForest(n_estimators=10, max_features=0.5),
    IForest(n_estimators=10, max_features=0.6),
    IForest(n_estimators=10, max_features=0.7),
    IForest(n_estimators=10, max_features=0.8),
    IForest(n_estimators=10, max_features=0.9),
    IForest(n_estimators=20, max_features=0.1),
    IForest(n_estimators=20, max_features=0.2),
    IForest(n_estimators=20, max_features=0.3),
    IForest(n_estimators=20, max_features=0.4),
    IForest(n_estimators=20, max_features=0.5),
    IForest(n_estimators=20, max_features=0.6),
    IForest(n_estimators=20, max_features=0.7),
    IForest(n_estimators=20, max_features=0.8),
    IForest(n_estimators=20, max_features=0.9),
    IForest(n_estimators=30, max_features=0.1),
    IForest(n_estimators=30, max_features=0.2),
    IForest(n_estimators=30, max_features=0.3),
    IForest(n_estimators=30, max_features=0.4),
    IForest(n_estimators=30, max_features=0.5),
    IForest(n_estimators=30, max_features=0.6),
    IForest(n_estimators=30, max_features=0.7),
    IForest(n_estimators=30, max_features=0.8),
    IForest(n_estimators=30, max_features=0.9),
    IForest(n_estimators=40, max_features=0.1),
    IForest(n_estimators=40, max_features=0.2),
    IForest(n_estimators=40, max_features=0.3),
    IForest(n_estimators=40, max_features=0.4),
    IForest(n_estimators=40, max_features=0.5),
    IForest(n_estimators=40, max_features=0.6),
    IForest(n_estimators=40, max_features=0.7),
    IForest(n_estimators=40, max_features=0.8),
    IForest(n_estimators=40, max_features=0.9),
    IForest(n_estimators=50, max_features=0.1),
    IForest(n_estimators=50, max_features=0.2),
    IForest(n_estimators=50, max_features=0.3),
    IForest(n_estimators=50, max_features=0.4),
    IForest(n_estimators=50, max_features=0.5),
    IForest(n_estimators=50, max_features=0.6),
    IForest(n_estimators=50, max_features=0.7),
    IForest(n_estimators=50, max_features=0.8),
    IForest(n_estimators=50, max_features=0.9),
    IForest(n_estimators=75, max_features=0.1),
    IForest(n_estimators=75, max_features=0.2),
    IForest(n_estimators=75, max_features=0.3),
    IForest(n_estimators=75, max_features=0.4),
    IForest(n_estimators=75, max_features=0.5),
    IForest(n_estimators=75, max_features=0.6),
    IForest(n_estimators=75, max_features=0.7),
    IForest(n_estimators=75, max_features=0.8),
    IForest(n_estimators=75, max_features=0.9),
    IForest(n_estimators=100, max_features=0.1),
    IForest(n_estimators=100, max_features=0.2),
    IForest(n_estimators=100, max_features=0.3),
    IForest(n_estimators=100, max_features=0.4),
    IForest(n_estimators=100, max_features=0.5),
    IForest(n_estimators=100, max_features=0.6),
    IForest(n_estimators=100, max_features=0.7),
    IForest(n_estimators=100, max_features=0.8),
    IForest(n_estimators=100, max_features=0.9),
    IForest(n_estimators=150, max_features=0.1),
    IForest(n_estimators=150, max_features=0.2),
    IForest(n_estimators=150, max_features=0.3),
    IForest(n_estimators=150, max_features=0.4),
    IForest(n_estimators=150, max_features=0.5),
    IForest(n_estimators=150, max_features=0.6),
    IForest(n_estimators=150, max_features=0.7),
    IForest(n_estimators=150, max_features=0.8),
    IForest(n_estimators=150, max_features=0.9),
    IForest(n_estimators=200, max_features=0.1),
    IForest(n_estimators=200, max_features=0.2),
    IForest(n_estimators=200, max_features=0.3),
    IForest(n_estimators=200, max_features=0.4),
    IForest(n_estimators=200, max_features=0.5),
    IForest(n_estimators=200, max_features=0.6),
    IForest(n_estimators=200, max_features=0.7),
    IForest(n_estimators=200, max_features=0.8),
    IForest(n_estimators=200, max_features=0.9),
    KNN(n_neighbors=1, method="largest"),
    KNN(n_neighbors=5, method="largest"),
    KNN(n_neighbors=10, method="largest"),
    KNN(n_neighbors=15, method="largest"),
    KNN(n_neighbors=20, method="largest"),
    KNN(n_neighbors=25, method="largest"),
    KNN(n_neighbors=50, method="largest"),
    KNN(n_neighbors=60, method="largest"),
    KNN(n_neighbors=70, method="largest"),
    KNN(n_neighbors=80, method="largest"),
    KNN(n_neighbors=90, method="largest"),
    KNN(n_neighbors=100, method="largest"),
    KNN(n_neighbors=1, method="mean"),
    KNN(n_neighbors=5, method="mean"),
    KNN(n_neighbors=10, method="mean"),
    KNN(n_neighbors=15, method="mean"),
    KNN(n_neighbors=20, method="mean"),
    KNN(n_neighbors=25, method="mean"),
    KNN(n_neighbors=50, method="mean"),
    KNN(n_neighbors=60, method="mean"),
    KNN(n_neighbors=70, method="mean"),
    KNN(n_neighbors=80, method="mean"),
    KNN(n_neighbors=90, method="mean"),
    KNN(n_neighbors=100, method="mean"),
    KNN(n_neighbors=1, method="median"),
    KNN(n_neighbors=5, method="median"),
    KNN(n_neighbors=10, method="median"),
    KNN(n_neighbors=15, method="median"),
    KNN(n_neighbors=20, method="median"),
    KNN(n_neighbors=25, method="median"),
    KNN(n_neighbors=50, method="median"),
    KNN(n_neighbors=60, method="median"),
    KNN(n_neighbors=70, method="median"),
    KNN(n_neighbors=80, method="median"),
    KNN(n_neighbors=90, method="median"),
    KNN(n_neighbors=100, method="median"),
    LOF(n_neighbors=1, metric="manhattan"),
    LOF(n_neighbors=5, metric="manhattan"),
    LOF(n_neighbors=10, metric="manhattan"),
    LOF(n_neighbors=15, metric="manhattan"),
    LOF(n_neighbors=20, metric="manhattan"),
    LOF(n_neighbors=25, metric="manhattan"),
    LOF(n_neighbors=50, metric="manhattan"),
    LOF(n_neighbors=60, metric="manhattan"),
    LOF(n_neighbors=70, metric="manhattan"),
    LOF(n_neighbors=80, metric="manhattan"),
    LOF(n_neighbors=90, metric="manhattan"),
    LOF(n_neighbors=100, metric="manhattan"),
    LOF(n_neighbors=1, metric="euclidean"),
    LOF(n_neighbors=5, metric="euclidean"),
    LOF(n_neighbors=10, metric="euclidean"),
    LOF(n_neighbors=15, metric="euclidean"),
    LOF(n_neighbors=20, metric="euclidean"),
    LOF(n_neighbors=25, metric="euclidean"),
    LOF(n_neighbors=50, metric="euclidean"),
    LOF(n_neighbors=60, metric="euclidean"),
    LOF(n_neighbors=70, metric="euclidean"),
    LOF(n_neighbors=80, metric="euclidean"),
    LOF(n_neighbors=90, metric="euclidean"),
    LOF(n_neighbors=100, metric="euclidean"),
    LOF(n_neighbors=1, metric="minkowski"),
    LOF(n_neighbors=5, metric="minkowski"),
    LOF(n_neighbors=10, metric="minkowski"),
    LOF(n_neighbors=15, metric="minkowski"),
    LOF(n_neighbors=20, metric="minkowski"),
    LOF(n_neighbors=25, metric="minkowski"),
    LOF(n_neighbors=50, metric="minkowski"),
    LOF(n_neighbors=60, metric="minkowski"),
    LOF(n_neighbors=70, metric="minkowski"),
    LOF(n_neighbors=80, metric="minkowski"),
    LOF(n_neighbors=90, metric="minkowski"),
    LOF(n_neighbors=100, metric="minkowski"),
    HBOS(n_bins=5, alpha=0.1),
    HBOS(n_bins=5, alpha=0.2),
    HBOS(n_bins=5, alpha=0.3),
    HBOS(n_bins=5, alpha=0.4),
    HBOS(n_bins=5, alpha=0.5),
    HBOS(n_bins=10, alpha=0.1),
    HBOS(n_bins=10, alpha=0.2),
    HBOS(n_bins=10, alpha=0.3),
    HBOS(n_bins=10, alpha=0.4),
    HBOS(n_bins=10, alpha=0.5),
    HBOS(n_bins=20, alpha=0.1),
    HBOS(n_bins=20, alpha=0.2),
    HBOS(n_bins=20, alpha=0.3),
    HBOS(n_bins=20, alpha=0.4),
    HBOS(n_bins=20, alpha=0.5),
    HBOS(n_bins=30, alpha=0.1),
    HBOS(n_bins=30, alpha=0.2),
    HBOS(n_bins=30, alpha=0.3),
    HBOS(n_bins=30, alpha=0.4),
    HBOS(n_bins=30, alpha=0.5),
    HBOS(n_bins=40, alpha=0.1),
    HBOS(n_bins=40, alpha=0.2),
    HBOS(n_bins=40, alpha=0.3),
    HBOS(n_bins=40, alpha=0.4),
    HBOS(n_bins=40, alpha=0.5),
    HBOS(n_bins=50, alpha=0.1),
    HBOS(n_bins=50, alpha=0.2),
    HBOS(n_bins=50, alpha=0.3),
    HBOS(n_bins=50, alpha=0.4),
    HBOS(n_bins=50, alpha=0.5),
    HBOS(n_bins=75, alpha=0.1),
    HBOS(n_bins=75, alpha=0.2),
    HBOS(n_bins=75, alpha=0.3),
    HBOS(n_bins=75, alpha=0.4),
    HBOS(n_bins=75, alpha=0.5),
    HBOS(n_bins=100, alpha=0.1),
    HBOS(n_bins=100, alpha=0.2),
    HBOS(n_bins=100, alpha=0.3),
    HBOS(n_bins=100, alpha=0.4),
    HBOS(n_bins=100, alpha=0.5),
    OCSVM(nu=0.1, kernel="linear"),
    OCSVM(nu=0.2, kernel="linear"),
    OCSVM(nu=0.3, kernel="linear"),
    OCSVM(nu=0.4, kernel="linear"),
    OCSVM(nu=0.5, kernel="linear"),
    OCSVM(nu=0.6, kernel="linear"),
    OCSVM(nu=0.7, kernel="linear"),
    OCSVM(nu=0.8, kernel="linear"),
    OCSVM(nu=0.9, kernel="linear"),
    OCSVM(nu=0.1, kernel="poly"),
    OCSVM(nu=0.2, kernel="poly"),
    OCSVM(nu=0.3, kernel="poly"),
    OCSVM(nu=0.4, kernel="poly"),
    OCSVM(nu=0.5, kernel="poly"),
    OCSVM(nu=0.6, kernel="poly"),
    OCSVM(nu=0.7, kernel="poly"),
    OCSVM(nu=0.8, kernel="poly"),
    OCSVM(nu=0.9, kernel="poly"),
    OCSVM(nu=0.1, kernel="rbf"),
    OCSVM(nu=0.2, kernel="rbf"),
    OCSVM(nu=0.3, kernel="rbf"),
    OCSVM(nu=0.4, kernel="rbf"),
    OCSVM(nu=0.5, kernel="rbf"),
    OCSVM(nu=0.6, kernel="rbf"),
    OCSVM(nu=0.7, kernel="rbf"),
    OCSVM(nu=0.8, kernel="rbf"),
    OCSVM(nu=0.9, kernel="rbf"),
    OCSVM(nu=0.1, kernel="sigmoid"),
    OCSVM(nu=0.2, kernel="sigmoid"),
    OCSVM(nu=0.3, kernel="sigmoid"),
    OCSVM(nu=0.4, kernel="sigmoid"),
    OCSVM(nu=0.5, kernel="sigmoid"),
    OCSVM(nu=0.6, kernel="sigmoid"),
    OCSVM(nu=0.7, kernel="sigmoid"),
    OCSVM(nu=0.8, kernel="sigmoid"),
    OCSVM(nu=0.9, kernel="sigmoid"),
    COF(n_neighbors=3),
    COF(n_neighbors=5),
    COF(n_neighbors=10),
    COF(n_neighbors=15),
    COF(n_neighbors=20),
    COF(n_neighbors=25),
    COF(n_neighbors=50),
]