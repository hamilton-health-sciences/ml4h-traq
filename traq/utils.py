import numpy as np
from sklearn.impute import SimpleImputer

MIN_SAMPLES = 10
MIN_COLUMNS = 5
MAX_MISSINGNESS_FRAC = 0.999


def preprocess_plate(plate):
    df = plate.to_frame()
    if len(df) < MIN_SAMPLES:
        return

    df = df.select_dtypes("number")
    df = df.loc[:, df.isnull().mean(axis=0) < MAX_MISSINGNESS_FRAC].copy()
    X = np.array(df)
    y = plate.labels

    if X.shape[1] < MIN_COLUMNS:
        return

    imputer = SimpleImputer(strategy="most_frequent")
    X_transformed = imputer.fit_transform(X)

    return X_transformed, y
