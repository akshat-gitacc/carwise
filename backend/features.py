import pandas as pd
from backend.config import NUMERIC, CATEGORICAL, DERIVED

def build_matrix(df, columns=None):
    X = df[NUMERIC + DERIVED + CATEGORICAL].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X