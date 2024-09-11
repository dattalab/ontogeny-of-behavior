"""
This module is the cousin of the `bio_age_model.py` file, as it hosts a
more "standard" suite of models to predict age from behavior. We use these
models to compare their residuals (which we think contain information about
the non-parallel biological age axis) to the predictions of biological age
in our model.
"""

import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


MODEL_OPTIONS = {
    "linear": LinearRegression,
    "elastic": ElasticNet,
    "pls": PLSRegression,
    "svm": SVR,
    "knn": KNeighborsRegressor,
    "rf": RandomForestRegressor,
    "gb": GradientBoostingRegressor,
    "ada": AdaBoostRegressor,
    "mlp": MLPRegressor,
}


def generate_pipeline(model, use_poly=False, use_log=False):
    opts = []
    if use_log:
        opts.append(("log_func", FunctionTransformer(func=np.log)))
    if use_poly:
        opts.append(("poly", PolynomialFeatures(degree=2)))
        opts.append(('pca', PCA(n_components=30)))
    opts.append(("scaler", StandardScaler()))
    opts.append(("model", model))
    return Pipeline(opts)


def fit_model(pipeline, X, y, n_repeats=3):

    r2s = []
    y_preds = []
    for i in range(n_repeats):
        y_pred = cross_val_predict(
            pipeline,
            X,
            y,
            cv=KFold(5, shuffle=True, random_state=i),
            n_jobs=-1,
        )
        r2s.append(
            r2_score(
                y,
                y_pred,
            )
        )
        y_preds.append(y_pred)

    y_pred = y_preds[np.argmax(r2s)]

    return r2s, y_pred
