import numpy as np
from functools import reduce
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Literal

# https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
ScoringType = Literal[
    # Classification
    "accuracy",
    "balanced_accuracy",
    "average_precision",
    "neg_brier_score",
    "f1",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "f1_samples",
    "neg_log_loss",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
    # Regression
    "neg_mean_squared_error",
    "r2",
]


def best_search(model, params: dict, train_x, train_y, scoring: ScoringType = "balanced_accuracy", random_iter=True):
    num_params = reduce(lambda x, y: x * y, [len(param_values) for param_values in params.values()])
    print(f"Всего параметров в сетке: {num_params}")

    common_params = {"cv": 5, "scoring": scoring, "n_jobs": -1, "refit": True}
    if random_iter:
        grid = RandomizedSearchCV(model, params, **common_params, n_iter=int(np.ceil(0.6 * num_params)))
    else:
        grid = GridSearchCV(model, params, **common_params)

    grid.fit(train_x, train_y)

    print("Model best score: ", grid.best_score_)
    print("Model best params: ", grid.best_params_)

    return grid
