import optuna
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_samples=1000, n_features=20, noise=0.3, random_state=42)

def objective(trial):
    # Decision Tree as base estimator
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    base_estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
    loss = trial.suggest_categorical("loss", ["linear", "square", "exponential"])

    model = AdaBoostRegressor(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return -scores.mean()

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)



from sklearn.linear_model import ElasticNet

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    max_iter = trial.suggest_int("max_iter", 1000, 20000)
    tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores.mean()  # maximize R²

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)


from sklearn.linear_model import Ridge

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1000.0, log=True)
    solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    max_iter = trial.suggest_int("max_iter", 1000, 50000)
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)

    model = Ridge(
        alpha=alpha,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return -scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)
