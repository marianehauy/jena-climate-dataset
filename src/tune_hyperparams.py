import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def tune_lgbm_hyperparameters(
    model, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), verbose=1
):
    """
    Tune hyperparameters for a given model using RandomizedSearchCV.

    This function performs hyperparameter tuning for a given model using RandomizedSearchCV
    with a predefined parameter distribution. It uses TimeSeriesSplit for cross-validation
    and negative mean absolute error as the scoring metric.

    Parameters:
    model (object): The base model to be tuned.
    X_train (array-like): The input features for training.
    y_train (array-like): The target values for training.
    cv (object, optional): Cross-validation strategy. Defaults to tscv (TimeSeriesSplit).
    verbose (bool, optional): If True, print verbose output. Defaults to False.

    Returns:
    dict: A dictionary containing the best hyperparameters found during the search.
    """
    # 1. Ajuste inicial dos hiperparâmetros com RandomizedSearchCV
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "num_leaves": stats.randint(30, 200),
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": stats.randint(20, 50),
    }

    tscv = TimeSeriesSplit(n_splits=5)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=verbose,
        random_state=42,
    )

    # Ajuste do RandomizedSearchCV
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    if not verbose:
        print(f"Melhores hiperparâmetros: {best_params}")
    return best_params
