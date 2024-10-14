from typing import Dict, Tuple

import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import cross_validate


def _cross_validator_adaboost(
    training_X,
    training_y,
    params: Dict = None,
    n_fold: int = 5,
    random_seed: int = 2731,
) -> Tuple[float, float]:
    """AdaBoost cross-validator that provides cv scores for both
    training and testing set based on provided hyperparameters

    Args:
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        params:
            dictionary with key as model hyperparameter and value
            as hyperparameter value
        n_fold:
            number of folds for cross-validation
        random_seed:
            control randomness of the model training in order to get
            repetitive results.
    Return:
        average training balanced-accuracy
        average testing balanced-accuracy
    """

    if params is None:
        raise ValueError("No parameters had been passed for cross-validation.")

    cv_results = cross_validate(
        AdaBoostClassifier(
            learning_rate=params.get("learning_rate"),
            n_estimators=params.get("n_estimators"),
            random_state=random_seed,
        ),
        training_X.values,
        training_y.values,
        cv=n_fold,
        scoring="balanced_accuracy",
        return_train_score=True,
    )

    return (cv_results["train_score"].mean(), cv_results["test_score"].mean())


def _cross_validator_balanced_random_forest(
    training_X,
    training_y,
    params: Dict = None,
    n_fold: int = 5,
    random_seed: int = 2731,
) -> Tuple[float, float]:
    """BalancedRandomForest cross-validator that provides cv scores for
    both training and testing set based on provided hyperparameters

    Args:
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        params:
            dictionary with key as model hyperparameter and value
            as hyperparameter value
        n_fold:
            training parameter, number of folds for cross-validation
        random_seed:
            control randomness of the model training in order to get
            repetitive results.
    Return:
        average training balanced-accuracy
        average testing balanced-accuracy
    """
    if params is None:
        raise ValueError("No parameters had been passed for cross-validation.")

    cv_results = cross_validate(
        BalancedRandomForestClassifier(
            max_depth=params.get("max_depth"),
            max_features=params.get("max_features"),
            max_samples=params.get("max_samples"),
            n_estimators=params.get("n_estimators"),
            random_state=random_seed,
        ),
        training_X.values,
        training_y.values,
        cv=n_fold,
        scoring="balanced_accuracy",
        return_train_score=True,
    )

    return (cv_results["train_score"].mean(), cv_results["test_score"].mean())


def _cross_validator_extra_trees(
    training_X,
    training_y,
    params: Dict = None,
    n_fold: int = 5,
    random_seed: int = 2731,
) -> Tuple[float, float]:
    """ExtraTrees cross-validator that provides cv scores for both
    training and testing set based on provided hyperparameters

    Args:
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        params:
            dictionary with key as model hyperparameter and value
            as hyperparameter value
        n_fold:
            training parameter, number of folds for cross-validation
        random_seed:
            control randomness of the model training in order to get
            repetitive results.
    Return:
        average training balanced-accuracy
        average testing balanced-accuracy
    """
    if params is None:
        raise ValueError("No parameters had been passed for cross-validation.")

    cv_results = cross_validate(
        ExtraTreesClassifier(
            max_depth=params.get("max_depth"),
            max_features=params.get("max_features"),
            n_estimators=params.get("n_estimators"),
            random_state=random_seed,
        ),
        training_X.values,
        training_y.values,
        cv=n_fold,
        scoring="balanced_accuracy",
        return_train_score=True,
    )

    return (cv_results["train_score"].mean(), cv_results["test_score"].mean())


def _cross_validator_random_forest(
    training_X,
    training_y,
    params: Dict = None,
    n_fold: int = 5,
    random_seed: int = 2731,
) -> Tuple[float, float]:
    """RandomForest cross-validator that provides cv scores for both
    training and testing set based on provided hyperparameters

    Args:
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        params:
            dictionary with key as model hyperparameter and value
            as hyperparameter value
        n_fold:
            training parameter, number of folds for cross-validation
        random_seed:
            control randomness of the model training in order to get
            repetitive results.
    Return:
        average training balanced-accuracy
        average testing balanced-accuracy
    """
    if params is None:
        raise ValueError("No parameters had been passed for cross-validation.")

    cv_results = cross_validate(
        RandomForestClassifier(
            max_depth=params.get("max_depth"),
            max_features=params.get("max_features"),
            max_samples=params.get("max_samples"),
            n_estimators=params.get("n_estimators"),
            random_state=random_seed,
        ),
        training_X.values,
        training_y.values,
        cv=n_fold,
        scoring="balanced_accuracy",
        return_train_score=True,
    )

    return (cv_results["train_score"].mean(), cv_results["test_score"].mean())


def _cross_validator_xgboost(
    training_data,
    params: Dict = None,
    n_fold: int = 5,
    num_boost: int = 200,
    early_stop: int = 5,
    random_seed: int = 2731,
) -> Tuple[float, float]:
    """XGBoost cross-validator that provides cv scores for both
    training and testing set based on provided hyperparameters

    Args:
        training_data: n_samples by n_features matrix,
            transformed dmatrix of training data
        params:
            dictionary with key as model hyperparameter and value
            as hyperparameter value
        n_fold:
            training parameter, number of folds for cross-validation
        num_boost:
            training parameter for XGBoost
        early_stop:
            training parameter for XGBoost
        random_seed:
            control randomness of the model training in order to get
            repetitive results.
    Return:
        average training error
        average testing error
    """

    if params is None:
        raise ValueError("No parameters had been passed for cross-validation.")

    cv_results = xgb.cv(
        dtrain=training_data,
        params=params,
        nfold=n_fold,
        stratified=True,
        num_boost_round=num_boost,
        early_stopping_rounds=early_stop,
        metrics="error",
        as_pandas=True,
        seed=random_seed,
    )

    best_on_train_idx = cv_results["test-error-mean"].idxmin()

    return (
        float(cv_results.iloc[best_on_train_idx]["train-error-mean"]),
        float(cv_results.iloc[best_on_train_idx]["test-error-mean"]),
    )


CROSS_VALIDATORS = {
    "AdaBoost": {
        "cross_validator": _cross_validator_adaboost,
        "find_min_matrix": False,
    },
    "BalancedRandomForest": {
        "cross_validator": _cross_validator_balanced_random_forest,
        "find_min_matrix": False,
    },
    "ExtraTrees": {
        "cross_validator": _cross_validator_extra_trees,
        "find_min_matrix": False,
    },
    "RandomForest": {
        "cross_validator": _cross_validator_random_forest,
        "find_min_matrix": False,
    },
    "XGBoost": {"cross_validator": _cross_validator_xgboost, "find_min_matrix": True},
}
