import datetime
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from _cross_validators import CROSS_VALIDATORS
from colorama import Fore


def _load_best_hyperparameter(
    training_cv_results: List,
    testing_cv_results: List,
    hyper_parameter_sets: List,
    check_overfitting: bool = True,
    model_type: str = "",
    verbose: bool = True,
):
    """Load best hyperparameter set based on model cross-validation
    results.

    Note:
    Different models use difference scoring in cross-validation.
    Sklearn provides different scoring while xgb.cv uses errors.
    This will affect if the function should find minimum or maximum
    from cv scores.

    Args:
        training_cv_results: 1d array,
            list of cross-validation results (errors, accuracies, etc.)
            from training set.
        testing_cv_results: 1d array,
            list of cross-validation results (errors, accuracies, etc.)
            from testing set.
        hyper_parameter_sets: 1d array,
            list of hyperparameters in the same order of cv_results
        check_overfitting:
            if True, check if potential overfitting presented in best
            iteration's cross-validation results.
        model_type:
            type of model. Supported values are the keys in variable
            CROSS_VALIDATORS
        verbose:
            if True, print out the best hyperparameters
    Return:
        dictionary of best combination of hyperparameters
    """
    assert len(testing_cv_results) == len(training_cv_results), (
        "length of 'testing_cv_results' does not match the length of "
        "'training_cv_results'"
    )
    assert len(hyper_parameter_sets) == len(training_cv_results), (
        "length of 'hyper_parameter_sets' does not match the length of "
        "'training_cv_results'"
    )
    if model_type not in CROSS_VALIDATORS.keys():
        raise ValueError(f"{model_type} is not supported model.")

    find_min_matrix = CROSS_VALIDATORS.get(model_type).get("find_min_matrix")
    # if True, the function will find minimum in cv scores.

    training_cv_results = np.array(
        training_cv_results if find_min_matrix else [-i for i in training_cv_results]
    )
    testing_cv_results = np.array(
        testing_cv_results if find_min_matrix else [-i for i in testing_cv_results]
    )
    hyper_parameter_sets = hyper_parameter_sets.copy()

    if check_overfitting:
        print(
            "".join(["Overfitting prevention mechanism: ", Fore.CYAN, "On", Fore.BLACK])
        )
        not_overfitting_result_idx = (
            abs(training_cv_results - testing_cv_results) / training_cv_results <= 0.1
        )

        if any(not_overfitting_result_idx):
            hyper_parameter_sets = [
                s
                for s, keep in zip(hyper_parameter_sets, not_overfitting_result_idx)
                if keep
            ]
            best_params_index = testing_cv_results[not_overfitting_result_idx].argmin()
        else:
            print(
                "".join(
                    [
                        Fore.RED,
                        "Warning! Potential overfitting detected, " "please check.",
                        Fore.BLACK,
                    ]
                )
            )
            best_params_index = testing_cv_results.argmin()
    else:
        print(
            "".join(
                ["Over-fitting prevention mechanism: ", Fore.MAGENTA, "Off", Fore.BLACK]
            )
        )
        best_params_index = testing_cv_results.argmin()

    best_params = hyper_parameter_sets[best_params_index]

    if verbose:
        print("The best parameters sets:")
        for key, value in best_params.items():
            print("{" if key == list(best_params)[0] else " ", end="")
            if isinstance(value, int) | isinstance(value, float):
                print("".join(["'", key, "': ", str(value)]), end="")
            else:
                print("".join(["'", key, "': '", str(value), "'"]), end="")
            print("}" if key == list(best_params)[-1] else ",")

    return best_params


def _load_hyperparameter_sets(
    parameters_to_grid: Dict, base_parameters: Dict = None
) -> List:
    """Load a list of combinations of provided hyperparameters

    Args:
        parameters_to_grid:
            dictionary with key as hyperparameter and value as a
            list of hyperparameter values, a dictionary containing
            hyperparameters that will be tuned over cross-validation
        base_parameters:
            dictionary with key as hyperparameter and value as
            hyperparameter value, a dictionary containing the
            hyperparameters that will be constant over cross-
            validation
    Return:
        a list of dictionaries with unique combination of
        hyperparameters
    """

    if not base_parameters:
        base_parameters = {}

    parameters_sets = []

    for r in product(*parameters_to_grid.values()):
        base_params = base_parameters.copy()
        base_params.update(dict(zip([*parameters_to_grid.keys()], r)))
        parameters_sets.append(base_params)

    return parameters_sets


def _model_hyper_parameter_tuning(
    *args, params_sets: List = None, model_type: str = "", **kwargs
) -> Tuple[List, List]:
    """Tune model hyperparameters through cross-validations

    Args:
        *args:
            positional arguments that will be used in each cross-
            validators
                training_data:
                    dmatrix used in _cross_validator_xgboost
                training_X:
                    training data used in _cross_validator_adaboost,
                training_y:
                    training label used in _cross_validator_adaboost,

        params_sets: list of dictionaries,
            dictionaries with key as model hyperparameter and value
            as hyperparameter value
        model_type:
            type of model. Supported values are the keys in variable
            CROSS_VALIDATORS
        **kwargs:
            keyword arguments that will be used in each cross-
            validators
    Return:
        cv_matrices_training:
            list of training cv scores
        cv_matrices_testing:
            list of testing cv scores
    """
    if params_sets is None:
        raise ValueError("No parameters for tuning.")

    if model_type not in CROSS_VALIDATORS.keys():
        raise ValueError(f"{model_type} is not supported model.")

    cv_matrices_training = []
    cv_matrices_testing = []

    print(f"Start tuning hyperparameters: ({len(params_sets)} sets)")
    cv_start_time = datetime.datetime.now()
    for params in params_sets:
        cv_result_training, cv_result_test = CROSS_VALIDATORS.get(model_type).get(
            "cross_validator"
        )(*args, params=params, **kwargs)
        cv_matrices_training.append(cv_result_training)
        cv_matrices_testing.append(cv_result_test)
    cv_end_time = datetime.datetime.now()
    cv_time = str(cv_end_time - cv_start_time).split(":")
    print(
        " ".join(
            [
                "hyperparameters Tuning Time:",
                f"{int(cv_time[0])} hours" if int(cv_time[0]) > 0 else "",
                f"{int(cv_time[1])} minutes" if int(cv_time[1]) > 0 else "",
                f"{round(float(cv_time[2]))} seconds.",
            ]
        )
    )

    return cv_matrices_training, cv_matrices_testing
