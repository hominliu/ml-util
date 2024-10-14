from abc import ABC, abstractmethod
from itertools import compress, product
from typing import Any, Dict

import numpy as np
from colorama import Fore, Style

from ..utils.decorators import time_process
from .helpers import binary_model_analyses


class BasicTuner(ABC):
    def __init__(
        self,
        model_type: str = "",
        training_params: Dict[str, Any] = None,
        hyper_params_tuning_values: Dict[str, list] = None,
        fixed_hyper_params: Dict[str, Any] = None,
        find_min_matrix: bool = False,
        check_overfitting: bool = True,
        verbose_best_params: bool = True,
    ):
        """ML Model hyperparameters tuner

        Args:
            model_type: type of the model
            training_params: training parameters that will be used in both tuning
                process and final model training process
            hyper_params_tuning_values: hyperparameters with list of values that will
                be trained (tuned) over the tuning process
            fixed_hyper_params: hyperparameters with fixed value over the tuning
                process
            find_min_matrix: indicator for whether to find the minimum in the matrices
                from the tuning process for the best hyperparameters set
            check_overfitting: check if potential overfitting presented in the best
                iteration's cross-validation results if True
            verbose_best_params: print out the best hyperparameters if True
        """
        self.model_type = model_type
        self.training_params = training_params
        self._set_hyper_params_combination(
            hyper_params_tuning_values, base_params=fixed_hyper_params
        )
        self.find_min_matrix = find_min_matrix
        self.check_overfitting = check_overfitting
        self.verbose_best_params = verbose_best_params

    @abstractmethod
    def _cross_validate(self, train_x, train_y, params: dict):
        """cross-validation that provides cv scores for both training and testing
        (within cv) set based on provided hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def _train_final_model(
        self, train_X, train_y, test_X, test_y, best_params: Dict[str, Any], **kwargs
    ):
        """use the best hyperparameters set get from tuning to train the final model"""
        raise NotImplementedError

    def _load_best_hyper_params_post_tuning(
        self,
        train_cv_scores: list,
        test_cv_scores: list,
    ):
        """Load the best hyperparameter set based on model cross-validation results.
        Note:
        Different models use difference scoring in cross-validation.
        Sklearn provides different scoring while xgb.cv uses errors.
        This will affect if the function should find minimum or maximum
        from cv scores.

        Args:
            train_cv_scores: 1d array, list of cross-validation results (errors,
                accuracies, etc.) from training set.
            test_cv_scores: 1d array, list of cross-validation results (errors,
                accuracies, etc.) from testing set.
        Return:
            dictionary of the best combination of hyperparameters
        """
        assert len(train_cv_scores) == len(test_cv_scores), (
            "length of 'test_cv_scores' does not match "
            "the length of 'train_cv_scores'"
        )
        assert len(train_cv_scores) == len(self.hyper_params_sets), (
            "length of 'train_cv_scores' does not match the length of "
            "'hyper_params_sets'"
        )

        # data copy and conversion
        train_cv_scores = np.array(
            train_cv_scores if self.find_min_matrix else [-i for i in train_cv_scores]
        )
        test_cv_scores = np.array(
            test_cv_scores if self.find_min_matrix else [-i for i in test_cv_scores]
        )
        hyper_params_sets = self.hyper_params_sets.copy()

        if self.check_overfitting:
            print(
                "".join(
                    [
                        "Overfitting prevention mechanism: ",
                        Fore.CYAN,
                        "On",
                        Style.RESET_ALL,
                    ]
                )
            )
            not_overfitting_result_idx = (
                abs(train_cv_scores - test_cv_scores) / train_cv_scores <= 0.1
            )

            if any(not_overfitting_result_idx):
                hyper_params_sets = list(
                    compress(hyper_params_sets, not_overfitting_result_idx)
                )
                best_params_index = test_cv_scores[not_overfitting_result_idx].argmin()
            else:
                print(
                    "".join(
                        [
                            Fore.RED,
                            "Warning! Potential overfitting detected, ",
                            "please check.",
                            Style.RESET_ALL,
                        ]
                    )
                )
                best_params_index = test_cv_scores.argmin()
        else:
            print(
                "".join(
                    [
                        "Over-fitting prevention mechanism: ",
                        Fore.MAGENTA,
                        "Off",
                        Style.RESET_ALL,
                    ]
                )
            )
            best_params_index = test_cv_scores.argmin()

        best_params = hyper_params_sets[best_params_index]

        if self.verbose_best_params:
            print("The best parameters sets:")
            for key, value in best_params.items():
                print("{" if key == list(best_params)[0] else " ", end="")
                if isinstance(value, int) | isinstance(value, float):
                    print("".join(["'", key, "': ", str(value)]), end="")
                else:
                    print("".join(["'", key, "': '", str(value), "'"]), end="")
                print("}" if key == list(best_params)[-1] else ",")

        return best_params

    @time_process(process_name="hyperparameters Tuning")
    def _model_hyper_params_tuning(self, train_X, train_y):
        """Tune model hyperparameters through cross-validations

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
        Return:
            train_cv_scores: list of training cv scores
            test_cv_scores: list of testing cv scores

        """
        train_cv_scores = []
        test_cv_scores = []

        print(f"Start tuning hyperparameters: ({len(self.hyper_params_sets)} sets)")
        for params in self.hyper_params_sets:
            train_score, test_score = self._cross_validate(
                train_X,
                train_y,
                params=params,
            )
            train_cv_scores.append(train_score)
            test_cv_scores.append(test_score)
        return train_cv_scores, test_cv_scores

    def _set_hyper_params_combination(self, params_to_grid, base_params: dict = None):
        """Load combinations of provided hyperparameters into iterable list

        Args:
            params_to_grid: dictionary with key as hyperparameter and value as a
                list of hyperparameter values, a dictionary containing hyperparameters
                that will be tuned over cross-validation
            base_params: dictionary with key as the hyperparameter and value as the
                hyperparameter value, a dictionary containing the hyperparameters that
                will be constant over cross-validation
        Return:
            a list of dictionaries with unique combination of hyperparameters
        """
        if not base_params:
            base_params = {}

        hyper_params_sets = []

        for r in product(*params_to_grid.values()):
            hyper_parameters = base_params.copy()
            hyper_parameters.update(dict(zip([*params_to_grid.keys()], r)))
            hyper_params_sets.append(hyper_parameters)

        self.hyper_params_sets = hyper_params_sets

    def tune(self, train_X, train_y, test_X, test_y, **kwargs):
        """Tune the model and perform analyses

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, training data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            kwargs:
                _train_final_model(method): arguments varies across different child
                    Tuner class
                binary_model_analyses: (see function for details)
                    class_labels
                    classification_threshold
                    run_model_performance_analysis
                    run_feature_importance_analysis
                    run_roc_auc_analysis
                    run_threshold_analysis
                    plot_scores
        Return:

        """
        print("===================================================")

        # Cross-validation
        train_cv_scores, test_cv_scores = self._model_hyper_params_tuning(
            train_X,
            train_y,
        )

        # Load best parameters based on testing set's error
        best_params = self._load_best_hyper_params_post_tuning(
            train_cv_scores,
            test_cv_scores,
        )
        print(f"{self.model_type} hyperparameters tuning end.")

        # train final model
        final_model, train_proba, test_proba = self._train_final_model(
            train_X, train_y, test_X, test_y, best_params, **kwargs
        )

        # performance analysis
        binary_model_analyses_kwargs = {
            "class_labels": ["False", "True"],
            "classification_threshold": 0.5,
            "analyze_model_performance": True,
            "plot_shap": False,
            "plot_roc_auc": True,
            "plot_thresholds": False,
            "plot_proba": False,
        }
        for k in binary_model_analyses_kwargs.keys():
            if k in kwargs:
                binary_model_analyses_kwargs.update({k: kwargs.get(k)})
        binary_model_analyses(
            final_model,
            train_X,
            train_y,
            train_proba,
            test_X,
            test_y,
            test_proba,
            **binary_model_analyses_kwargs,
        )

        return final_model
