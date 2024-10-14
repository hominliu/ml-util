import contextlib
import os
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import cross_validate

from ..basic_tuner import BasicTuner


class BinaryTunerAdaBoost(BasicTuner):
    def __init__(
        self,
        set_n_estimators: int = None,
        set_learning_rate: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for AdaBoost Model and present training results
        with best set of hyperparameters.
        The default setting will iterate through combinations of 2 AdaBoost
        hyperparameters:
        6 sets of [n_estimators] ([50, 100, 150, 200, 250, 300])
        and 5 sets of [learning_rate] ([0.2, 0.1, 0.05, 0.01, 0.001]).
        User can all use "set_n_estimators" and "set_learning_rate" to set
        each hyperparameter as single value.

        Args:
            set_learning_rate: set AdaBoost hyperparameter [learning_rate] to a single
                value.
            set_n_estimators: set AdaBoost hyperparameter [n_estimators] to a single
                value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        super().__init__(
            model_type="AdaBoost",
            training_params={
                "n_fold": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "n_estimators": (
                    [set_n_estimators]
                    if set_n_estimators
                    else [50, 100, 150, 200, 250, 300]
                ),
                "learning_rate": (
                    [set_learning_rate]
                    if set_learning_rate
                    else [0.2, 0.1, 0.05, 0.01, 0.001]
                ),
            },
            fixed_hyper_params=None,
            find_min_matrix=False,
        )

    def _cross_validate(self, train_X, train_y, params: dict) -> Tuple[float, float]:
        """AdaBoost cross-validator that provides cv scores for both training and
        testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as the
                hyperparameter value
        Return:
            average training balanced-accuracy
            average testing balanced-accuracy
        """
        cv_results = cross_validate(
            AdaBoostClassifier(
                n_estimators=params.get("n_estimators"),
                learning_rate=params.get("learning_rate"),
                random_state=self.training_params.get("random_state"),
            ),
            train_X.values,
            train_y.values,
            cv=self.training_params.get("n_fold"),
            scoring="balanced_accuracy",
            return_train_score=True,
        )

        return cv_results["train_score"].mean(), cv_results["test_score"].mean()

    def _train_final_model(
        self, train_X, train_y, test_X, test_y, best_params: Dict[str, Any], **kwargs
    ):
        """train AdaBoost final model with hyperparameters get from tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
        Return:
            fitted
        """
        final_model = AdaBoostClassifier(
            n_estimators=best_params.get("n_estimators"),
            learning_rate=best_params.get("learning_rate"),
            random_state=self.training_params.get("random_state"),
        )
        final_model.fit(train_X, train_y)

        train_proba = final_model.predict_proba(train_X)[:, 1]
        test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba


class BinaryTunerBalancedRandomForest(BasicTuner):
    def __init__(
        self,
        set_n_estimators: int = None,
        max_depth_max: int = 10,
        set_max_depth: int = None,
        set_max_features: float = None,
        set_max_samples: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for BalancedRandomForest Model and present
        training results with best set of hyperparameters.

        The default setting will iterate through combinations of 4 the hyperparameters:
        4 sets of [n_estimators] ([50, 100, 150, 200]),
        4 sets of [max_depth] ([7, 8, 9, 10]),
        4 sets of [max_sample] ([1.0, 0.75, 0.5, 0.25])
        and 4 sets of [max_features] ([[1.0, 0.75, 0.5, 0.25]]).
        User can use "max_depth_max" to decide the maximum of [max_depth]
        and function will auto-generate a list of 4 consecutive integers
        with maximum as the specified max_depth_max.
        For Example, if "max_depth_max" = 7, the function will iterate
        through [max_depth] = [4, 5, 6, 7].
        User can all use "set_max_depth", "set_max_features",
        "set_max_samples", and "set_n_estimators" to set each hyperparameter
        as single value.

        Args:
            max_depth_max: the maximum of the max_depth in
                BalancedRandomForestClassifier that hyperparameters tuning will
                iterate through.
            set_max_depth: set BalancedRandomForestClassifier hyperparameter
                [max_depth] to a single value.
            set_n_estimators: set BalancedRandomForestClassifier hyperparameter
                [n_estimators] to a single value.
            set_max_features: set BalancedRandomForestClassifier hyperparameter
                [max_features] to a single value.
            set_max_samples: set BalancedRandomForestClassifier hyperparameter
                [max_samples] to a single value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        super().__init__(
            model_type="BalancedRandomForest",
            training_params={
                "n_fold": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "n_estimators": (
                    [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
                ),
                "max_depth": (
                    [set_max_depth]
                    if set_max_depth
                    else list(range(max_depth_max + 1))[-4:]
                ),
                "max_features": (
                    [set_max_features] if set_max_features else [1.0, 0.75, 0.5, 0.25]
                ),
                "max_samples": (
                    [set_max_samples] if set_max_samples else [1.0, 0.75, 0.5, 0.25]
                ),
            },
            fixed_hyper_params=None,
            find_min_matrix=False,
        )

    def _cross_validate(self, train_X, train_y, params: dict) -> Tuple[float, float]:
        """BalancedRandomForest cross-validator that provides cv scores for both
        training and testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as the
                hyperparameter value
        Return:
            average training balanced-accuracy
            average testing balanced-accuracy
        """
        cv_results = cross_validate(
            BalancedRandomForestClassifier(
                n_estimators=params.get("n_estimators"),
                max_depth=params.get("max_depth"),
                max_features=params.get("max_features"),
                random_state=self.training_params.get("random_state"),
                max_samples=params.get("max_samples"),
            ),
            train_X.values,
            train_y.values,
            cv=self.training_params.get("n_fold"),
            scoring="balanced_accuracy",
            return_train_score=True,
        )

        return cv_results["train_score"].mean(), cv_results["test_score"].mean()

    def _train_final_model(
        self, train_X, train_y, test_X, test_y, best_params: Dict[str, Any], **kwargs
    ):
        """train BalancedRandomForest final model with hyperparameters get from
        tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
        Return:
            fitted
        """
        final_model = BalancedRandomForestClassifier(
            n_estimators=best_params.get("n_estimators"),
            max_depth=best_params.get("max_depth"),
            max_samples=best_params.get("max_samples"),
            random_state=self.training_params.get("random_state"),
            max_features=best_params.get("max_features"),
        )
        final_model.fit(train_X, train_y)

        train_proba = final_model.predict_proba(train_X)[:, 1]
        test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba


class BinaryTunerExtraTrees(BasicTuner):
    def __init__(
        self,
        set_n_estimators: int = None,
        max_depth_max: int = 10,
        set_max_depth: int = None,
        set_max_features: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for ExtraTrees Model and present
        training results with best set of hyperparameters.

        The default setting will iterate through combinations of 3 hyperparameters:
        4 sets of [n_estimators] ([50, 100, 150, 200]),
        4 sets of [max_depth] ([7, 8, 9, 10]),
        and 4 sets of [max_features] ([[1.0, 0.75, 0.5, 0.25]]).
        User can use "max_depth_max" to decide the maximum of [max_depth] and class
        will auto-generate a list of 4 consecutive integers with maximum as the
        specified max_depth_max.
        For Example, if "max_depth_max" = 7, the function will iterate through
        [max_depth] = [4, 5, 6, 7].
        User can all use "set_max_depth", "set_max_features", and "set_n_estimators"
        to set each hyperparameter
        as single value.

        Args:
            set_n_estimators: set ExtraTreesClassifier hyperparameter [n_estimators]
                to a single value.
            max_depth_max: the maximum of the max_depth in ExtraTreesClassifier that
                hyperparameters tuning will iterate through.
            set_max_depth: set ExtraTreesClassifier hyperparameter [max_depth] to a
                single value.
            set_max_features: set ExtraTreesClassifier hyperparameter [max_features]
                to a single value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        assert max_depth_max > 0, "[max_depth] should be positive integer."

        super().__init__(
            model_type="ExtraTrees",
            training_params={
                "n_fold": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "n_estimators": (
                    [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
                ),
                "max_depth": (
                    [set_max_depth]
                    if set_max_depth
                    else list(range(max_depth_max + 1))[-4:]
                ),
                "max_features": (
                    [set_max_features] if set_max_features else [1.0, 0.75, 0.5, 0.25]
                ),
            },
            fixed_hyper_params=None,
            find_min_matrix=False,
        )

    def _cross_validate(self, train_X, train_y, params: dict) -> Tuple[float, float]:
        """ExtraTrees cross-validator that provides cv scores for both training and
        testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as the
                hyperparameter value
        Return:
            average training balanced-accuracy
            average testing balanced-accuracy
        """
        cv_results = cross_validate(
            ExtraTreesClassifier(
                n_estimators=params.get("n_estimators"),
                max_depth=params.get("max_depth"),
                max_features=params.get("max_features"),
                random_state=self.training_params.get("random_state"),
            ),
            train_X.values,
            train_y.values,
            cv=self.training_params.get("n_fold"),
            scoring="balanced_accuracy",
            return_train_score=True,
        )

        return cv_results["train_score"].mean(), cv_results["test_score"].mean()

    def _train_final_model(
        self, train_X, train_y, test_X, test_y, best_params: Dict[str, Any], **kwargs
    ):
        """train ExtraTrees final model with hyperparameters get from tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
        Return:
            fitted
        """
        final_model = ExtraTreesClassifier(
            n_estimators=best_params.get("n_estimators"),
            max_depth=best_params.get("max_depth"),
            max_features=best_params.get("max_features"),
            random_state=self.training_params.get("random_state"),
        )
        final_model.fit(train_X, train_y)

        train_proba = final_model.predict_proba(train_X)[:, 1]
        test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba


class BinaryTunerLightGBM(BasicTuner):
    def __init__(
        self,
        max_depth_max: int = 10,
        set_max_depth: int = None,
        set_scale_pos_weight: Optional[float] = None,
        set_bagging_fraction: float = None,
        set_feature_fraction: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for LightGBM Model and present training results
        with best set of hyperparameters.

        The default setting will iterate through combinations of 3 LightGBM
        hyperparameters:
        5 sets of [bagging_fraction] ([1.0, 0.875, 0.75, 0.625, 0.5]),
        5 sets of [colsample_bytree] [1.0, 0.875, 0.75, 0.625, 0.5],
        and 5 sets of [max_depth] ([6, 7, 8, 9, 10]).
        User can use "max_depth_max" to decide the maximum of [max_depth] and the
        function will auto-generate a list of 5 consecutive integers with
        maximum as the specified max_depth_max.
        For Example, if "max_depth_max" = 7, the function will iterate through
        [max_depth] = [3, 4, 5, 6, 7].
        User can all use "set_subsample", "set_max_depth", and "set_colsample_bytree"
         to set each hyperparameter as single
        value.

        Args:
            max_depth_max: the maximum of the max_depth in LightGBM that the
                hyperparameters tuning will iterate through.
            set_max_depth: set LightGBM hyperparameter [max_depth] to a single value.
            set_scale_pos_weight: set LightGBM hyperparameter [scale_pos_weight] to a
                single value.
            set_bagging_fraction: set LightGBM hyperparameter [set_subsample] to a
                single value.
            set_feature_fraction: set LightGBM hyperparameter [colsample_bytree] to a
                single value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        # Check hyperparameters
        if set_scale_pos_weight:
            assert (
                set_scale_pos_weight >= 0
            ), "[scale_pos_weight] should be larger than or equal to 0."
        assert max_depth_max > 0, "[max_depth_max] should be positive integer."

        super().__init__(
            model_type="XGBoost",
            training_params={
                "num_boost_round": 200,
                "n_fold": 5,
                "early_stopping_rounds": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "max_depth": (
                    [set_max_depth]
                    if set_max_depth
                    else list(range(max_depth_max + 1))[-5:]
                ),
                "bagging_fraction": (
                    [set_bagging_fraction]
                    if set_bagging_fraction
                    else [1.0, 0.875, 0.75, 0.625, 0.5]
                ),  # equivalent to "subsample"
                "feature_fraction": (
                    [set_feature_fraction]
                    if set_feature_fraction
                    else [1.0, 0.875, 0.75, 0.625, 0.5]
                ),  # equivalent to "colsample_bytree"
            },
            fixed_hyper_params={
                "objective": "binary",
                "learning_rate": 0.1,
                "num_leaves": None,
                "scale_pos_weight": set_scale_pos_weight,
            },
            find_min_matrix=True,
        )

        # update hyperparameters: num_leaves
        for param_set in self.hyper_params_sets:
            param_set.update({"num_leaves": pow(2, param_set.get("max_depth")) - 1})

    def _cross_validate(self, train_X, train_y, params: dict) -> Tuple[float, float]:
        """LightGBM cross-validator that provides cv scores for both training and
        testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as
                hyperparameter value
        Return:
            average training error
            average testing error
        """
        train_data = lgb.Dataset(train_X, label=train_y)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            cv_results = lgb.cv(
                params=params,
                train_set=train_data,
                num_boost_round=self.training_params.get("num_boost_round"),
                nfold=self.training_params.get("n_fold"),
                stratified=True,
                metrics="",  # corresponding to specified objective
                seed=self.training_params.get("random_state"),
                eval_train_metric=True,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.training_params.get(
                            "early_stopping_rounds"
                        ),
                    )
                ],
            )

        best_on_valid_idx = cv_results["valid binary_logloss-mean"].index(
            min(cv_results["valid binary_logloss-mean"])
        )

        return (
            float(cv_results["train binary_logloss-mean"][best_on_valid_idx]),
            float(cv_results["valid binary_logloss-mean"][best_on_valid_idx]),
        )

    def _train_final_model(
        self,
        train_X,
        train_y,
        test_X,
        test_y,
        best_params: Dict[str, Any],
        use_default_api: bool = False,
        **kwargs,
    ):
        """train LightGBM final model with hyperparameters get from tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
            use_default_api: use xgboost api to train model if True, else use sklearn
                api to train the model
        Return:
            fitted XGBClassifier
        """
        if use_default_api:
            train_data = lgb.Dataset(train_X, label=train_y)
            test_data = lgb.Dataset(test_X, label=test_y)

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                final_model = lgb.train(
                    params=best_params,
                    train_set=train_data,
                    num_boost_round=self.training_params.get("num_boost_round"),
                    valid_sets=[test_data],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=self.training_params.get(
                                "early_stopping_rounds"
                            ),
                        )
                    ],
                )

            train_proba = final_model.predict(
                train_X, num_iteration=final_model.best_iteration
            )
            test_proba = final_model.predict(
                test_X, num_iteration=final_model.best_iteration
            )
        else:
            final_model = lgb.LGBMClassifier(
                num_leaves=best_params.get("num_leaves"),
                max_depth=best_params.get("max_depth"),
                learning_rate=best_params.get("learning_rate"),
                subsample=best_params.get("bagging_fraction"),
                colsample_bytree=best_params.get("feature_fraction"),
                scale_pos_weight=best_params.get("scale_pos_weight"),
            )
            final_model.fit(
                train_X,
                train_y,
                eval_set=[(test_X, test_y)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.training_params.get(
                            "early_stopping_rounds"
                        ),
                    )
                ],
            )

            train_proba = final_model.predict_proba(train_X)[:, 1]
            test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba


class BinaryTunerRandomForest(BasicTuner):
    def __init__(
        self,
        set_n_estimators: int = None,
        max_depth_max: int = 10,
        set_max_depth: int = None,
        set_max_features: float = None,
        set_max_samples: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for RandomForest Model and present
        training results with best set of hyperparameters.

        The default setting will iterate through combinations of 4 hyperparameters:
        4 sets of [n_estimators] ([50, 100, 150, 200]),
        4 sets of [max_depth] ([7, 8, 9, 10]),
        4 sets of [max_sample] ([1.0, 0.75, 0.5, 0.25])
        and 4 sets of [max_features] ([[1.0, 0.75, 0.5, 0.25]]).
        User can use "max_depth_max" to decide the maximum of [max_depth]
        and function will auto-generate a list of 4 consecutive integers
        with maximum as the specified max_depth_max.
        For Example, if "max_depth_max" = 7, the function will iterate
        through [max_depth] = [4, 5, 6, 7].
        User can all use "set_max_depth", "set_max_features",
        "set_max_samples", and "set_n_estimators" to set each hyperparameter
        as single value.

        Args:
            set_n_estimators: set RandomForestClassifier hyperparameter [n_estimators]
                to a single value.
            max_depth_max: the maximum of the max_depth in BalancedRandomForest that
                hyperparameters tuning will iterate through.
            set_max_depth: set RandomForestClassifier hyperparameter [max_depth] to a
                single value.
            set_max_features: set RandomForestClassifier hyperparameter [max_features]
                to a single value.
            set_max_samples: set RandomForestClassifier hyperparameter [max_samples] to
                a single value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        super().__init__(
            model_type="RandomForest",
            training_params={
                "n_fold": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "n_estimators": (
                    [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
                ),
                "max_depth": (
                    [set_max_depth]
                    if set_max_depth
                    else list(range(max_depth_max + 1))[-4:]
                ),
                "max_features": (
                    [set_max_features] if set_max_features else [1.0, 0.75, 0.5, 0.25]
                ),
                "max_samples": (
                    [set_max_samples] if set_max_samples else [1.0, 0.75, 0.5, 0.25]
                ),
            },
            fixed_hyper_params=None,
            find_min_matrix=False,
        )

    def _cross_validate(self, train_X, train_y, params: dict):
        """RandomForest cross-validator that provides cv scores for both training and
        testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as the
                hyperparameter value
        Return:
            average training balanced-accuracy
            average testing balanced-accuracy
        """
        cv_results = cross_validate(
            RandomForestClassifier(
                n_estimators=params.get("n_estimators"),
                max_depth=params.get("max_depth"),
                max_features=params.get("max_features"),
                random_state=self.training_params.get("random_state"),
                max_samples=params.get("max_samples"),
            ),
            train_X.values,
            train_y.values,
            cv=self.training_params.get("n_fold"),
            scoring="balanced_accuracy",
            return_train_score=True,
        )

        return cv_results["train_score"].mean(), cv_results["test_score"].mean()

    def _train_final_model(
        self, train_X, train_y, test_X, test_y, best_params: Dict[str, Any], **kwargs
    ):
        """train RandomForest final model with hyperparameters get from tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
        Return:
            fitted
        """
        final_model = RandomForestClassifier(
            n_estimators=best_params.get("n_estimators"),
            max_depth=best_params.get("max_depth"),
            max_features=best_params.get("max_features"),
            random_state=self.training_params.get("random_state"),
            max_samples=best_params.get("max_samples"),
        )
        final_model.fit(train_X, train_y)

        train_proba = final_model.predict_proba(train_X)[:, 1]
        test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba


class BinaryTunerXGBoost(BasicTuner):
    def __init__(
        self,
        max_depth_max: int = 10,
        set_max_depth: int = None,
        # set_sample_weights: list = None,
        set_scale_pos_weight: Optional[float] = None,
        set_subsample: float = None,
        set_colsample_bytree: float = None,
        random_state: int = 2731,
    ):
        """hyperparameters tuning for XGBoost Model and present training
        results with best set of hyperparameters.

        The default setting will iterate through combinations of 3 XGBoost
        hyperparameters:
        5 sets of [subsample] ([1.0, 0.875, 0.75, 0.625, 0.5]),
        5 sets of [colsample_bytree] [1.0, 0.875, 0.75, 0.625, 0.5],
        and 5 sets of [max_depth] ([6, 7, 8, 9, 10]).
        User can use "max_depth_max" to decide the maximum of [max_depth]
        and function will auto-generate a list of 5 consecutive integers
        with maximum as the specified max_depth_max.
        For Example, if "max_depth_max" = 7, the function will iterate
        through [max_depth] = [3, 4, 5, 6, 7].
        User can all use "set_subsample", "set_max_depth", and
        "set_colsample_bytree" to set each hyperparameter as single
        value.

        Args:
            max_depth_max: the maximum of the max_depth in XGBoost that the
                hyperparameters tuning will iterate through.
            set_max_depth: set XGBoost hyperparameter [max_depth] to a single value.
            set_scale_pos_weight: set XGBoost hyperparameter [scale_pos_weight] to a
                single value.
            set_subsample: set XGBoost hyperparameter [set_subsample] to a single
                value.
            set_colsample_bytree: set XGBoost hyperparameter [colsample_bytree] to a
                single value.
            random_state: control randomness of the model training in order to get
                repetitive results.
        """
        # TODO: enable set_sample_weights (in scikit-learn api mode)

        # Check hyperparameters
        if set_scale_pos_weight:
            assert (
                set_scale_pos_weight >= 0
            ), "[scale_pos_weight] should be larger than or equal to 0."
            # assert (
            #     set_sample_weights is None
            # ), "Can only use [scale_pos_weight] or [weight] at a time."
        # if set_sample_weights:
        #     assert (
        #         set_scale_pos_weight is None
        #     ), "Can only use [weight] or [scale_pos_weight] at a time."
        #     assert len(set_sample_weights) == len(training_X), (
        #         "The length of [weight] should be the same as the length of "
        #         "training data."
        #     )
        assert max_depth_max > 0, "[max_depth_max] should be positive integer."

        super().__init__(
            model_type="XGBoost",
            training_params={
                "n_fold": 5,
                "num_boost_round": 200,
                "early_stopping_rounds": 5,
                "random_state": random_state,
            },
            hyper_params_tuning_values={
                "max_depth": (
                    [set_max_depth]
                    if set_max_depth
                    else list(range(max_depth_max + 1))[-5:]
                ),
                "subsample": (
                    [set_subsample] if set_subsample else [1.0, 0.875, 0.75, 0.625, 0.5]
                ),
                "colsample_bytree": (
                    [set_colsample_bytree]
                    if set_colsample_bytree
                    else [1.0, 0.875, 0.75, 0.625, 0.5]
                ),
            },
            fixed_hyper_params={
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "tree_method": "exact",
                "scale_pos_weight": set_scale_pos_weight,
                "eval_metric": "error",
            },
            find_min_matrix=True,
        )

    def _cross_validate(self, train_X, train_y, params: dict) -> Tuple[float, float]:
        """XGBoost cross-validator that provides cv scores for both training and
        testing set based on provided hyperparameters

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            params: dictionary with key as model hyperparameter and value as the
                hyperparameter value
        Return:
            average training error
            average testing error
        """
        train_dmatrix = xgb.DMatrix(data=train_X, label=train_y)

        cv_results = xgb.cv(
            dtrain=train_dmatrix,
            params=params,
            nfold=self.training_params.get("n_fold"),
            stratified=True,
            num_boost_round=self.training_params.get("num_boost_round"),
            early_stopping_rounds=self.training_params.get("early_stopping_rounds"),
            metrics="error",
            as_pandas=True,
            seed=self.training_params.get("random_state"),
        )

        best_on_test_idx = cv_results["test-error-mean"].idxmin()

        return (
            float(cv_results.iloc[best_on_test_idx]["train-error-mean"]),
            float(cv_results.iloc[best_on_test_idx]["test-error-mean"]),
        )

    def _train_final_model(
        self,
        train_X,
        train_y,
        test_X,
        test_y,
        best_params: Dict[str, Any],
        use_default_api: bool = False,
        **kwargs,
    ):
        """train XGBoost final model with hyperparameters get from tuning process

        Args:
            train_X: n_samples by n_features matrix, training data from training set.
            train_y: 1d array-like, ground truth target values from training set.
            test_X: n_samples by n_features matrix, data from testing set.
            test_y: 1d array-like, ground truth target values from testing set.
            best_params: dictionary of the best combination of hyperparameters
            use_default_api: use xgboost api to train model if True, else use sklearn
                api to train the model
        Return:
            fitted XGBClassifier
        """
        if use_default_api:
            train_dmatrix = xgb.DMatrix(data=train_X, label=train_y)
            test_dmatrix = xgb.DMatrix(data=test_X, label=test_y)

            # Train the model with xgboost api
            final_model = xgb.train(
                params=best_params,
                dtrain=train_dmatrix,
                num_boost_round=self.training_params.get("num_boost_round"),
                evals=[(test_dmatrix, "Test")],
                early_stopping_rounds=self.training_params.get("early_stopping_rounds"),
                verbose_eval=False,
            )
            train_proba = final_model.predict(
                train_dmatrix, iteration_range=(0, final_model.best_iteration + 1)
            )
            test_proba = final_model.predict(
                test_dmatrix, iteration_range=(0, final_model.best_iteration + 1)
            )
            # use ntree_limit=xgb_final_model.best_ntree_limit
            # instead of ntree_limit for xgboost < 1.4
        else:
            # Train the model with sklearn api
            final_model = xgb.XGBClassifier(
                max_depth=best_params.get("max_depth"),
                learning_rate=best_params.get("learning_rate"),
                objective=best_params.get("objective"),
                tree_method=best_params.get("tree_method"),
                subsample=best_params.get("subsample"),
                colsample_bytree=best_params.get("colsample_bytree"),
                scale_pos_weight=best_params.get("scale_pos_weight", None),
                eval_metric=best_params.get("eval_metric"),
                early_stopping_rounds=self.training_params.get("early_stopping_rounds"),
            )
            final_model.fit(
                train_X,
                train_y,
                eval_set=[(test_X, test_y)],
                verbose=False,
            )
            train_proba = final_model.predict_proba(train_X)[:, 1]
            test_proba = final_model.predict_proba(test_X)[:, 1]

        return final_model, train_proba, test_proba
