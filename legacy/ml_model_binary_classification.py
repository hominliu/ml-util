import warnings
from typing import List, Optional

import xgboost as xgb

# import catboost as cat
from imblearn.ensemble import BalancedRandomForestClassifier
from ml_model_analysis_utils import model_analyses_binary_classification
from ml_model_training_utils import (
    _load_best_hyperparameter,
    _load_hyperparameter_sets,
    _model_hyper_parameter_tuning,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)


def binary_classification_adaboost(
    training_X,
    training_y,
    testing_X,
    testing_y,
    set_learning_rate: float = None,
    set_n_estimators: int = None,
    **kwargs,
):
    """hyperparameters tuning for AdaBoost Model and present training
    results with best set of hyperparameters.
    The default setting will iterate through combinations of 2 AdaBoost
    hyperparameters:
    6 sets of [n_estimators] ([50, 100, 150, 200, 250, 300])
    and 5 sets of [learning_rate] ([0.2, 0.1, 0.05, 0.01, 0.001]).
    User can all use "set_n_estimators" and "set_learning_rate" to set
    each hyperparameter as single value.

    Args:
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        testing_X: n_samples by n_features matrix,
            training data from testing set.
        testing_y: 1d array-like,
            ground truth target values from testing set.
        set_learning_rate:
            set AdaBoost hyperparameter [learning_rate] to a single
            value.
        set_n_estimators:
            set AdaBoost hyperparameter [n_estimators] to a single
            value.
        **kwargs:
            class_labels:
                keyword arguments for model_performance_analysis()
                in model_analyses_binary_classification()
            check_overfitting: bool
                keyword arguments for _load_best_hyperparameter()
            classification_threshold: float
                keyword arguments for model_analyses_binary_classification()
            random_seed: int
                keyword arguments for _cross_validator_adaboost() in
                _model_hyper_parameter_tuning() and AdaBoostClassifier()
            run_model_performance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_feature_importance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_roc_auc_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_threshold_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            show_scores: bool
                keyword arguments for model_analyses_binary_classification().

    Return:
        fitted AdaBoostClassifier
    """
    model_type = "AdaBoost"

    print("==================================================")

    # Setting hyperparameters
    params_sets = _load_hyperparameter_sets(
        {
            "learning_rate": (
                [set_learning_rate]
                if set_learning_rate
                else [0.2, 0.1, 0.05, 0.01, 0.001]
            ),
            "n_estimators": (
                [set_n_estimators]
                if set_n_estimators
                else [50, 100, 150, 200, 250, 300]
            ),
        }
    )

    # Setting the training parameters
    n_fold = 5

    # Cross-validation
    training_cv_matrices, testing_cv_matrices = _model_hyper_parameter_tuning(
        training_X,
        training_y,
        params_sets=params_sets,
        model_type=model_type,
        n_fold=n_fold,
        random_seed=kwargs.get("random_seed", 2731),
    )
    # Load best parameters based on testing set's error
    best_params = _load_best_hyperparameter(
        training_cv_matrices,
        testing_cv_matrices,
        params_sets,
        model_type=model_type,
        check_overfitting=kwargs.get("check_overfitting", True),
    )
    print("\nAdaBoost hyperparameters tuning end.")

    # Train the model with AdaBoost
    adaboost_final_model = AdaBoostClassifier(
        learning_rate=best_params.get("learning_rate"),
        n_estimators=best_params.get("n_estimators"),
        random_state=kwargs.get("random_seed", 2731),
    )
    adaboost_final_model.fit(training_X, training_y)

    # Training Results Analyses
    model_analyses_binary_classification(
        adaboost_final_model,
        training_X,
        training_y,
        testing_X,
        testing_y,
        class_labels=kwargs.get("class_labels", ["False", "True"]),
        classification_threshold=kwargs.get("classification_threshold", 0.5),
        run_model_performance_analysis=kwargs.get(
            "run_model_performance_analysis", True
        ),
        run_feature_importance_analysis=kwargs.get(
            "run_feature_importance_analysis", False
        ),
        run_roc_auc_analysis=kwargs.get("run_roc_auc_analysis", True),
        run_threshold_analysis=kwargs.get("run_threshold_analysis", False),
        show_scores=kwargs.get("show_scores", False),
    )

    return adaboost_final_model


def binary_classification_balanced_random_forest(
    training_X,
    training_y,
    testing_X,
    testing_y,
    max_depth_max: int = 10,
    set_max_depth: int = None,
    set_max_features: float = None,
    set_max_samples: float = None,
    set_n_estimators: int = None,
    **kwargs,
):
    """hyperparameters tuning for BalancedRandomForest Model and present
    training results with best set of hyperparameters.
    The default setting will iterate through combinations of 4 hyper-
    parameters:
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
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        testing_X: n_samples by n_features matrix,
            training data from testing set.
        testing_y: 1d array-like,
            ground truth target values from testing set.
        max_depth_max:
            the maximum of the max_depth in BalancedRandomForestClassifier
            that hyperparameters tuning will iterate through.
        set_max_depth:
            set BalancedRandomForestClassifier hyperparameter
            [max_depth] to a single value.
        set_n_estimators:
            set BalancedRandomForestClassifier hyperparameter
            [n_estimators] to a single value.
        set_max_features:
            set BalancedRandomForestClassifier hyperparameter
            [max_features] to a single value.
        set_max_samples:
            set BalancedRandomForestClassifier hyperparameter
            [max_samples] to a single value.
        **kwargs:
            class_labels:
                keyword arguments for model_performance_analysis()
                in binary_classification_analyses()
            check_overfitting: bool
                keyword arguments for _load_best_hyperparameter()
            classification_threshold: float
                keyword arguments for binary_classification_analyses()
            random_seed: int
                keyword arguments for
                _cross_validator_balanced_random_forest() in
                _model_hyper_parameter_tuning() and
                BalancedRandomForestClassifier()
            run_model_performance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_feature_importance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_roc_auc_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_threshold_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            show_scores: bool
                keyword arguments for model_analyses_binary_classification().

    Return:
        fitted BalancedRandomForestClassifier
    """
    model_type = "BalancedRandomForest"

    # Check hyperparameters
    assert max_depth_max > 0, "[max_depth] should be positive integer."

    print("==================================================")

    # Setting hyperparameters
    params_sets = _load_hyperparameter_sets(
        {
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
            "n_estimators": (
                [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
            ),
        }
    )

    # Setting the training parameters
    n_fold = 5

    # Cross-validation
    training_cv_matrices, testing_cv_matrices = _model_hyper_parameter_tuning(
        training_X,
        training_y,
        params_sets=params_sets,
        model_type=model_type,
        n_fold=n_fold,
        random_seed=kwargs.get("random_seed", 2731),
    )
    # Load best parameters based on testing set's error
    best_params = _load_best_hyperparameter(
        training_cv_matrices,
        testing_cv_matrices,
        params_sets,
        model_type=model_type,
        check_overfitting=kwargs.get("check_overfitting", True),
    )
    print("\nBalancedRandomForest hyperparameters tuning end.")

    # Train the model with Balanced Random Forest
    balanced_random_forest_final_model = BalancedRandomForestClassifier(
        max_depth=best_params.get("max_depth"),
        max_samples=best_params.get("max_samples"),
        max_features=best_params.get("max_features"),
        n_estimators=best_params.get("n_estimators"),
        random_state=kwargs.get("random_seed", 2731),
    )
    balanced_random_forest_final_model.fit(training_X, training_y)

    model_analyses_binary_classification(
        balanced_random_forest_final_model,
        training_X,
        training_y,
        testing_X,
        testing_y,
        class_labels=kwargs.get("class_labels", ["False", "True"]),
        classification_threshold=kwargs.get("classification_threshold", 0.5),
        run_model_performance_analysis=kwargs.get(
            "run_model_performance_analysis", True
        ),
        run_feature_importance_analysis=kwargs.get(
            "run_feature_importance_analysis", False
        ),
        run_roc_auc_analysis=kwargs.get("run_roc_auc_analysis", True),
        run_threshold_analysis=kwargs.get("run_threshold_analysis", False),
        show_scores=kwargs.get("show_scores", False),
    )

    return balanced_random_forest_final_model


# def binary_classification_catboost(
#     X_train,
#     X_test,
#     Y_train,
#     Y_test,
#     cat_features,
#     class_labels,
#     set_scale_pos_weight=0,
#     set_sample_weights=None,
#     set_depth=0,
#     max_depth_max=10,
#     overfitting_prevent=True,
#     classify_threshold=0.5,
#     matrix_table=False,
#     threshold_max=0.75,
#     threshold_min=0.25
# ):
#
#     # Copy mutable data
#     X_train_cat = X_train.copy()
#     X_test_cat = X_test.copy()
#     Y_train = Y_train.copy()
#     Y_test = Y_test.copy()
#
#     # Copy the data and transform data type
#    X_train_cat[cat_features] = X_train_cat[cat_features].astype(str)
#     X_test_cat[cat_features] = X_test_cat[cat_features].astype(str)
#
#     # Check hyperparameters
#     if set_scale_pos_weight:
#         assert set_scale_pos_weight >= 0, \
#             "[scale_pos_weight] should be larger than or equal to 0."
#         assert set_sample_weights is None, \
#             "Can only use [scale_pos_weight] or [weight] at a time."
#     if set_sample_weights:
#         assert set_scale_pos_weight is None, \
#             "Can only use [weight] or [scale_pos_weight] at a time."
#         assert len(set_sample_weights) == len(X_train), \
#             "The length of [weight] should be the same as the length of " \
#             "training data."
#     assert max_depth_max > 0, "[max_depth_max] should be positive integer."
#
#     print("==================================================")
#
#     # Setting data
#     cat_train = cat.Pool(
#         data=X_train, label=Y_train, cat_features=cat_features
#     )
#     if set_sample_weights:
#         cat_train.set_weight(set_sample_weights)
#     cat_test = cat.Pool(
#         data=X_test, label=Y_test, cat_features=cat_features
#     )
#
#     # Setting training parameters
#     loss_func = 'Logloss'
#     n_fold = 5
#     num_boost = 200
#     early_stop = 5
#
#     # Setting lists for storing the results of each training process
#     CAT_TrainError = []
#     CAT_TestError = []
#     CAT_Parameters = []
#
#     # Setting the lists of hyperparameters for CatBoost
#     Subsample = [1.0, 0.875, 0.75, 0.625, 0.5]
#     if set_depth != 0:
#         MaxDepth = [set_depth]
#     else:
#         MaxDepth = list(range(max_depth_max + 1))[-5:]
#     ColSampleByLevel = [1.0, 0.875, 0.75, 0.625, 0.5]
#
#     # Setting counters
#     TotalSets = len(Subsample) * len(MaxDepth) * len(ColSampleByLevel)
#
#     for cv_subsample in Subsample:
#
#         for cv_max_depth in MaxDepth:
#
#             for cv_colsample_bylevel in ColSampleByLevel:
#
#                 # Setting the parameters
#                 Params = {'loss_function': loss_func,
#                           'eval_metric': loss_func,
#                           'learning_rate': 0.1,
#                           'subsample': cv_subsample,
#                           'depth': cv_max_depth,
#                           'rsm': cv_colsample_bylevel,
#                           'iterations': num_boost,
#                           'early_stopping_rounds': early_stop,
#                           'random_seed': 1308,
#                           'bootstrap_type': 'Bernoulli'}
#
#                 if set_scale_pos_weight:
#                     Params.update({'scale_pos_weight': set_scale_pos_weight})
#
#                 CAT_cv_results = cat.cv(pool = cat_train,
#                                         params = Params,
#                                         fold_count = n_fold,
#                                         stratified = True,
#                                         as_pandas = True,
#                                         logging_level = 'Silent')
#
#                 # Updating the lists
#                 best_on_train_idx = CAT_cv_results[
#                     'test-' + loss_func + '-mean'
#                     ].idxmin()
#                 CAT_TrainError.append(
#                     float(
#                         CAT_cv_results.iloc[best_on_train_idx][
#                             'train-' + loss_func + '-mean'
#                             ]
#                     )
#                 )
#                 CAT_TestError.append(
#                     float(
#                         CAT_cv_results.iloc[best_on_train_idx][
#                             'test-' + loss_func + '-mean'
#                             ]
#                     )
#                 )
#                 CAT_Parameters.append(Params)
#
#                 # End of the loop indicator
#                 TotalSets -= 1
#                 if((TotalSets % 25 == 0) & (TotalSets > 0)):
#                     print(
#                         ''.join(
#                             ["Still have ", str(TotalSets), " sets to go."]
#                         )
#                     )
#
#     # Error Tracking Table
#     CAT_ErrorTable = pd.DataFrame({'TrainError': CAT_TrainError,
#                                    'TestError': CAT_TestError,
#                                    'Parameters': CAT_Parameters})
#
#
#
#     # 3. Train the model with CatBoost
#         # a. Train the model
#     CAT_final_model = cat.CatBoost(params = CAT_BestParams)
#     CAT_final_model.fit(cat_train,
#                         eval_set = cat_test,
#                         verbose = False)
#
#     #4. Training Results
#     # 4.1. Train set
#     CAT_predict_proba_train = CAT_final_model.predict(
#         cat_train, prediction_type = 'Probability'
#     )[:, 1]
#
#     # 4.2. Test set
#     CAT_predict_proba_test = CAT_final_model.predict(
#         cat_test, prediction_type = 'Probability'
#     )[:, 1]
#
#     return CAT_final_model


def binary_classification_extra_trees(
    training_X,
    training_y,
    testing_X,
    testing_y,
    max_depth_max: int = 10,
    set_max_depth: int = None,
    set_max_features: float = None,
    set_n_estimators: int = None,
    **kwargs,
):
    model_type = "ExtraTrees"

    # Check hyperparameters
    assert max_depth_max > 0, "[max_depth] should be positive integer."

    print("==================================================")

    # Setting hyperparameters
    params_sets = _load_hyperparameter_sets(
        {
            "max_depth": (
                [set_max_depth]
                if set_max_depth
                else list(range(max_depth_max + 1))[-4:]
            ),
            "max_features": (
                [set_max_features] if set_max_features else [1.0, 0.75, 0.5, 0.25]
            ),
            "n_estimators": (
                [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
            ),
        }
    )

    # Setting the training parameters
    n_fold = 5

    # Cross-validation
    training_cv_matrices, testing_cv_matrices = _model_hyper_parameter_tuning(
        training_X,
        training_y,
        params_sets=params_sets,
        model_type=model_type,
        n_fold=n_fold,
        random_seed=kwargs.get("random_seed", 2731),
    )
    # Load best parameters based on testing set's error
    best_params = _load_best_hyperparameter(
        training_cv_matrices,
        testing_cv_matrices,
        params_sets,
        model_type=model_type,
        check_overfitting=kwargs.get("check_overfitting", True),
    )
    print("\nExtraTrees hyperparameters tuning end.")

    # 3. Train the model with Extra Trees
    extra_trees_final_model = ExtraTreesClassifier(
        max_depth=best_params.get("max_depth"),
        max_features=best_params.get("max_features"),
        n_estimators=best_params.get("n_estimators"),
        random_state=kwargs.get("random_seed", 2731),
    )
    extra_trees_final_model.fit(training_X, training_y)

    model_analyses_binary_classification(
        extra_trees_final_model,
        training_X,
        training_y,
        testing_X,
        testing_y,
        class_labels=kwargs.get("class_labels", ["False", "True"]),
        classification_threshold=kwargs.get("classification_threshold", 0.5),
        run_model_performance_analysis=kwargs.get(
            "run_model_performance_analysis", True
        ),
        run_feature_importance_analysis=kwargs.get(
            "run_feature_importance_analysis", False
        ),
        run_roc_auc_analysis=kwargs.get("run_roc_auc_analysis", True),
        run_threshold_analysis=kwargs.get("run_threshold_analysis", False),
        show_scores=kwargs.get("show_scores", False),
    )

    return extra_trees_final_model


def binary_classification_random_forest(
    training_X,
    training_y,
    testing_X,
    testing_y,
    max_depth_max: int = 10,
    set_max_depth: int = None,
    set_max_features: float = None,
    set_max_samples: float = None,
    set_n_estimators: int = None,
    **kwargs,
):
    """hyperparameters tuning for RandomForest Model and present
    training results with best set of hyperparameters.
    The default setting will iterate through combinations of 4 hyper-
    parameters:
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
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        testing_X: n_samples by n_features matrix,
            training data from testing set.
        testing_y: 1d array-like,
            ground truth target values from testing set.
        max_depth_max:
            the maximum of the max_depth in BalancedRandomForest that
            hyperparameters tuning will iterate through.
        set_max_depth:
            set RandomForestClassifier hyperparameter [max_depth] to a
            single value.
        set_n_estimators:
            set RandomForestClassifier hyperparameter [n_estimators] to
            a single value.
        set_max_features:
            set RandomForestClassifier hyperparameter [max_features] to
            a single value.
        set_max_samples:
            set RandomForestClassifier hyperparameter [max_samples] to
            a single value.
        **kwargs:
            class_labels:
                keyword arguments for model_performance_analysis()
                in binary_classification_analyses()
            check_overfitting: bool
                keyword arguments for _load_best_hyperparameter()
            classification_threshold: float
                keyword arguments for binary_classification_analyses()
            random_seed: int
                keyword arguments for _cross_validator_random_forest()
                in _model_hyper_parameter_tuning() and
                RandomForestClassifier()
            run_model_performance_analysis: bool
                keyword arguments for binary_classification_analyses().
            run_feature_importance_analysis: bool
                keyword arguments for binary_classification_analyses().
            run_roc_auc_analysis: bool
                keyword arguments for binary_classification_analyses().
            run_threshold_analysis: bool
                keyword arguments for binary_classification_analyses().
            show_scores: bool
                keyword arguments for binary_classification_analyses().

    Return:
        fitted RandomForestClassifier
    """
    model_type = "RandomForest"

    # Check hyperparameters
    assert max_depth_max > 0, "[max_depth] should be positive integer."

    print("==================================================")

    # Setting hyperparameters
    params_sets = _load_hyperparameter_sets(
        {
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
            "n_estimators": (
                [set_n_estimators] if set_n_estimators else [50, 100, 150, 200]
            ),
        }
    )

    # Setting the training parameters
    n_fold = 5

    # Cross-validation
    training_cv_matrices, testing_cv_matrices = _model_hyper_parameter_tuning(
        training_X,
        training_y,
        params_sets=params_sets,
        model_type=model_type,
        n_fold=n_fold,
        random_seed=kwargs.get("random_seed", 2731),
    )
    # Load best parameters based on testing set's error
    best_params = _load_best_hyperparameter(
        training_cv_matrices,
        testing_cv_matrices,
        params_sets,
        model_type=model_type,
        check_overfitting=kwargs.get("check_overfitting", True),
    )
    print("\nRandomForest hyperparameters tuning end.")

    # Train the model with Balanced Random Forest
    random_forest_final_model = RandomForestClassifier(
        max_depth=best_params.get("max_depth"),
        max_samples=best_params.get("max_samples"),
        max_features=best_params.get("max_features"),
        n_estimators=best_params.get("n_estimators"),
        random_state=kwargs.get("random_seed", 2731),
    )
    random_forest_final_model.fit(training_X, training_y)

    model_analyses_binary_classification(
        random_forest_final_model,
        training_X,
        training_y,
        testing_X,
        testing_y,
        class_labels=kwargs.get("class_labels", ["False", "True"]),
        classification_threshold=kwargs.get("classification_threshold", 0.5),
        run_model_performance_analysis=kwargs.get(
            "run_model_performance_analysis", True
        ),
        run_feature_importance_analysis=kwargs.get(
            "run_feature_importance_analysis", False
        ),
        run_roc_auc_analysis=kwargs.get("run_roc_auc_analysis", True),
        run_threshold_analysis=kwargs.get("run_threshold_analysis", False),
        show_scores=kwargs.get("show_scores", False),
    )

    return random_forest_final_model


def binary_classification_xgboost(
    training_X,
    training_y,
    testing_X,
    testing_y,
    max_depth_max: int = 10,
    set_colsample_bytree: float = None,
    set_max_depth: int = None,
    set_sample_weights: List = None,
    set_scale_pos_weight: Optional[float] = None,
    set_subsample: float = None,
    use_default_api: bool = False,
    **kwargs,
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
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        testing_X: n_samples by n_features matrix,
            training data from testing set.
        testing_y: 1d array-like,
            ground truth target values from testing set.
        max_depth_max:
            the maximum of the max_depth in XGBoost that hyper-
            parameters tuning will iterate through.
        set_colsample_bytree:
            set XGBoost hyperparameter [colsample_bytree] to a single
            value.
        set_max_depth:
            set XGBoost hyperparameter [max_depth] to a single value.
        set_sample_weights: 1d array-like,
            weights of each sample if specified.
        set_scale_pos_weight:
            set XGBoost hyperparameter [scale_pos_weight] to a single
            value.
        set_subsample:
            set XGBoost hyperparameter [subsample] to a single value.
        use_default_api:
            if True, use xgboost api to train model, else use sklearn
            api to train the model
        **kwargs:
            class_labels:
                keyword arguments for model_performance_analysis()
                in model_analyses_binary_classification()
            check_overfitting: bool
                keyword arguments for _load_best_hyperparameter()
            classification_threshold: float
                keyword arguments for model_analyses_binary_classification()
            random_seed: int
                keyword arguments for _cross_validator_xgboost() in
                _model_hyper_parameter_tuning()
            run_model_performance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_feature_importance_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_roc_auc_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            run_threshold_analysis: bool
                keyword arguments for model_analyses_binary_classification().
            show_scores: bool
                keyword arguments for model_analyses_binary_classification().

    Return:
        fitted XGBClassifier
    """
    model_type = "XGBoost"

    # Check hyperparameters
    if set_scale_pos_weight:
        assert (
            set_scale_pos_weight >= 0
        ), "[scale_pos_weight] should be larger than or equal to 0."
        assert (
            set_sample_weights is None
        ), "Can only use [scale_pos_weight] or [weight] at a time."
    if set_sample_weights:
        assert (
            set_scale_pos_weight is None
        ), "Can only use [weight] or [scale_pos_weight] at a time."
        assert len(set_sample_weights) == len(training_X), (
            "The length of [weight] should be the same as the length of "
            "training data."
        )
    assert max_depth_max > 0, "[max_depth_max] should be positive integer."

    print("==================================================")

    # Setting data
    warnings.simplefilter(action="ignore", category=FutureWarning)
    training_dmatrix = xgb.DMatrix(data=training_X, label=training_y)
    if set_sample_weights:
        training_dmatrix.set_weight(set_sample_weights)
    testing_dmatrix = xgb.DMatrix(data=testing_X, label=testing_y)
    warnings.simplefilter(action="always", category=FutureWarning)

    # Setting hyperparameters
    params_sets = _load_hyperparameter_sets(
        {
            "colsample_bytree": (
                [set_colsample_bytree]
                if set_colsample_bytree
                else [1.0, 0.875, 0.75, 0.625, 0.5]
            ),
            "max_depth": (
                [set_max_depth]
                if set_max_depth
                else list(range(max_depth_max + 1))[-5:]
            ),
            "subsample": (
                [set_subsample] if set_subsample else [1.0, 0.875, 0.75, 0.625, 0.5]
            ),
        },
        base_parameters={
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "exact",
            "learning_rate": 0.1,
            "scale_pos_weight": set_scale_pos_weight,
        },
    )

    # Setting training parameters
    n_fold = 5
    num_boost = 200
    early_stop = 5

    # Cross-validation
    training_cv_matrices, testing_cv_matrices = _model_hyper_parameter_tuning(
        training_dmatrix,
        params_sets=params_sets,
        model_type=model_type,
        n_fold=n_fold,
        num_boost=num_boost,
        early_stop=early_stop,
        random_seed=kwargs.get("random_seed", 2731),
    )
    # Load best parameters based on testing set's error
    best_params = _load_best_hyperparameter(
        training_cv_matrices,
        testing_cv_matrices,
        params_sets,
        model_type=model_type,
        check_overfitting=kwargs.get("check_overfitting", True),
    )
    print("XGBoost hyperparameters tuning end.")

    if use_default_api:
        # Train the model with xgboost api
        xgb_final_model = xgb.train(
            params=best_params,
            dtrain=training_dmatrix,
            num_boost_round=num_boost,
            evals=[(testing_dmatrix, "Test")],
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )
        xgb_training_score = xgb_final_model.predict(
            training_dmatrix, ntree_limit=xgb_final_model.best_ntree_limit
        )
        xgb_testing_score = xgb_final_model.predict(
            testing_dmatrix, ntree_limit=xgb_final_model.best_ntree_limit
        )
        # use iteration_range = (0, xgb_final_model.best_iteration + 1)
        # instead of ntree_limit for xgboost >= 1.4
    else:
        # Train the model with sklearn api
        xgb_final_model = xgb.XGBClassifier(
            objective=best_params.get("objective"),
            eval_metric=best_params.get("eval_metric"),
            tree_method=best_params.get("tree_method"),
            learning_rate=best_params.get("learning_rate"),
            subsample=best_params.get("subsample"),
            max_depth=best_params.get("max_depth"),
            colsample_bytree=best_params.get("colsample_bytree"),
            scale_pos_weight=best_params.get("scale_pos_weight", None),
            use_label_encoder=False,
        )
        xgb_final_model.fit(
            training_X,
            training_y,
            eval_set=[(testing_X, testing_y)],
            early_stopping_rounds=early_stop,
            verbose=False,
        )
        xgb_training_score = xgb_final_model.predict_proba(training_X)[:, 1]
        xgb_testing_score = xgb_final_model.predict_proba(testing_X)[:, 1]

    # Training Results Analyses
    model_analyses_binary_classification(
        xgb_final_model,
        training_X,
        training_y,
        testing_X,
        testing_y,
        training_score=xgb_training_score,
        testing_score=xgb_testing_score,
        class_labels=kwargs.get("class_labels", ["False", "True"]),
        classification_threshold=kwargs.get("classification_threshold", 0.5),
        run_model_performance_analysis=kwargs.get(
            "run_model_performance_analysis", True
        ),
        run_feature_importance_analysis=kwargs.get(
            "run_feature_importance_analysis", False
        ),
        run_roc_auc_analysis=kwargs.get("run_roc_auc_analysis", True),
        run_threshold_analysis=kwargs.get("run_threshold_analysis", False),
        show_scores=kwargs.get("show_scores", False),
    )

    return xgb_final_model
