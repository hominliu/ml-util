import numpy as np

from ..utils.model_analysis import model_performance_analysis
from ..utils.plotting import (
    plot_roc_curve,
    plot_scores,
    plot_shap_feature_importance,
    plot_threshold_recall_precision,
)


def binary_model_analyses(
    trained_model,
    train_X,
    train_y,
    train_score,
    test_X,
    test_y,
    test_score,
    classification_threshold: float = 0.5,
    class_labels: list = None,
    analyze_model_performance: bool = True,
    plot_shap: bool = False,
    plot_roc_auc: bool = True,
    plot_thresholds: bool = False,
    plot_proba: bool = False,
):
    """Conduct binary classification analyses on training and testing

    Args:
        trained_model: model object (from scikit-learn, xgboost) trained model.
        train_X: n_samples by n_features matrix, training data from training set.
        train_y: 1d array-like, ground truth target values from training set.
        train_score: 1d array-like, predicted probabilities  for training set.
        test_X: n_samples by n_features matrix, training data from testing set.
        test_y: 1d array-like, ground truth target values from testing set.
        test_score: 1d array-like, predicted probabilities for testing set.
        classification_threshold: threshold for model to decide the class of the
            sample based on predicted score
        class_labels: list of labels for target values
        analyze_model_performance: run analyses for Overall Accuracy, Recall,
            Precision and Confusion Matrix if True
        plot_shap: run SHAP feature importance analysis and plot top 15 important
            features if True
        plot_roc_auc: run analyses for ROC and AUC and plot ROC curve if True
        plot_thresholds: run analysis of Recall and Precision against thresholds
            and plot Recall and Precision under different thresholds if True
        plot_proba: plot predicted score for training set and testing set if True
    """
    assert (
        0 < classification_threshold < 1
    ), "Classification threshold should be between 0 and 1."

    if analyze_model_performance:
        model_performance_analysis(
            train_y,
            np.where(train_score >= classification_threshold, 1, 0),
            test_y,
            np.where(test_score >= classification_threshold, 1, 0),
            class_labels=class_labels,
        )

    if plot_shap:
        plot_shap_feature_importance(
            trained_model,
            train_X,
            num_features_display=15,
            plot_title="Training Set",
        )
        plot_shap_feature_importance(
            trained_model, test_X, num_features_display=15, plot_title="Testing Set"
        )

    if plot_roc_auc:
        plot_roc_curve(train_y, train_score, test_y, test_score)

    if plot_thresholds:
        plot_threshold_recall_precision(train_y, train_score, test_y, test_score)

    if plot_proba:
        plot_scores(train_y, train_score, test_y, test_score)
