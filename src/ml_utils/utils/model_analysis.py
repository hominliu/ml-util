from typing import Tuple

import numpy as np
import shap
from colorama import Fore, Style
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from tabulate import tabulate


def model_performance_analysis(
    train_y_true,
    train_y_pred,
    test_y_true,
    test_y_pred,
    class_labels: list = None,
) -> None:
    """Conduct recall, precision, and confusion matrix analysis for both training and
    testing set based on prediction results.
    Works for both binary and multi-class classifications.

    Args:
        train_y_true: 1d array-like, ground truth target values from training set.
        train_y_pred: 1d array-like, predicted class for training set.
        test_y_true: 1d array-like, ground truth target values from testing set.
        test_y_pred: 1d array-like, predicted class for testing set.
        class_labels: list of labels for target values
    Return:
        None
    """

    if not class_labels:
        class_labels = map(str, list(range(len(np.unique(train_y_true)))))

    assert len(train_y_true) > 0, "No ground truth target values for training set."
    assert len(test_y_true) > 0, "No ground truth target values for testing set."
    assert len(train_y_pred) > 0, "No predicted probabilities for training set."
    assert len(test_y_pred) > 0, "No predicted probabilities for testing set."
    assert len(train_y_pred) == len(
        train_y_true
    ), "length of 'train_y_pred' is not the same as the length of 'train_y_true'."
    assert len(test_y_pred) == len(
        test_y_true
    ), "length of 'test_y_pred' is not the same as the length of 'test_y_true'."
    assert len(np.unique(class_labels)) == len(np.unique(train_y_true)), (
        "length of 'class_labels' does not match the number of target "
        "classes in data provided."
    )

    # Training set
    train_cm = confusion_matrix(train_y_true, train_y_pred)
    train_recall = recall_score(
        train_y_true, train_y_pred, average=None, zero_division=0
    )
    train_precision = precision_score(
        train_y_true, train_y_pred, average=None, zero_division=0
    )

    # Testing set
    test_cm = confusion_matrix(test_y_true, test_y_pred)
    test_recall = recall_score(test_y_true, test_y_pred, average=None, zero_division=0)
    test_precision = precision_score(
        test_y_true, test_y_pred, average=None, zero_division=0
    )

    # Construct table
    train_matrices_table = []
    test_matrices_table = []
    train_cm_table = []
    test_cm_table = []
    cm_table_header = []
    for idx, class_label in enumerate(class_labels):
        train_matrices_table.append(
            [
                class_label,
                str(round(train_recall[idx] * 100, 2)) + "%",
                str(round(train_precision[idx] * 100, 2)) + "%",
            ]
        )
        test_matrices_table.append(
            [
                class_label,
                str(round(test_recall[idx] * 100, 2)) + "%",
                str(round(test_precision[idx] * 100, 2)) + "%",
            ]
        )
        train_cm_table.append([class_label, *train_cm[idx]])
        test_cm_table.append([class_label, *test_cm[idx]])
        cm_table_header.append("".join(["Predicted\n", class_label]))

    print("==================================================")
    print("".join([Fore.RED, "PERFORMANCE MATRIX:", Style.RESET_ALL]))
    print(
        "".join(
            [
                Fore.BLUE,
                "Training",
                Style.RESET_ALL,
                " set's Overall accuracy is : ",
                str(round(accuracy_score(train_y_true, train_y_pred) * 100, 2)),
                "%",
            ]
        )
    )
    print(
        tabulate(
            train_matrices_table,
            headers=["", "Recall", "Precision"],
            tablefmt="pretty",
        )
    )
    print("\n")
    print(
        "".join(
            [
                Fore.GREEN,
                "Testing",
                Style.RESET_ALL,
                " set's Overall accuracy is : ",
                str(round(accuracy_score(test_y_true, test_y_pred) * 100, 2)),
                "%",
            ]
        )
    )
    print(
        tabulate(
            test_matrices_table,
            headers=["", "Recall", "Precision"],
            tablefmt="pretty",
        )
    )

    print("\n\n==================================================")
    print("".join([Fore.RED, "CONFUSION MATRIX", Style.RESET_ALL]))
    print(
        tabulate(
            train_cm_table,
            headers=[Fore.BLUE + "Training" + Style.RESET_ALL, *cm_table_header],
            tablefmt="pretty",
        )
    )
    print("")
    print(
        tabulate(
            test_cm_table,
            headers=[Fore.GREEN + "Testing" + Style.RESET_ALL, *cm_table_header],
            tablefmt="pretty",
        )
    )


def shap_feature_importance(
    fitted_estimator,
    X,
    feature_names: list = None,
) -> np.ndarray:
    """Calculate feature importance from shap

    Args:
        fitted_estimator: fitted estimator class (sklearn api like)
        X: data that could be predicted by fitted_estimator
        feature_names: list of feature names
    Return:
         shap_importance (numpy.ndarray): importance value calculated based
         on shap
         shap_values (numpy.ndarray): shap values
    """

    # TODO: check estimator type then assign different type of explainer
    explainer = shap.TreeExplainer(fitted_estimator)
    shap_values = explainer.shap_values(X)

    if fitted_estimator.n_classes_ > 2:
        shap_importance = np.array([0] * len(feature_names))
        for shap_value in shap_values:
            shap_importance = shap_importance + np.abs(shap_value).mean(0)
    else:
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_importance = np.abs(shap_values).mean(0)

    return shap_importance, shap_values


def thresholds_recall_precision(y_true, y_score) -> Tuple[list, list, list]:
    """Load recall and precision under different thresholds (from 0 to 1 with
    0.01 increment)

    Args:
        y_true: 1d array-like, ground truth target values from training set.
        y_score: 1d array-like, predicted probabilities for training set.
    Return:
        thresholds: thresholds from 0 to 1 with 0.01 increment
        recalls: recall scores under thresholds listed above
        precisions: precision scores under thresholds listed above
    """

    thresholds = []
    recalls = []
    precisions = []

    for idx in np.linspace(0, 1, 101):
        threshold = round(idx, 2)
        y_pred = np.where(y_score >= threshold, 1, 0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        if recall > 0 or precision > 0:
            thresholds.append(threshold)
            recalls.append(recall)
            precisions.append(precision)

    return thresholds, recalls, precisions
