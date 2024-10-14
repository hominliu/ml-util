from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from colorama import Fore, Style
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tabulate import tabulate


def model_analyses_binary_classification(
    trained_model,
    training_X,
    training_y,
    testing_X,
    testing_y,
    training_score=None,
    testing_score=None,
    classification_threshold: float = 0.5,
    run_model_performance_analysis: bool = True,
    run_feature_importance_analysis: bool = False,
    run_roc_auc_analysis: bool = True,
    run_threshold_analysis: bool = False,
    show_scores: bool = False,
    **kwargs,
) -> None:
    """Conduct binary classification analyses on training and testing
    set based on model training results:
        shap_feature_importance: optional analysis, .
        plot_threshold_recall_precision: optional analysis, analysis of
            Recall and Precision against thresholds and plot Recall and
            Precision under different thresholds
        show_scores: optional analysis,

    Args:
        trained_model: model object (from scikit-learn, xgboost)
            trained model. Need to support method predict_proba
            if not providing scores
        training_X: n_samples by n_features matrix,
            training data from training set.
        training_y: 1d array-like,
            ground truth target values from training set.
        testing_X: n_samples by n_features matrix,
            training data from testing set.
        testing_y: 1d array-like,
            ground truth target values from testing set.
        training_score: 1d array-like,
            predictions for training set. If none is provided then
            function will generate one using method predict_proba
        testing_score: 1d array-like,
            predictions for testing set. If none is provided then
            function will generate one using method predict_proba
        classification_threshold:
            threshold for model to decide the class of the sample based on
            predicted score
        run_model_performance_analysis:
            if True, run analyses for Overall Accuracy, Recall, Precision
            and Confusion Matrix
        run_feature_importance_analysis:
            if True, run SHAP feature importance analysis and plot top
            15 important features.
        run_roc_auc_analysis:
            if True, run analyses for ROC and AUC and plot ROC curve.
        run_threshold_analysis:
            if True, run analysis of Recall and Precision against
            thresholds and plot Recall and Precision under different
            thresholds.
        show_scores:
            if True, plot predicted score for training set and testing set.
        **kwargs:
            class_labels: List
                kwargs for model_performance_analysis().
    Return:
        None
    """
    if training_score is None:
        training_score = trained_model.predict_proba(training_X)[:, 1]
    if testing_score is None:
        testing_score = trained_model.predict_proba(testing_X)[:, 1]

    assert (
        0 < classification_threshold < 1
    ), "Classification threshold should be between 0 and 1."

    if run_model_performance_analysis:
        model_performance_analysis(
            training_y,
            np.where(training_score >= classification_threshold, 1, 0),
            testing_y,
            np.where(testing_score >= classification_threshold, 1, 0),
            class_labels=kwargs.get("class_labels", ["False", "True"]),
        )

    if run_roc_auc_analysis:
        plot_roc_curve(training_y, training_score, testing_y, testing_score)

    if run_feature_importance_analysis:
        shap_feature_importance(
            trained_model,
            training_X,
            num_features_display=15,
            plot_title="Training Set",
        )
        shap_feature_importance(
            trained_model, testing_X, num_features_display=15, plot_title="Testing Set"
        )

    if run_threshold_analysis:
        plot_threshold_recall_precision(
            training_y, training_score, testing_y, testing_score
        )

    if show_scores:
        plot_scores(training_y, training_score, testing_y, testing_score)


def model_performance_analysis(
    training_y_true,
    training_y_pred,
    testing_y_true,
    testing_y_pred,
    class_labels: List = None,
) -> None:
    """Conduct recall, precision, and confusion matrix analysis for
    both training and testing set based on prediction results.
    Works for both binary and multi-class classifications.

    Args:
        training_y_true: 1d array-like,
            ground truth target values from training set.
        training_y_pred: 1d array-like,
            predicted class for training set.
        testing_y_true: 1d array-like,
            ground truth target values from testing set.
        testing_y_pred: 1d array-like,
            predicted class for testing set.
        class_labels:
            list of labels for target values
    Return:
        None
    """

    if not class_labels:
        class_labels = map(str, list(range(len(np.unique(training_y_true)))))

    assert len(training_y_true) > 0, "No ground truth target values for training set."
    assert len(testing_y_true) > 0, "No ground truth target values for testing set."
    assert len(training_y_pred) > 0, "No predicted probabilities for training set."
    assert len(testing_y_pred) > 0, "No predicted probabilities for testing set."
    assert len(training_y_pred) == len(training_y_true), (
        "length of 'training_y_pred' should be the same as the length of "
        "'training_y_true'."
    )
    assert len(testing_y_pred) == len(testing_y_true), (
        "length of 'testing_y_pred' should be the same as the length of "
        "'testing_y_true'."
    )
    assert len(np.unique(class_labels)) == len(np.unique(training_y_true)), (
        "length of 'class_labels' does not match the number of target "
        "classes in data provided."
    )

    # Training set
    confusion_matrix_training = confusion_matrix(training_y_true, training_y_pred)
    recall_training = recall_score(
        training_y_true, training_y_pred, average=None, zero_division=0
    )
    precision_training = precision_score(
        training_y_true, training_y_pred, average=None, zero_division=0
    )

    # Testing set
    confusion_matrix_testing = confusion_matrix(testing_y_true, testing_y_pred)
    recall_testing = recall_score(
        testing_y_true, testing_y_pred, average=None, zero_division=0
    )
    precision_testing = precision_score(
        testing_y_true, testing_y_pred, average=None, zero_division=0
    )

    # Construct table
    matrices_table_training = []
    matrices_table_testing = []
    confusion_matrix_table_training = []
    confusion_matrix_table_testing = []
    confusion_matrix_table_header = []
    for idx, class_label in enumerate(class_labels):
        matrices_table_training.append(
            [
                class_label,
                str(round(recall_training[idx] * 100, 2)) + "%",
                str(round(precision_training[idx] * 100, 2)) + "%",
            ]
        )
        matrices_table_testing.append(
            [
                class_label,
                str(round(recall_testing[idx] * 100, 2)) + "%",
                str(round(precision_testing[idx] * 100, 2)) + "%",
            ]
        )
        confusion_matrix_table_training.append(
            [class_label, *confusion_matrix_training[idx]]
        )
        confusion_matrix_table_testing.append(
            [class_label, *confusion_matrix_testing[idx]]
        )
        confusion_matrix_table_header.append("".join(["Predicted\n", class_label]))

    print("==================================================")
    print("".join([Fore.RED, "PERFORMANCE MATRIX:", Style.RESET_ALL]))
    print(
        "".join(
            [
                Fore.BLUE,
                "Training",
                Style.RESET_ALL,
                " set's Overall accuracy is : ",
                str(round(accuracy_score(training_y_true, training_y_pred) * 100, 2)),
                "%",
            ]
        )
    )
    print(
        tabulate(
            matrices_table_training,
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
                str(round(accuracy_score(testing_y_true, testing_y_pred) * 100, 2)),
                "%",
            ]
        )
    )
    print(
        tabulate(
            matrices_table_testing,
            headers=["", "Recall", "Precision"],
            tablefmt="pretty",
        )
    )

    print("\n\n==================================================")
    print("".join([Fore.RED, "CONFUSION MATRIX", Style.RESET_ALL]))
    print(
        tabulate(
            confusion_matrix_table_training,
            headers=[
                Fore.BLUE + "Training" + Style.RESET_ALL,
                *confusion_matrix_table_header,
            ],
            tablefmt="pretty",
        )
    )
    print("")
    print(
        tabulate(
            confusion_matrix_table_testing,
            headers=[
                Fore.GREEN + "Testing" + Style.RESET_ALL,
                *confusion_matrix_table_header,
            ],
            tablefmt="pretty",
        )
    )


def plot_roc_curve(training_y_true, training_y_score, testing_y_true, testing_y_score):
    """Plot ROC curve and calculate AUC score

    Args:
        training_y_true: 1d array-like,
            ground truth target values from training set.
        training_y_score: 1d array-like,
            predicted probabilities for training set.
        testing_y_true: 1d array-like,
            ground truth target values from testing set.
        testing_y_score: 1d array-like,
            predicted probabilities for testing set.
    """

    plt.clf()
    plt.close()
    plt.rcParams["figure.figsize"] = [5, 5]
    # train data
    roc_auc_train = roc_auc_score(training_y_true, training_y_score)
    fpr_train, tpr_train, thresholds_train = roc_curve(
        training_y_true, training_y_score
    )
    # test data
    roc_auc_test = roc_auc_score(testing_y_true, testing_y_score)
    fpr_test, tpr_test, thresholds_test = roc_curve(testing_y_true, testing_y_score)

    # plotting
    plt.figure()
    plt.plot(fpr_train, tpr_train, label="Train set (area = %0.4f)" % roc_auc_train)
    plt.plot(fpr_test, tpr_test, label="Test set (area = %0.4f)" % roc_auc_test)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


def plot_scores(training_y_true, training_y_score, testing_y_true, testing_y_score):
    """Plot predicted scores (probabilities)

    Args:
        training_y_true: 1d array-like,
            ground truth target values from training set.
        training_y_score: 1d array-like,
            predicted probabilities for training set.
        testing_y_true: 1d array-like,
            ground truth target values from testing set.
        testing_y_score: 1d array-like,
            predicted probabilities for testing set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    sns.kdeplot(
        data=pd.DataFrame(
            {"label": training_y_true, "Predicted Probability": training_y_score}
        ),
        x="Predicted Probability",
        hue="label",
        linewidth=3,
        ax=axes[0],
    ).set(title="Training Predicted Probabilities")
    sns.kdeplot(
        data=pd.DataFrame(
            {"label": testing_y_true, "Predicted Probability": testing_y_score}
        ),
        x="Predicted Probability",
        hue="label",
        linewidth=3,
        ax=axes[1],
    ).set(title="Testing Predicted Probabilities")
    plt.show()
    plt.close()


def plot_threshold_recall_precision(
    training_y_true, training_y_score, testing_y_true, testing_y_score
):
    """Plot recall scores and precision scores under different thresholds

    Args:
        training_y_true: 1d array-like,
            ground truth target values from training set.
        training_y_score: 1d array-like,
            predicted probabilities for training set.
        testing_y_true: 1d array-like,
            ground truth target values from testing set.
        testing_y_score: 1d array-like,
            predicted probabilities for testing set.
    """
    plt.figure(figsize=(5, 5))

    (
        training_thresholds,
        training_recall,
        training_precision,
    ) = thresholds_recall_precision(training_y_true, training_y_score)
    testing_thresholds, testing_recall, testing_precision = thresholds_recall_precision(
        testing_y_true, testing_y_score
    )

    plt.plot(
        training_thresholds,
        training_recall,
        label="Training Recall",
        linestyle=":",
        color="tab:blue",
    )
    plt.plot(
        training_thresholds,
        training_precision,
        label="Training Precision",
        linestyle=":",
        color="tab:orange",
    )
    plt.plot(
        testing_thresholds,
        testing_recall,
        label="Testing Recall",
        linestyle="-",
        color="tab:blue",
    )
    plt.plot(
        testing_thresholds,
        testing_precision,
        label="Testing Precision",
        linestyle="-",
        color="tab:orange",
    )
    plt.title("Recall/Precision vs. Thresholds")
    plt.xlabel("Thresholds")
    plt.ylabel("Recall/Precision")
    plt.grid(linestyle="--")
    plt.legend()
    plt.show()
    plt.close()


def shap_feature_importance(
    fitted_estimator,
    transformed_data,
    feature_names: List = None,
    num_features_display: int = 20,
    plot_type: str = "dot",
    plot_title: str = "SHAP Feature Importance",
) -> np.ndarray:
    """Plot feature importance from shap
    Args:
        fitted_estimator:
            fitted estimator class (sklearn api like)
        transformed_data:
            data that could be predicted by fitted_estimator
        feature_names:
            list of feature names
        num_features_display:
            number of top features to display
        plot_type:
            type of the plot from shap.summary_plot
        plot_title:
            title for the plot
    Return:
         shap_importance (numpy.ndarray): importance value calculated based
         on shap
    """

    # TODO: check estimator type then assign different type of explainer
    explainer = shap.TreeExplainer(fitted_estimator)
    shap_values = explainer.shap_values(transformed_data)
    plt.figure()
    if fitted_estimator.n_classes_ > 2:
        shap_importance = np.array([0] * len(feature_names))
        for shap_value in shap_values:
            shap_importance = shap_importance + np.abs(shap_value).mean(0)
        shap.summary_plot(
            shap_values,
            transformed_data,
            feature_names=feature_names,
            max_display=num_features_display,
            plot_type="bar",
            class_names=fitted_estimator.classes_,
            show=False,
            plot_size=(5, 5),
        )
        fig = plt.gcf()
        fig.set_figheight(12)
        fig.set_figwidth(14)
        ax = plt.gca()
        ax.set_ylabel("Features", fontsize=16)
        ax.legend(loc="lower right", ncol=2)
    else:
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_importance = np.abs(shap_values).mean(0)
        shap.summary_plot(
            shap_values,
            transformed_data,
            feature_names=feature_names,
            max_display=num_features_display,
            plot_type=plot_type,
            show=False,
            plot_size=(5, 5),
        )
    plt.title("".join([plot_title, f" - Top {num_features_display} Features"]))
    plt.show()
    plt.close()

    return shap_importance


def thresholds_recall_precision(y_true, y_score) -> Tuple[List, List, List]:
    """Load recall and precision under different thresholds (from 0 to 1 with
    0.01 increment)
    Args:
        y_true: 1d array-like,
            ground truth target values from training set.
        y_score: 1d array-like,
            predicted probabilities for training set.
    Return:
        thresholds:
            thresholds from 0 to 1 with 0.01 increment
        recalls:
            recall scores under thresholds listed above
        precisions:
            precision scores under thresholds listed above
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
