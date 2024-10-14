import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score, roc_curve

from .model_analysis import shap_feature_importance, thresholds_recall_precision


def plot_roc_curve(train_y_true, train_y_score, test_y_true, test_y_score):
    """Plot ROC curve and calculate AUC score

    Args:
        train_y_true: 1d array-like, ground truth target values from training set.
        train_y_score: 1d array-like, predicted probabilities for training set.
        test_y_true: 1d array-like, ground truth target values from testing set.
        test_y_score: 1d array-like, predicted probabilities for testing set.
    """

    plt.clf()
    plt.close()
    plt.rcParams["figure.figsize"] = [5, 5]
    # train data
    roc_auc_train = roc_auc_score(train_y_true, train_y_score)
    fpr_train, tpr_train, thresholds_train = roc_curve(train_y_true, train_y_score)
    # test data
    roc_auc_test = roc_auc_score(test_y_true, test_y_score)
    fpr_test, tpr_test, thresholds_test = roc_curve(test_y_true, test_y_score)

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


def plot_scores(train_y_true, train_y_score, test_y_true, test_y_score):
    """Plot predicted scores (probabilities)

    Args:
        train_y_true: 1d array-like, ground truth target values from training set.
        train_y_score: 1d array-like, predicted probabilities for training set.
        test_y_true: 1d array-like, ground truth target values from testing set.
        test_y_score: 1d array-like, predicted probabilities for testing set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    sns.kdeplot(
        data=pd.DataFrame(
            {"label": train_y_true, "Predicted Probability": train_y_score}
        ),
        x="Predicted Probability",
        hue="label",
        linewidth=3,
        ax=axes[0],
    ).set(title="Training Predicted Probabilities")
    sns.kdeplot(
        data=pd.DataFrame(
            {"label": test_y_true, "Predicted Probability": test_y_score}
        ),
        x="Predicted Probability",
        hue="label",
        linewidth=3,
        ax=axes[1],
    ).set(title="Testing Predicted Probabilities")
    plt.show()
    plt.close()


def plot_shap_feature_importance(
    fitted_estimator,
    X,
    feature_names: list = None,
    num_features_display: int = 20,
    plot_type: str = "dot",
    plot_title: str = "SHAP Feature Importance",
):
    """Plot feature importance from shap

    Args:
        fitted_estimator: fitted estimator class (sklearn api like)
        X: data that could be predicted by fitted_estimator
        feature_names: list of feature names
        num_features_display: number of top features to display
        plot_type: type of the plot from shap.summary_plot
        plot_title: title for the plot
    """
    # TODO: check estimator type then assign different type of explainer
    shap_importance, shap_values = shap_feature_importance(
        fitted_estimator, X, feature_names
    )

    plt.figure()
    if fitted_estimator.n_classes_ > 2:
        shap.summary_plot(
            shap_values,
            X,
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
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=num_features_display,
            plot_type=plot_type,
            show=False,
            plot_size=(5, 5),
        )
    plt.title("".join([plot_title, f" - Top {num_features_display} Features"]))
    plt.show()
    plt.close()


def plot_threshold_recall_precision(
    train_y_true, train_y_score, test_y_true, test_y_score
):
    """Plot recall scores and precision scores under different thresholds

    Args:
        train_y_true: 1d array-like,
            ground truth target values from training set.
        train_y_score: 1d array-like,
            predicted probabilities for training set.
        test_y_true: 1d array-like,
            ground truth target values from testing set.
        test_y_score: 1d array-like,
            predicted probabilities for testing set.
    """
    plt.figure(figsize=(5, 5))

    train_thresholds, train_recall, train_precision = thresholds_recall_precision(
        train_y_true, train_y_score
    )
    test_thresholds, test_recall, test_precision = thresholds_recall_precision(
        test_y_true, test_y_score
    )

    plt.plot(
        train_thresholds,
        train_recall,
        label="Training Recall",
        linestyle=":",
        color="tab:blue",
    )
    plt.plot(
        train_thresholds,
        train_precision,
        label="Training Precision",
        linestyle=":",
        color="tab:orange",
    )
    plt.plot(
        test_thresholds,
        test_recall,
        label="Testing Recall",
        linestyle="-",
        color="tab:blue",
    )
    plt.plot(
        test_thresholds,
        test_precision,
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
