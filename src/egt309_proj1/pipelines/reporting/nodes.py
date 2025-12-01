import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix



def compare_subscription_by_occupation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute subscription rate by job type using the final model_input_table.

    Assumes columns:
      - 'Job_Type' (or your actual job feature name)
      - 'Subscription Status' with values 'yes' / 'no'
    """
    grouped = (
        df.groupby("Job_Type")["Subscription Status"]   # <-- changed from "Occupation"
          .apply(lambda s: (s == "yes").mean())
          .reset_index(name="Subscription_Rate")
          .sort_values("Subscription_Rate", ascending=False)
    )
    return grouped



def compare_subscription_by_contact_method(df: pd.DataFrame):
    grouped = (
        df.groupby("Contact Method")["Subscription Status"]
          .apply(lambda s: (s == "yes").mean())
          .reset_index(name="Subscription_Rate")
          .sort_values("Subscription_Rate", ascending=False)
    )

    fig = go.Figure(
        [
            go.Bar(
                x=grouped["Contact Method"],
                y=grouped["Subscription_Rate"],
            )
        ]
    )
    fig.update_layout(
        title="Subscription Rate by Contact Method",
        xaxis_title="Contact Method",
        yaxis_title="Subscription Rate",
    )
    return fig


def create_confusion_matrix(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    matplotlib.use("Agg")

    # Get predictions from the trained best model
    y_pred = best_model.predict(X_test)

    # Build confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix based on Best Model Predictions")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    return fig


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_metrics(results: dict):
    """
    Plot model performance metrics from evaluation_metrics.json.

    Expected structure of `results`:
        {
          "log_reg": {
              "accuracy": ...,
              "f1": ...,
              "roc_auc": ...,
              "best_threshold": ...
          },
          "random_forest": { ... },
          ...
        }
    """
    matplotlib.use("Agg")

    # Convert dict to DataFrame: index = model name, columns = metrics
    model_metrics = pd.DataFrame(results).T

    # Move index into a proper column for plotting labels
    model_metrics = model_metrics.reset_index().rename(columns={"index": "model_name"})

    # Keep only the metrics we actually have
    metrics_to_plot = ["accuracy", "f1", "roc_auc"]
    available = [m for m in metrics_to_plot if m in model_metrics.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    model_metrics.set_index("model_name")[available].plot(kind="bar", ax=ax)

    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    return fig

def plot_feature_importance_for_MLModels(trained_models: dict):
    """
    Plot feature importance for tree-based models stored in trained_models dict.
    Assumes each value is a Pipeline with steps: 'preprocessor' and 'clf'.
    """
    matplotlib.use("Agg")

    # Use one model to get feature names from the preprocessor
    any_model = trained_models["random_forest"]  # or any that exists
    preprocessor = any_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    # Pick models that have feature_importances_ (tree-based)
    model_names = ["random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"]
    display_models = [m for m in model_names if m in trained_models]

    fig, axes = plt.subplots(1, len(display_models), figsize=(5 * len(display_models), 6))

    if len(display_models) == 1:
        axes = [axes]

    for ax, name in zip(axes, display_models):
        pipe = trained_models[name]
        clf = pipe.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            continue

        idx = importances.argsort()[::-1]
        ax.barh([feature_names[i] for i in idx], importances[idx])
        ax.invert_yaxis()
        ax.set_title(f"{name} Feature Importance")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    return fig

# This function plots ROC curves for the 6 models
def plot_roc_curve_for_MLModels(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipe in trained_models.items():
        clf = pipe.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            continue

        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve for ML Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig
