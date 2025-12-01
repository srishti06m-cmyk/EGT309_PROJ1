import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from sklearn.metrics import roc_curve, auc


# This function uses plotly.express
def compare_subscription_by_occupation(load_raw_data: SparkDataFrame):
    spark = SparkSession.builder.appName("SubscriptionbyOccupation").getOrCreate()
    # Register the DataFrame as a temporary table
    load_raw_data.createOrReplaceTempView("bank_marketing")
    # Perform the grouping and aggregation using SQL
    query = """
            SELECT Occupation, AVG(CASE 
            WHEN `Subscription Status` = 'yes' THEN 1 
            ELSE 0 
            END) AS Subscription_Rate
            FROM bank_marketing
            GROUP BY Occupation
            ORDER BY Subscription_Rate DESC
        """
    grouped_data = spark.sql(query)
    # Convert Spark DataFrame to Pandas for visualization
    pandas_grouped_data = grouped_data.toPandas()
    return pandas_grouped_data

# This function uses plotly.graph_objects
def compare_subscription_by_contact_method(load_raw_data: SparkDataFrame):
    spark = SparkSession.builder.appName("SubscriptionbyContactMethod").getOrCreate()
    # Register the DataFrame as a temporary table
    load_raw_data.createOrReplaceTempView("bank_marketing")
    # Perform the grouping and aggregation using SQL
    query = """
        SELECT `Contact Method` AS Contact_Method, AVG(CASE 
        WHEN Subscription Status = 'yes' THEN 1 
        ELSE 0 
        END) AS Subscription_Rate
        FROM bank_marketing
        GROUP BY `Contact Method`
        ORDER BY Subscription_Rate DESC
    """
    grouped_data = spark.sql(query)
    # Convert Spark DataFrame to Pandas for visualization
    pandas_grouped_data = grouped_data.toPandas()
    # Create the Plotly figure
    fig = go.Figure(
        [
            go.Bar(
                x=pandas_grouped_data["Contact_Method"],
                y=pandas_grouped_data["Subscription_Rate"],
            )
        ]
    )
    fig.update_layout(
        title="Subscription Rate by Contact Method",
        xaxis_title="Contact Method",
        yaxis_title="Subscription Rate",
    )
    return fig

# This function creates a confusion matrix for the best model
def create_confusion_matrix(y_test: pd.Series, y_pred: pd.Series):
    matplotlib.use('Agg')

    cm = pd.crosstab(
        y_test, y_pred, 
        rownames=['Actual'], colnames=['Predicted'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    return fig

# This function plots model performance metrics for the 6 models
def plot_model_metrics(metrics: dict[str, dict]):
    """
    metrics dict example:
    { "log_reg": {"accuracy":..., "precision":..., "recall":..., "f1_score": ..., "roc_auc":..., "best_threshold":...},
    """
    
    matplotlib.use('Agg')

    df = pd.DataFrame(metrics).T
    df.rename(index={
        "log_reg": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost"
        }, inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))

    df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind='bar', ax=ax)

    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    return fig

# This function plots feature importance for the 6 models
def plot_feature_importance_for_MLModels(trained_models: dict[str, object]):
    matplotlib.use('Agg')

    first_model = next(iter(trained_models.values()))
    preproccessor = first_model.named_steps['preprocessor']
    feature_names = preproccessor.get_feature_names_out()

    model_order = [
        "random_forest","gradient_boosting","log_reg",
        "xgboost","lightgbm","catboost"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes= axes.flatten()

    title_map = {
        "random_forest": "Random Forest Feature Importance",
        "gradient_boosting": "Gradient Boosting Feature Importance",
        "log_reg": "Logistic Regression Feature Importance",
        "xgboost": "XGBoost Feature Importance",
        "lightgbm": "LightGBM Feature Importance",
        "catboost": "CatBoost Feature Importance"
    }
    for i, name in enumerate(model_order):
        ax = axes[i]
        model = trained_models.get(name)

        if model is None:
            ax.axis('off')
            continue

        clf = model.named_steps['clf']

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = abs(clf.coef_).flatten()
        else:
            ax.axis('off')
            continue

        idx=np.argsort(importances)[::-1][:20]
        ax.barh([feature_names[i] for i in idx], importances[idx])
        ax.invert_yaxis()
        ax.set_title(title_map[name])
        ax.set_xlabel("Importance Score")
    
    plt.tight_layout()
    
    return fig

# This function plots ROC curves for the 6 models
def plot_roc_curve_for_MLModels(trained_models: dict[str, object], X_test, y_test):
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(8, 6))

    name_map = {
    "log_reg": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost"
    }
    
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label = f"{name_map[name]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curve for All ML Models')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

    plt.tight_layout()

    return fig