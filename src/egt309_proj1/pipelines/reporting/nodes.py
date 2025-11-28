import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession


# This function uses plotly.express
def compare_subscription_by_occupation(load_raw_data: SparkDataFrame):
    spark = SparkSession.builder.appName("SubscriptionbyOccupation").getOrCreate()
    # Register the DataFrame as a temporary table
    load_raw_data.createOrReplaceTempView("bank_marketing")
    # Perform the grouping and aggregation using SQL
    query = """
            SELECT Occupation, AVG(CASE 
            WHEN Subscription Status = 'yes' THEN 1.0 
            ELSE 0.0 
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
        SELECT Contact Method AS Contact_Method, AVG(CASE 
        WHEN Subscription Status = 'yes' THEN 1.0 
        ELSE 0.0 
        END) AS Subscription_Rate
        FROM bank_marketing
        GROUP BY Contact Method
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

# This function creates a confusion matrix from model predictions
def create_confusion_matrix(bank_marketing: pd.DataFrame):
    matplotlib.use('Agg')

    df = bank_marketing.copy()
    actual = df["y_test"]
    predicted = df["y_pred"]

    confusion_matrix = pd.crosstab(
        actual,
        predicted, 
        rownames=["Actual"], 
        colnames=["Predicted"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix based on Model Predictions')
    plt.tight_layout()

    return fig

# This function plots model performance metrics for the 3 models
def plot_model_metrics(results: dict):
    
    model_metrics=pd.DataFrame(results).T
    model_metrics.rename(columns={"model": "model_name"}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    model_metrics.set_index("model_name")[["accuracy", "precision", "recall", "f1_score"]].plot(kind='bar', ax=ax)

    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    return fig

# This function plots feature importance for the 3 models
def plot_feature_importance_for_MLModels(rf_model, gb_model, lr_model, feature_names: list):
    matplotlib.use('Agg')

    fig, ax = plt.subplots(1, 3, figsize=(14, 18))

    # Random Forest Feature Importance
    rf_importances = rf_model.feature_importances_
    rf_indices = rf_importances.argsort()[::-1]

    ax[0].barh([feature_names[i] for i in rf_indices], rf_importances[rf_indices], align='center')
    ax[0].invert_yaxis()
    ax[0].set_title('Random Forest Feature Importance')
    ax[0].set_xlabel('Importance Score')

    # Gradient Boosting Feature Importance
    gb_importances = gb_model.feature_importances_
    gb_indices = gb_importances.argsort()[::-1]

    ax[1].barh([feature_names[i] for i in gb_indices], gb_importances[gb_indices], align='center')
    ax[1].invert_yaxis()
    ax[1].set_title('Gradient Boosting Feature Importance')
    ax[1].set_xlabel('Importance Score')

    # Logistic Regression Feature Importance
    lr_importances = abs(lr_model.coef_[0])
    lr_indices = lr_importances.argsort()[::-1]

    ax[2].barh([feature_names[i] for i in lr_indices], lr_importances[lr_indices], align='center')
    ax[2].invert_yaxis()
    ax[2].set_title('Logistic Regression Feature Importance')
    ax[2].set_xlabel('Coeefficient Magnitude')

    plt.tight_layout()
    
    return fig