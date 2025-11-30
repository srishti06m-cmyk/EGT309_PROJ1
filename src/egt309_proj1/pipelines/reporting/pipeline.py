from kedro.pipeline import Node, Pipeline

from .nodes import (
    compare_subscription_by_occupation,
    compare_subscription_by_contact_method,
    create_confusion_matrix,
    plot_model_metrics,
    plot_feature_importance_for_MLModels,
    plot_roc_curve_for_MLModels,
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return Pipeline(
        [
            Node(
                func=compare_subscription_by_occupation,
                inputs="load_raw_data",
                outputs="Comparison_Subscription_by_Occupation_plot",
            ),
            Node(
                func=compare_subscription_by_contact_method,
                inputs="load_raw_data",
                outputs="Comparison_Subscription_by_Contact_Method_plot",
            ),
            Node(
                func=create_confusion_matrix,
                inputs="bank_marketing",
                outputs="Models_Confusion_Matrix",
            ),
            Node(
                func=plot_model_metrics,
                inputs="results",
                outputs="Model_Performance_Plots",
            ),
            Node(
                func=plot_feature_importance_for_MLModels,
                inputs=["rf_model", "gb_model", "lr_model", "xgb_model", "lgbm_model", "catboost_model", "feature_names"],
                outputs="Feature_Importance_Plots_for_MLModels",
            ),
            Node(
                func=plot_roc_curve_for_MLModels,
                inputs=["rf_model", "gb_model", "lr_model", "xgb_model", "lgbm_model", "catboost_model", "X_test", "y_test"],
                outputs="ROC_Curve_Plots_for_MLModels",
            ),
        ]
    )
