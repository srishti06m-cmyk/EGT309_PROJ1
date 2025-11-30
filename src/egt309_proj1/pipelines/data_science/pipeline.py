from kedro.pipeline import Node, Pipeline

from .nodes import (
    split_data, 
    train_RandomForestClassifier, 
    train_GradientBoostingClassifier, 
    train_LogisticRegression, 
    evaluate_MachineLearningModels
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            Node(
                func=train_RandomForestClassifier,
                inputs=["X_train", "y_train"],
                outputs="rf_model_metrics",
                name="train_rf_model_node",
            ),
            Node(
                func=train_GradientBoostingClassifier,
                inputs=["X_train", "y_train"],
                outputs="gb_model_metrics",
                name="train_gb_model_node",
            ),
            Node(
                func=train_LogisticRegression,
                inputs=["X_train", "y_train"],
                outputs="lr_model_metrics",
                name="train_lr_model_node",
            ),
            Node(
                func=train_XGBClassifier,
                inputs=["X_train", "y_train"],
                outputs="xgb_model_metrics",
                name="train_xgb_model_node",
            ),
            Node(
                func=train_LGBMClassifier,
                inputs=["X_train", "y_train"],
                outputs="lgbm_model_metrics",
                name="train_lgbm_model_node",
            ),
            Node(
                func=train_CatBoostClassifier,
                inputs=["X_train", "y_train"],
                outputs="catboost_model_metrics",
                name="train_catboost_model_node",
            ),
            Node(
                func=evaluate_MachineLearningModels,
                inputs=["rf_model","gb_model", "lr_model", "xgb_model", "lgbm_model", "catboost_model", "X_test", "y_test"],
                outputs="all_models_metrics",
                name="evaluate_all_models_node",
            ),

        ]
    )