from kedro.pipeline import Node, Pipeline

from .nodes import split_data, train_RandomForestClassifier, train_GradientBoostingClassifier, train_LogisticRegression, evaluate_MachineLearningModels


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
                outputs="rf_model",
                name="train_rf_model_node",
            ),
            Node(
                func=train_GradientBoostingClassifier,
                inputs=["X_train", "y_train"],
                outputs="gb_model",
                name="train_gb_model_node",
            ),
            Node(
                func=train_LogisticRegression,
                inputs=["X_train", "y_train"],
                outputs="lr_model",
                name="train_lr_model_node",
            ),
            Node(
                func=evaluate_MachineLearningModels,
                inputs=["rf_model", "X_test", "y_test", "params:model_options.model_name_rf"],
                outputs="rf_model_evaluation",
                name="evaluate_rf_model_node",
            ),
            Node(
                func=evaluate_MachineLearningModels,
                inputs=["gb_model", "X_test", "y_test", "params:model_options.model_name_gb"],
                outputs="gb_model_evaluation",
                name="evaluate_gb_model_node",
            ),
            Node(
                func=evaluate_MachineLearningModels,
                inputs=["lr_model", "X_test", "y_test", "params:model_options.model_name_lr"],
                outputs="lr_model_evaluation",
                name="evaluate_lr_model_node",
            ),
         
        ]
    )
