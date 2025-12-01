from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_models, evaluate_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=dict(
                    df="model_input_table",
                    test_size="params:model_test_size",
                    random_state="params:model_random_state",
                ),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_models,
                inputs=["X_train", "y_train"],
                outputs="trained_models",
                name="train_models_node",
            ),
            node(
                func=evaluate_models,
                inputs=["trained_models", "X_test", "y_test"],
                outputs=["best_model", "evaluation_metrics"],
                name="evaluate_models_node",
            ),
        ]
    )