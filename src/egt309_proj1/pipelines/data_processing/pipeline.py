from kedro.pipeline import Node, Pipeline

from .nodes import create_model_input_table, preprocess_bmarket

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=preprocess_bmarket,
                inputs="bmarket",
                outputs="preprocessed_bmarket",
                name="preprocess_bmarket_node",
            ),
            Node(
                func=create_model_input_table,
                inputs=["preprocessed_bmarket"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
