from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_raw_data,
    clean_age,
    encode_binary_flags,
    clean_previous_contact_days,
    standardize_categories,
    drop_unused_columns,
    reorder_columns_for_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_raw_data,
                inputs="params:db_path",
                outputs="raw_bank_data",
                name="load_raw_data_node",
            ),
            node(
                func=clean_age,
                inputs=dict(
                    df="raw_bank_data",
                    max_age="params:max_age",
                ),
                outputs="bank_data_age_clean",
                name="clean_age_node",
            ),
            node(
                func=encode_binary_flags,
                inputs=dict(
                    df="bank_data_age_clean",
                    binary_columns="params:binary_columns",
                ),
                outputs="bank_data_binary_encoded",
                name="encode_binary_flags_node",
            ),
            node(
                func=clean_previous_contact_days,
                inputs=dict(
                    df="bank_data_binary_encoded",
                    no_contact_value="params:no_contact_value",
                ),
                outputs="bank_data_contact_clean",
                name="clean_previous_contact_days_node",
            ),
            node(
                func=standardize_categories,
                inputs=dict(
                    df="bank_data_contact_clean",
                    category_columns="params:category_columns",
                ),
                outputs="bank_data_categoricals_clean",
                name="standardize_categories_node",
            ),
            node(
                func=drop_unused_columns,
                inputs=dict(
                    df="bank_data_categoricals_clean",
                    columns_to_drop="params:columns_to_drop",
                ),
                outputs="bank_data_reduced",
                name="drop_unused_columns_node",
            ),
            node(
                func=reorder_columns_for_model,
                inputs=dict(
                    df="bank_data_reduced",
                    target_col="params:target_col",
                ),
                outputs="model_input_table",
                name="reorder_columns_for_model_node",
            ),
        ]
    )
