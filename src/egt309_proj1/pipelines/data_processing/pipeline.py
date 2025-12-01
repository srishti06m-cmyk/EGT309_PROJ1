from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_raw_data,
    clean_age,
    standardize_categories,
    replace_categories,
    clean_previous_contact_days,
    clean_campaign_calls,
    encode_binary_flags,
    add_age_bins,
    add_job_classification,
    add_loan_count,
    clean_unknown_categories,
    drop_unused_columns,
    reorder_columns_for_model
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        #loading raw data from SQLite
        node(
            func=load_raw_data,
            inputs="params:db_path",
            outputs="raw_loaded",
            name="load_raw_data_node"
        ),

        #cleaning the age column
        node(
            func=clean_age,
            inputs=dict(
                df="raw_loaded",
                age_column="params:age_column",
                max_age="params:max_age"
            ),
            outputs="age_cleaned",
            name="clean_age_node"
        ),

        #standardizing text categories such as lowercase and trim
        node(
            func=standardize_categories,
            inputs=dict(
                df="age_cleaned",
                category_columns="params:category_columns"
            ),
            outputs="categories_standardized",
            name="standardize_categories_node"
        ),

        #replacing specific category inconsistencies such as cell = cellular etc.
        node(
            func=replace_categories,
            inputs=dict(
                df="categories_standardized",
                replacements="params:replacement_mappings"
            ),
            outputs="categories_replaced",
            name="replace_categories_node"
        ),

        #cleaning previous contact days such as 999 = NaN and create had previous contact
        node(
            func=clean_previous_contact_days,
            inputs=dict(
                df="categories_replaced",
                column="params:prev_contact_column",
                no_contact_value="params:prev_contact_code"
            ),
            outputs="previous_contact_cleaned",
            name="clean_previous_contact_node"
        ),
        
        #cleaning campaign calls to absolute values
        node(
            func=clean_campaign_calls,
            inputs=dict(
                df="previous_contact_cleaned",
                column="params:campaign_calls_column"
            ),
            outputs="campaign_calls_cleaned",
            name="clean_campaign_calls_node"
        ),

       #removing unknown / unwanted categories
        node(
            func=clean_unknown_categories,
            inputs=dict(
                df="campaign_calls_cleaned",
            ),
            outputs="cleaned_categories",
            name="clean_unknown_categories_node"
        ),

        #Adding age bins (youth, adult, senior, etc.)
        node(
            func=add_age_bins,
            inputs=dict(
                df="cleaned_categories",
                age_column="params:age_column",
                bins="params:age_bins",
                labels="params:age_labels"
            ),
            outputs="age_binned",
            name="add_age_bins_node"
        ),

        #adding job classification feature (occupation = job type)
        node(
            func=add_job_classification,
            inputs=dict(
                df="age_binned",
                occupation_column="params:occupation_column",
                job_map="params:job_map"
            ),
            outputs="job_classified",
            name="add_job_classification_node"
        ),

        #adding loan count (housing loan + personal loan)
        node(
            func=add_loan_count,
            inputs=dict(
                df="job_classified",
                housing_col="params:housing_column",
                personal_col="params:personal_column"
            ),
            outputs="loan_count_added",
            name="add_loan_count_node"
        ),

        #dropping unused columns (IDs, redundant fields)
        node(
            func=drop_unused_columns,
            inputs=dict(
                df="loan_count_added",
                drop_cols="params:drop_columns"
            ),
            outputs="columns_dropped",
            name="drop_unused_columns_node"
        ),

        #reordering columns so target is at the end for easier processing
        node(
            func=reorder_columns_for_model,
            inputs=dict(
                df="columns_dropped",
                target_col="params:target_column"
            ),
            outputs="model_input_table",
            name="reorder_columns_node"
        )
    ])
