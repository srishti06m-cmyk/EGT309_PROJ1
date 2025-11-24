import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import IntegerType

def preprocess_bmarket(bmarket: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for bmarket.

    Args:
        bmarket: Raw data.
    Returns:
        Preprocessed data, with cleaned types and standardized columns.
    """
    bmarket = bmarket.withColumn("Age",regexp_replace(col("Age"), "years", "").cast(IntegerType()))
    bmarket = bmarket.withColumn("Subscription Status",when(col("Subscription Status")=="yes",True).otherwise(False).cast("BoonleanType"()))

    text_columns =["Client ID","Age","Occupation","Marital Status",
                   "Education Level","Credit Default","Housing Loan",
                   "Personal Loan","Contact Method","Campaign Calls",
                   "Previous Contact Days","Subscription Status"]
    for column in text_columns:
        if column in bmarket.columns:
            bmarket[column] = bmarket[column].astype(str).str.lower()
    
    return bmarket

def create_model_input_table(bmarket: pd.DataFrame) -> pd.DataFrame:
    """Create model input table from bmarket.
    Args:
        bmarket: Raw dataframe for bmarket dataset.
    Returns:
        Preproessed Model input table.

    """
 
    model_input_table = preprocess_bmarket(bmarket)

    return model_input_table