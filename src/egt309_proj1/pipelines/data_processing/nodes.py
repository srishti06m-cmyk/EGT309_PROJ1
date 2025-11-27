from typing import List
import sqlite3
import numpy as np
import pandas as pd


#Loading raw database, path defined in pipeline.py and returns as dataframe for processing
def load_raw_data(db_path: str) -> pd.DataFrame:
    """Load raw data from SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)
    finally:
        conn.close()
    return df


#cleaning the age. EDA states 150 is an outlier and max age as 95. any age beyond 95 set as NaN.
def clean_age(df: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """Convert Age to integer and handle outliers."""
    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df.loc[df["Age"] > max_age, "Age"] = np.nan
    return df

#Standerdize coloumns with 'yes/no' to 1/0 for ML training. coloumns defined in pipelines.py
def encode_binary_flags(df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
    """Convert yes/no columns to 1/0."""
    df = df.copy()
    mapping = {"yes": 1, "no": 0}

    for col in binary_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )
    return df

#EDA mentions 999 means 'no previous contact'. makes coloumn neumeric and lists 'Had previous contact' as 0 and none as 1
def clean_previous_contact_days(df: pd.DataFrame, no_contact_value: int) -> pd.DataFrame:
    """Handle 'Previous Contact Days' and create 'Had Previous Contact'."""
    df = df.copy()
    col = "Previous Contact Days"
    
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Had Previous Contact"] = (df[col] != no_contact_value).astype(int)

    df.loc[df[col] == no_contact_value, col] = np.nan

    return df

#takes coloumns such as marital status, occupation, etc and makes all string lowercased and trimmed
def standardize_categories(df: pd.DataFrame, category_columns: List[str]) -> pd.DataFrame:
    """Lowercase + strip categorical text."""
    df = df.copy()
    for col in category_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
    return df

#to drop unused coloumns. defined in pipelines.py
def drop_unused_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop unneeded columns such as Client ID."""
    df = df.copy()
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])

#ensure important/target variable is at the end for convinience while training ML models.
def reorder_columns_for_model(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Move target column to end."""
    df = df.copy()
    if target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    return df
