from typing import List
import sqlite3
import numpy as np
import pandas as pd

#Loads table into a pandas dataframe 
def load_raw_data(db_path: str) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM bank_marketing", sqlite3.connect(db_path))

#removes outliers such as 150, change text like 39 years to 39, converts to neumeric and invalid values become NaN
def clean_age(df: pd.DataFrame, age_column: str, max_age: int) -> pd.DataFrame:
    df = df.copy()

    df[age_column] = (
        df[age_column].astype(str)
        .str.replace("years", "", regex=False)
        .str.strip()
    )
    df[age_column] = pd.to_numeric(df[age_column], errors="coerce")
    df.loc[df[age_column]> max_age, age_column] = np.nan
    median_age = df[age_column].median(skipna=True)
    df[age_column]=df[age_column].fillna(median_age)
    return df

#standerdizing categories, making everything lowercase, remove extra spaces, convert to string 
def standardize_categories(df: pd.DataFrame, category_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in category_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

#Applies specific caetgory changes. defined in parameters.
def replace_categories(df: pd.DataFrame, replacements: dict) -> pd.DataFrame:
    """
    Example replacements:
        {"Contact Method": {"cell": "cellular", "telephone":"telephone"}}
    """
    df = df.copy()
    for col, mapping in replacements.items():
        df[col] = df[col].replace(mapping)
    return df

#converts previous contact days to neumeric, replace 999 with NaN as it is a placeholder value else 'Had previous ocntact'
def clean_previous_contact_days(df: pd.DataFrame, column: str, no_contact_value: int) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Had Previous Contact"] = (df[column] != no_contact_value).astype(int)
    df.loc[df[column] == no_contact_value, column] = np.nan
    return df

def clean_campaign_calls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cleans the Campaign Calls column by converting to absolute values"""
    df= df.copy()
    df[column]= pd.to_numeric(df[column], errors="coerce").abs()
    return df

#convert using mapping provided, string
def encode_binary_flags(df: pd.DataFrame, binary_columns: List[str], mapping: dict) -> pd.DataFrame:
    df = df.copy()
    for col in binary_columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
    return df

#adding age bins to show differences between age groups and subscription correlation 
def add_age_bins(df: pd.DataFrame, age_column: str, bins: list, labels: list) -> pd.DataFrame:
    df = df.copy()
    df["Age_Group"] = pd.cut(df[age_column], bins=bins, labels=labels)
    return df

#mapping occupations into broader groups as occupation has many unique categories
def add_job_classification(df: pd.DataFrame, occupation_column: str, job_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["Job_Type"] = df[occupation_column].map(job_map)
    return df

#makes loans a neumeric feature
def add_loan_count(df: pd.DataFrame, housing_col: str, personal_col: str) -> pd.DataFrame:
    df = df.copy()
    df["Loan_Count"] = (
        (df[housing_col] == "yes").astype(int)
        + (df[personal_col] == "yes").astype(int)
    )
    return df

#Remove redundant old coloumns, remove junk, keep dataset compact
def drop_unused_columns(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

#removing unkown or rare categories 
def clean_unknown_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EDA-based rules to remove rare / low-quality categories:
    - Drop rows where Education Level is 'unknown' or 'illiterate'
    - Drop rows where Credit Default is 'yes'
    - Drop rows where Housing Loan is 'unknown'
    - Drop rows where Personal Loan is 'unknown'
    - Fill remaining missing Housing/Personal Loan with 'no_info'
    """
    df = df.copy()

    if "Education Level" in df.columns:
        df = df[~df["Education Level"].isin(["unknown", "illiterate"])]

    if "Marital Status" in df.columns:
        df = df[df["Marital Status"]!="unknown"]

    if "Occupation" in df.columns:
        df = df[df["Occupation"]!="no_info"]

    if "Credit Default" in df.columns:
        df = df[df["Credit Default"] != "yes"]

    if "Housing Loan" in df.columns:
        df = df[~df["Housing Loan"].isin(["unknown"])]
    if "Personal Loan" in df.columns:
        df = df[~df["Personal Loan"].isin(["unknown"])]

    if "Housing Loan" in df.columns:
        df["Housing Loan"] = df["Housing Loan"].replace("none", "no_info").fillna("no_info")

    if "Personal Loan" in df.columns:
        df["Personal Loan"] = df["Personal Loan"].replace("none", "no_info").fillna("no_info")

    return df

#moving the target coloumn to the end of the dataframe for easier ML training.
def reorder_columns_for_model(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target_col] + [target_col]
    return df[cols]
