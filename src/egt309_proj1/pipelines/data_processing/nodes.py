from typing import List
import sqlite3
import numpy as np
import pandas as pd


def load_raw_data(db_path: str) -> pd.DataFrame:
    """Load raw data from SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)
    finally:
        conn.close()
    return df


def clean_age(df: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """Convert Age to integer and handle outliers."""
    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df.loc[df["Age"] > max_age, "Age"] = np.nan
    return df


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


def clean_previous_contact_days(df: pd.DataFrame, no_contact_value: int) -> pd.DataFrame:
    """Handle 'Previous Contact Days' and create 'Had Previous Contact'."""
    df = df.copy()
    col = "Previous Contact Days"
    
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Had Previous Contact"] = (df[col] != no_contact_value).astype(int)

    df.loc[df[col] == no_contact_value, col] = np.nan

    return df


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


def drop_unused_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop unneeded columns such as Client ID."""
    df = df.copy()
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])


def reorder_columns_for_model(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Move target column to end."""
    df = df.copy()
    if target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    return df
