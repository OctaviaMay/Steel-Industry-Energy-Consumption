import numpy as np
import pandas as pd

from pathlib import Path

# Project root = two levels up from this file (src/data/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Steel_industry_data.csv"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "interim_clean_data.csv"

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def get_data_overview(df):
    rows = df.shape[0]
    columns = df.shape[1]

    return rows, columns


def check_data(df):
    # Checking missing value
    col_with_missing = list(df.columns[df.isnull().any()])
    # Checking duplicated value
    duplicated_rows = df[df.duplicated(keep=False)]

    if len(col_with_missing) > 0:
        missing_value_msg = f"Columns with missing value: {col_with_missing}"
    else:
        missing_value_msg = f"There is no missing value in all columns."
    if(len(duplicated_rows) > 0):
        duplicated_rows_msg = f"Number of duplicated rows:{len(duplicated_rows)}"
    else:
        duplicated_rows_msg = f"No duplicated rows found in dataset."

    return missing_value_msg, duplicated_rows_msg

def validate_data(df):
    col_num_unique_list = []
    col_cat_unique_list = []

    col_num = df.select_dtypes(include=['float64','int64']).columns.tolist()
    for col in col_num:
        col_num_unique = {'column': col, 'Total_Unique_Value':df[col].nunique(), 'Max': df[col].max(), 'Min':df[col].min()}
        col_num_unique_list.append(col_num_unique)

    col_cat = df.select_dtypes(include='object').columns.tolist()
    for col in col_cat:
        col_cat_unique = {'column':col, 'Total_Unique_Value': df[col].nunique(), 'Unique_Cat_List': df[col].unique()}
        col_cat_unique_list.append(col_cat_unique)

    return col_num_unique_list, col_cat_unique_list

def clean_data(df):
    df["consumed_date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.drop('date', axis=1)

    # Save to interim data folder
    df.to_csv(INTERIM_DATA_PATH, index = False)

    return df

def get_outlier_ratio(df, detect_outliers_feature):
    dict_outlier = []
    for col in detect_outliers_feature:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1
        upper = q3 + (1.5 * iqr)
        lower = q1 - (1.5 * iqr)

        outlier_count = len(df[(df[col]<=lower) | (df[col]>=upper)])
        outlier_ratio = round((outlier_count/df.shape[0]) * 100, 2)
        if outlier_ratio > 0:
            temp_outlier_ratio = {'variable': col, 'outlier_ratio': outlier_ratio, 'lower_bound':lower,'upper_bound':upper}
            dict_outlier.append(temp_outlier_ratio)
            
    df_outlier = pd.DataFrame(dict_outlier)
    return df_outlier