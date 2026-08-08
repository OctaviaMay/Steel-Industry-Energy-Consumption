import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

def split_data(df, split_by_col, test_size):
    df_sorted = df.sort_values(split_by_col).reset_index(drop=True)
    split_index =  int(len(df_sorted) * (1 - test_size))

    df_train = df_sorted.iloc[:split_index].copy()
    df_test = df_sorted.iloc[split_index:].copy()

    return df_train, df_test


def develop_model(model,param_grid,num_feature, cat_feature, X, y):
    # TO DO: apply Yeo-Johnson transformation
    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feature)
        ]
    )

    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('regressor', model)
    ])

    print(type(model))

    # Set up GridSearchCV
    pipeline_grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='r2',
        n_jobs=-1)

    # Fit GridSearchCV-model
    pipeline_grid_search.fit(X, y)

    # Access best model and get best score
    best_model = pipeline_grid_search.best_estimator_
    best_params = pipeline_grid_search.best_params_
    best_score = round(pipeline_grid_search.best_score_,3)
   
    print("Best parameters found:", best_params)
    print("Best score:", best_score)

    return best_model , best_params, best_score




