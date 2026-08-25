import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import xgboost as xgb
import joblib

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITTED_DATA_PATH = PROJECT_ROOT / "data" / "split" 

MODELS_DIR.mkdir(exist_ok=True)
SPLITTED_DATA_PATH.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH,
                     parse_dates=["consumed_date"])
    return df

def split_data(df, split_by_col, test_size):
    # df_sorted = df.sort_values(split_by_col).reset_index(drop=True)
    # split_index =  int(len(df_sorted) * (1 - test_size))
    split_index =  int(len(df) * (1 - test_size))

    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()

    return df_train, df_test

def save_splitted_test_data(df_test,X_test, y_test):
    df_test.to_csv(SPLITTED_DATA_PATH / "test.csv", index = False)
    X_test.to_csv(SPLITTED_DATA_PATH / "X_test.csv", index=False)
    y_test.to_csv(SPLITTED_DATA_PATH / "y_test.csv", index=False)
    print(f"Test dataset is successfully saved.")


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
   
    return best_model , best_params, best_score


def get_models(num_feature, cat_feature, X_train, y_train):
    models = []

    # base model - Linear Regression
    model_linear = LinearRegression()
    param_grid = {
                'regressor__fit_intercept': [True,False]
    }

    pipeline_linear , params_linear, score_linear = develop_model(model_linear, param_grid,num_feature, cat_feature, X_train, y_train)
    model_temp = {'model': 'Linear Regression', 
                  'pipeline': pipeline_linear, 
                  'best_params': params_linear, 
                  'best_score': score_linear}
    models.append(model_temp)

    # compare model 1 - Random Forest
    model_rdf = RandomForestRegressor()
    param_grid = {
            'regressor__n_estimators': [50],
            'regressor__max_depth': [ 10],
            'regressor__min_samples_split': [ 5]
    }
    pipeline_rdf, params_rdf, score_rdf = develop_model(model_rdf, param_grid,num_feature, cat_feature, X_train, y_train)
    model_temp = {'model': 'Random Forest Regression', 
                      'pipeline': pipeline_rdf, 
                      'best_params': params_rdf, 
                      'best_score': score_rdf}
    models.append(model_temp)

    # compare model 2 - XGBRegressor
    model_xgb = xgb.XGBRegressor()
    param_grid = {
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.1, 0.01, 0.05],
                'regressor__n_estimators': [50, 100, 200]
    }
    pipeline_xgb, params_xgb, score_xgb = develop_model(model_xgb, param_grid,num_feature, cat_feature, X_train, y_train)
    model_temp = {'model': 'XGBoost', 
                        'pipeline': pipeline_xgb, 
                        'best_params': params_xgb, 
                        'best_score': score_xgb}
    models.append(model_temp)

    return models

def save_models(model_pipelines):
    for pipeline in model_pipelines:
        if pipeline['model'] == 'Linear Regression':
            joblib.dump(pipeline['pipeline'], MODELS_DIR / "linear_regression_pipeline.pkl")

        elif pipeline['model'] == 'Random Forest Regression':
            joblib.dump(pipeline['pipeline'], MODELS_DIR / "randomforest_regression_pipeline.pkl")

        elif pipeline['model'] == 'XGBoost':
            joblib.dump(pipeline['pipeline'], MODELS_DIR / "xgboost_pipeline.pkl")

    print('Models successfully saved.')
