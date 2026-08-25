import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error, mean_squared_error

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATA_PATH =PROJECT_ROOT / "data" / "split"
MODELS_DIR.mkdir(exist_ok=True)
TEST_DATA_PATH.mkdir(exist_ok=True)

def load_model():
    # Check if path exists
    model_lr_path = MODELS_DIR / "linear_regression_pipeline.pkl"
    model_rdf_path = MODELS_DIR / "randomforest_regression_pipeline.pkl"
    model_xgb_path = MODELS_DIR / "xgboost_pipeline.pkl"

    loaded_models = []
    if os.path.exists(model_lr_path):
        print(f"✅ Success: Model found at {os.path.abspath(model_lr_path)}")
        pipeline_lr = joblib.load(model_lr_path)
        loaded_models.append(pipeline_lr)

    if os.path.exists(model_rdf_path):
        print(f"✅ Success: Model found at {os.path.abspath(model_rdf_path)}")
        pipeline_rdf = joblib.load(model_rdf_path)
        loaded_models.append(pipeline_rdf)

    if os.path.exists(model_xgb_path):
        print(f"✅ Success: Model found at {os.path.abspath(model_xgb_path)}")
        pipeline_xgb = joblib.load(model_xgb_path)
        loaded_models.append(pipeline_xgb)

    if not loaded_models:
        print("❌ No model found.")

    print(f"✅ {len(loaded_models)} model(s) successfully loaded.")

    return loaded_models

def load_test_data():
    df_test = pd.read_csv(TEST_DATA_PATH / "test.csv")
    X_test = pd.read_csv(TEST_DATA_PATH / "X_test.csv")
    y_test = pd.read_csv(TEST_DATA_PATH / "y_test.csv")
   
    print('test data successfully loaded.')

    return df_test, X_test, y_test

def run_prediction_evaluation(model, X_test, y_test):
    results =[]

    # for m in models:
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_test, y_pred),4)
    mse = round(mean_squared_error(y_test,y_pred),4)
    rmse = round(root_mean_squared_error(y_test, y_pred),4)
    mae = round(mean_absolute_error(y_test, y_pred),4)

    result = { 'model': model,
                'y_pred': y_pred, 
                'R2': r2, 
                'MSE':mse, 
                'RMSE': rmse, 
                'MAE': mae }
    
    # results.append(result)

    return result


# to remove
def predict(model_pipeline, X_test):
    y_pred = model_pipeline.predict(X_test)

    return y_pred

# to remove
def get_predictions(models, X_test):
    y_preds=[]
    for m in models:
        y_pred = {'model': m['model'],'y_pred': m['pipeline'].predict(X_test)}
        y_preds.append(y_pred)

    return y_preds

# to remove
def get_evaluation_results(y_preds, y_test):
    eval_results = []
    for y in y_preds:
        r2 = round(r2_score(y_test, y['y_pred']),4)
        mse = round(mean_squared_error(y_test,y['y_pred']),4)
        rmse = round(root_mean_squared_error(y_test, y['y_pred']),4)
        mae = round(mean_absolute_error(y_test, y['y_pred']),4)
        eval = {'model':y['model'], 'R2': r2, 'MSE':mse, 'RMSE': rmse, 'MAE': mae}
        eval_results.append(eval)

    return eval_results




# to remove
def evaluate_model(y_test, y_pred):
    r2 = round(r2_score(y_test, y_pred),4)
    mse = round(mean_squared_error(y_test,y_pred),4)
    rmse = round(root_mean_squared_error(y_test, y_pred),4)
    mae = round(mean_absolute_error(y_test, y_pred),4)

    print(f"R² Score: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error:{mae}")

    return {'r2': r2, 'rmse': rmse, 'mse': mse, 'mae': mae}

def get_feature_importance(pipeline,model_name):
    regressor = pipeline.named_steps['regressor']
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    # Extract the actual feature name from the split list, making it a string
    feature_names = [f.split('__')[-1] for f in feature_names]

    if model_name == 'LinearRegression':
        impts = abs(regressor.coef_)
    else:
        impts = abs(regressor.feature_importances_)

    fea_imp = pd.DataFrame({
        "feature": feature_names,
        "Importance": impts
    }).sort_values("Importance", ascending=False)

    print(fea_imp.sort_values("Importance", ascending=False).head(10))
    plt.figure(figsize=(10,6))
    sns.barplot(data = fea_imp,x=fea_imp['feature'],y=fea_imp['Importance'], alpha=0.5,color='gray')
    plt.xticks(rotation='vertical')
    plt.title(f"Features Importance in {regressor.__class__.__name__}")
    plt.show()
    return fea_imp

def get_r2_by_threshold(df_model_r2, threshold):    
    df = df_model_r2.copy()
    df['Pass_Threshold'] = df['R2'].apply(
        lambda x: 'Pass' if x >= threshold else 'Fail'
    )
    return df

def get_mae_by_threshold(df_model_mae, df_test):
    naive_pred  = df_test['usage_lag_1'] 
    baseline_mae = mean_absolute_error(df_test['Usage_kWh'], naive_pred )

    df = df_model_mae.copy()
    improvement = (baseline_mae - df['MAE'] ) / baseline_mae * 100
    df['MAE_Baseline'] = round(baseline_mae,4)
    df['MAE_imprv_percent'] = round(improvement,4)
   
    return df
  
def get_mae_by_loadtype(model_pipeline, df_test, X_test, target, overall_mae_threshold):
    
    df = df_test.copy()
    y_pred = model_pipeline.predict(X_test)
    y_true = df_test[target]
    df['abs_error'] = abs(y_pred - y_true)

    mae_by_loadtype = df.groupby('Load_Type')['abs_error'].mean()
    overall_mae = df['abs_error'].mean()
    pass_threshold = np.where(mae_by_loadtype <= (overall_mae * overall_mae_threshold), 'Pass','Fail')

    dic_result = {'Overall MAE':overall_mae,'MAE_by_loadtype':mae_by_loadtype, 'Pass_Threshold':pass_threshold }
    df_result = pd.DataFrame(dic_result)

    return df_result

def get_mae_by_daytype(model_pipeline, df_test,X_test, target,usage_threshold):
    df = df_test.copy()
    y_pred = model_pipeline.predict(X_test)
    y_true = df_test[target]
    df['abs_error'] = abs(y_pred - y_true)

    mae_weekday = df[df['is_weekend']==0]['abs_error'].mean()
    mae_weekend = df[df['is_weekend']==1]['abs_error'].mean()
    mae_gap_by_daytype = abs(mae_weekday - mae_weekend)
    pass_threshold = np.where(mae_gap_by_daytype <= usage_threshold,'Pass','Fail')

    dic_result = {'Weekday MAE':mae_weekday,'Weekend MAE':mae_weekend,'MAE_gap_by_daytype':mae_gap_by_daytype,'Pass_Threshold':pass_threshold }
    df_result = pd.DataFrame([dic_result])

    return df_result


