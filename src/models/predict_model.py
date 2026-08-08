import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error, mean_squared_error

def predict(model_pipeline, X_test):
    y_pred = model_pipeline.predict(X_test)

    return y_pred

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

def get_threshold_r2(r2_list,threshold):
    best_r2 =[]
    for r in r2_list:
        if r>threshold:
            best_r2.append(r)

    return best_r2


