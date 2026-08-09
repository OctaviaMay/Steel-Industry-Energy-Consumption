import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "interim_clean_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

def load_data():
    df = pd.read_csv(INTERIM_DATA_PATH, 
        parse_dates=["consumed_date"]
        )
    return df

def inspect_target_co2_relation(df, col, target):
    corr = df[col].corr(df[target])
    n_unique = df[col].nunique()
    print(f"correlation value ={corr:.4f}, unique values in {col}={n_unique}")
    return corr

def get_skewed_features(df,col_num):
    left_skew_feat=[]
    right_skew_feat=[]
    for col in col_num:
        skewness = round(df[col].skew(),4)
        if abs(skewness) != 0.0:
            if abs(skewness) < 0.5 :
                left_skew_feat.append(col)
            else:
                right_skew_feat.append(col)

    print(f"Left skewed features: {left_skew_feat}")
    print(f"Right skewed features: {right_skew_feat}")
    return left_skew_feat, right_skew_feat

def add_time_features(df):
    df = df.copy()

    # cyclical encoding so midnight wraps cleanly
    df["nsm_sin"] = np.sin(2 * np.pi * df["NSM"] / 86400)
    df["nsm_cos"] = np.cos(2 * np.pi * df["NSM"] / 86400)

    if 'WeekStatus' in df.columns:
        df['is_weekend'] = np.where(df['WeekStatus'] !='Weekday', 1 , 0).astype(int)

    if 'Load_Type' in df.columns:
        loadtype_map= {
            'Light_Load': 1,
            'Medium_Load' : 2,
            'Maximum_Load' : 3
        }
            
        # df['nsm_loadtype'] = df['NSM'] * df['Load_Type'].map(loadtype_map)
        df['nsm_sin_loadtype'] = df['nsm_sin'] * df['Load_Type'].map(loadtype_map)
        df['nsm_cos_loadtype'] = df['nsm_cos'] * df['Load_Type'].map(loadtype_map)

    return df

def add_power_features(df, unity_threshold):
    df = df.copy()

    required_cols = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return df
    else:
        # perform feature engineering
        lag_power = df['Lagging_Current_Reactive.Power_kVarh']
        lead_power = df['Leading_Current_Reactive_Power_kVarh']
        lag_pf = df['Lagging_Current_Power_Factor']
        lead_pf = df['Leading_Current_Power_Factor']

        # Get total reactive power 
        df['total_reactive'] = lag_power + lead_power

        # Get dominance power mode
        condition = [
            lag_power > lead_power,
            lead_power > lag_power,
            abs(lag_power - lead_power) == 0
        ]
        df["reactive_mode"] = np.select(
            condition,
            ["lagging_dominant", "leading_dominant","near_zero"],
            default="near_zero",
        )

        # Get power loss : Distance from ideal power 100 can indicate how that machine is inefficient.
        # Power factor close to 100% indicates efficient electrical utilization. pf_loss measures deviation from the ideal.
        # As per EDA, leading PF is near-constant so that loss of leading pf won't be calculated
        df['pf_loss'] = 100 - lag_pf

        # Get electrical load : High reactive power and poor PF usually indicate heavier electrical loading.
        df["electrical_load"] = df["Lagging_Current_Reactive.Power_kVarh"] * (100 - lag_pf)

         # Leading_Current_Power_Factor mostly at 100
        condition_unity = [
            lead_pf < unity_threshold,
            lead_pf >= unity_threshold
        ]
        is_unity = [0,1]
        df['near_unity_pf'] = np.select(condition_unity, is_unity, default=np.nan).astype(int)

    return df

def apply_logtransform(df, col_list):
    for col in col_list:
        df[col + '_logged'] = np.log1p(df[col])

    return df
    
def save_processed_data(df):
    df.to_csv(PROCESSED_DATA_PATH,index = False)
    print(f"Preprocessed Data is sucessfully saved.")
    


