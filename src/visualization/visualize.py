import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Project root = two levels up from this file (src/visualization/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------
# Global styling
# -----------------------------
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

palette = sns.color_palette("Set2") # Custom palette

def plot_box(df, x, columns):
    plt.figure(figsize=(12,6))
    for col in columns:
        plt.subplot(1,2,columns.index(col)+1)
        sns.boxplot(data= df, x=x, y=col, color=palette[2])
        plt.title(f"{col} by Load Type")
    plt.show()


def plot_histogram(df,target, title, islog=False):
    if islog:
        data = np.log1p(df[target])
    else:
        data = df[target]

    sns.histplot(data= data,
                 kde=True,
                 color=palette[2],
                 edgecolor='white',
                linewidth=1,
                alpha=0.85)
    plt.title(title)
    plt.show()
    

def plot_bar(df, x, y, x_label, y_label, title):
    plt.figure(figsize=(8,6))
    sns.barplot(
    data=df,
    x=x,
    y=y,
    hue = x,
    legend=False
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    
def plot_scatter(df, col_num, target):
    plt.figure(figsize=(20,16))
    for col in col_num:
        plt.subplot(3,3, col_num.index(col)+1)
        sns.regplot(x=col, y=target, data=df,
            scatter_kws={"alpha":0.4},
            line_kws={"color":"red"})
    plt.show()

def analyze_distribution(df, col_num):
    left_skew_feat = []
    right_skew_feat = []
    plt.figure(figsize=(20,16))

    for col in col_num:
        plt.subplot(3, 2, col_num.index(col) + 1)
        sns.histplot(data=df, x=col,kde=True)
        skewness = round(df[col].skew(),4)
        if skewness != 0.0:
            if skewness < 0.0 :
                left_skew_feat.append(col)
            else:
                right_skew_feat.append(col)
        plt.title(f'Distribution of {col} [skewness:{skewness}]')
    plt.show()

    return left_skew_feat, right_skew_feat

def plot_right_skewed_features(right_skew_feat, df, title):
    plt.figure(figsize=(18,12))
    for col in right_skew_feat:
        plt.subplot(2,2,right_skew_feat.index(col)+1)
        sns.histplot(data = np.log1p(df[col]),
                    kde=True,
                    color = palette[2],
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.85)
        plt.title(f"{title} - {col}")
    plt.show()

def analyze_correlation(df, col_num, target):
    sns.heatmap(data= df[col_num].corr(), cmap='coolwarm', fmt='.2f', annot=True)
    plt.title("Feature Correlation Heatmap Between Numeric Variables", fontsize=12, fontweight='normal',pad=12)
    plt.show()

    return df[col_num].corr()[target].drop(target)

def plot_score_comparison_chart(models, r2_list, mae_list):
    bar_width = 0.35
    x = range(len(models))

    plt.figure(figsize=(7,5))
    rbar = plt.bar(x, r2_list, width=bar_width, label='R2', color='skyblue')
    plt.bar_label(rbar, fmt= '%.3f')
    mbar = plt.bar([i + bar_width for i in x], mae_list, width=bar_width, label='Mean Absolute Error', color='orange')
    plt.bar_label(mbar, fmt = '%.3f')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('R2 and Mean Absolute Error Across CV Folds Comparison\n')
    plt.xticks([i + bar_width/2 for i in x], models)
    plt.ylim(0, 15)
    plt.legend()
    plt.show()
