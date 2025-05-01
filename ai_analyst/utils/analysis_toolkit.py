import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from statsmodels.formula.api import ols


def correlation(c1: str, c2: str):   
    global df
    return df[c1].corr(df[c2])

def describe_df(columns: list = None, stats: list = None):
    global df
    desc = df[columns].describe() if columns else df.describe()
    if stats:
        desc = desc.loc[[s for s in stats if s in desc.index]]
    return desc.to_dict()

def groupby_aggregate(groupby_col: str, agg_col: str, agg_func: str):
    global df
    valid = {"mean", "sum", "min", "max", "count", "median", "std", "var"}
    if agg_func not in valid:
        raise ValueError("Unsupported agg_func")
    
    # Calculate the aggregation
    result = df.groupby(groupby_col)[agg_col].agg(agg_func)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    result.plot(kind='bar', ax=ax)
    ax.set_title(f'{agg_func.title()} of {agg_col} by {groupby_col}')
    ax.set_xlabel(groupby_col)
    ax.set_ylabel(f'{agg_func.title()} of {agg_col}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def groupby_aggregate_multi(groupby_cols: list, agg_dict: dict):
    global df
    return df.groupby(groupby_cols).agg(agg_dict).to_dict()

def filter_data(col: str, op: str, value):
    global df
    ops = {
        "==": df[col] == value, "!=": df[col] != value, "<": df[col] < value,
        ">": df[col] > value, "<=": df[col] <= value, ">=": df[col] >= value,
    }
    if op not in ops:
        raise ValueError("Bad operator")
    return len(df[ops[op]].index)

def boxplot_all_columns():
    global df
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def correlation_matrix():               
    global df
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def scatter_matrix_all_numeric():
    global df
    fig = sns.pairplot(df.select_dtypes(include=["int", "float"]))
    buf = io.BytesIO(); fig.fig.savefig(buf, format="png"); buf.seek(0)
    plt.close("all")
    return base64.b64encode(buf.read()).decode()

def line_plot_over_time(date_col: str, val_col: str, agg_func="mean", freq="D"):
    global df
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, val_col]).set_index(date_col)
    resampled = temp[val_col].resample(freq).agg(agg_func)
    fig, ax = plt.subplots()
    ax.plot(resampled.index, resampled.values)
    ax.set_title(f"{agg_func} of {val_col} by {freq}"); ax.set_xlabel(date_col); ax.set_ylabel(val_col)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def outlier_rows(col: str, z=3.0):
    global df
    z_scores = stats.zscore(df[col], nan_policy="omit")
    return df[abs(z_scores) > z].to_dict(orient="list")

def scatter_plot(x_col: str, y_col: str, hue_col: str = None, figsize: tuple = (10, 6)):
    """Create a scatter plot between two columns with optional hue parameter.
    
    Args:
        x_col (str): Name of the column for x-axis
        y_col (str): Name of the column for y-axis
        hue_col (str, optional): Name of the column to use for color encoding
        figsize (tuple, optional): Figure size (width, height)
    
    Returns:
        str: Base64 encoded PNG image of the scatter plot
    """
    global df
    fig, ax = plt.subplots(figsize=figsize)
    
    if hue_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    
    ax.set_title(f'Scatter Plot: {y_col} vs {x_col}')
    if hue_col:
        ax.set_title(f'Scatter Plot: {y_col} vs {x_col} (colored by {hue_col})')
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def analyze_missing_value_impact(column: str, target: str):
    """Analyze the impact of missing values in a column on regression with target variable.
    
    Args:
        column (str): Name of the column with missing values
        target (str): Name of the target column for regression
        
    Returns:
        str: Base64 encoded PNG image showing the regression results
    """
    global df
    # Create copies of the dataframe with different missing value treatments
    min_data_df = df.copy()
    min_data_df[column] = np.where(min_data_df[column].isna(), min_data_df[column].min(), 
                                 min_data_df[column])

    max_data_df = df.copy()
    max_data_df[column] = np.where(max_data_df[column].isna(), max_data_df[column].max(), 
                                 max_data_df[column])

    # Fit regression models
    min_model = ols(f"{target}~{column}", data=min_data_df).fit()
    max_model = ols(f"{target}~{column}", data=max_data_df).fit()

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for min imputation
    sns.regplot(x=column, y=target, data=min_data_df, ax=ax1)
    ax1.set_title(f'Regression with Min Imputation\nR² = {min_model.rsquared:.3f}')
    
    # Plot for max imputation
    sns.regplot(x=column, y=target, data=max_data_df, ax=ax2)
    ax2.set_title(f'Regression with Max Imputation\nR² = {max_model.rsquared:.3f}')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()
