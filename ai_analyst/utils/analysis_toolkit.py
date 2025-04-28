import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64


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
    fig, ax = plt.subplots(); ax.plot(resampled.index, resampled.values)
    ax.set_title(f"{agg_func} of {val_col} by {freq}"); ax.set_xlabel(date_col); ax.set_ylabel(val_col)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def outlier_rows(col: str, z=3.0):
    global df
    z_scores = stats.zscore(df[col], nan_policy="omit")
    return df[abs(z_scores) > z].to_dict(orient="list")
