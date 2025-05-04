import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import probplot


def correlation(c1: str, c2: str):   
    global df
    # Check if columns are numeric
    if not (pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2])):
        return f"Error: Both columns must be numeric. {c1} and {c2} types are {df[c1].dtype} and {df[c2].dtype}"
    return df[c1].corr(df[c2])

def describe_df(columns: list = None, stats: list = None):
    global df
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns.tolist()
    
    # Separate numeric and non-numeric columns
    numeric_cols = df[columns].select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = df[columns].select_dtypes(exclude=['int64', 'float64']).columns
    
    result = {}
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        numeric_desc = df[numeric_cols].describe()
        if stats:
            numeric_desc = numeric_desc.loc[[s for s in stats if s in numeric_desc.index]]
        result.update(numeric_desc.to_dict())
    
    # Handle non-numeric columns
    if len(non_numeric_cols) > 0:
        non_numeric_desc = df[non_numeric_cols].describe(include=['object'])
        if stats:
            non_numeric_desc = non_numeric_desc.loc[[s for s in stats if s in non_numeric_desc.index]]
        result.update(non_numeric_desc.to_dict())
    
    return result

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
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_df.columns) == 0:
        return "No numeric columns found in the dataset"
        
    fig, ax = plt.subplots(figsize=(12, 6))
    numeric_df.boxplot(ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def correlation_matrix():               
    global df
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_df.columns) == 0:
        return "No numeric columns found in the dataset"
        
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
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

def histogram_plot(column: str, bins: int = 30):
    """Create a histogram plot for a numeric column.
    
    Args:
        column (str): Name of the column to plot
        bins (int): Number of bins for the histogram
        
    Returns:
        str: Base64 encoded PNG image of the histogram
    """
    global df
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def qq_plot(column: str):
    """Create a Q-Q plot to assess normality of a numeric column.
    
    Args:
        column (str): Name of the column to plot
        
    Returns:
        str: Base64 encoded PNG image of the Q-Q plot
    """
    global df
    fig, ax = plt.subplots(figsize=(10, 6))
    probplot(df[column].dropna(), dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot of {column}')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def density_plot(column: str):
    """Create a density plot for a numeric column.
    
    Args:
        column (str): Name of the column to plot
        
    Returns:
        str: Base64 encoded PNG image of the density plot
    """
    global df
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x=column, ax=ax)
    ax.set_title(f'Density Plot of {column}')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def anova_test(group_col: str, value_col: str):
    """Perform ANOVA test to compare means across groups.
    
    Args:
        group_col (str): Name of the categorical grouping column
        value_col (str): Name of the numeric value column
        
    Returns:
        dict: ANOVA test results including F-statistic and p-value
    """
    global df
    groups = [group for _, group in df.groupby(group_col)[value_col]]
    f_stat, p_value = f_oneway(*groups)
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def chi_square_test(col1: str, col2: str):
    """Perform chi-square test of independence between two categorical variables.
    
    Args:
        col1 (str): Name of the first categorical column
        col2 (str): Name of the second categorical column
        
    Returns:
        dict: Chi-square test results including statistic and p-value
    """
    global df
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return {
        'chi2_statistic': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
        'significant': p < 0.05
    }

def t_test(col1: str, col2: str):
    """Perform independent t-test between two numeric columns.
    
    Args:
        col1 (str): Name of the first numeric column
        col2 (str): Name of the second numeric column
        
    Returns:
        dict: T-test results including statistic and p-value
    """
    global df
    t_stat, p_value = ttest_ind(df[col1].dropna(), df[col2].dropna())
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def seasonal_decomposition(date_col: str, value_col: str, freq: str = "D"):
    """Perform seasonal decomposition of time series data.
    
    Args:
        date_col (str): Name of the date column
        value_col (str): Name of the value column
        freq (str): Frequency of the time series (default: "D" for daily)
        
    Returns:
        str: Base64 encoded PNG image of the decomposition plot
    """
    global df
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    temp = temp.set_index(date_col)
    
    decomposition = seasonal_decompose(temp[value_col], period=7 if freq == "D" else 12)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax1)
    decomposition.trend.plot(ax=ax2)
    decomposition.seasonal.plot(ax=ax3)
    decomposition.resid.plot(ax=ax4)
    
    ax1.set_title('Observed')
    ax2.set_title('Trend')
    ax3.set_title('Seasonal')
    ax4.set_title('Residual')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def autocorrelation_plot(column: str, lags: int = 30):
    """Create an autocorrelation plot for time series data.
    
    Args:
        column (str): Name of the time series column
        lags (int): Number of lags to plot
        
    Returns:
        str: Base64 encoded PNG image of the autocorrelation plot
    """
    global df
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(df[column].dropna(), lags=lags, ax=ax)
    ax.set_title(f'Autocorrelation Plot of {column}')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def create_interaction(col1: str, col2: str):
    """Create an interaction term between two numeric columns.
    
    Args:
        col1 (str): Name of the first numeric column
        col2 (str): Name of the second numeric column
        
    Returns:
        str: Name of the new interaction column
    """
    global df
    interaction_name = f"{col1}_x_{col2}"
    df[interaction_name] = df[col1] * df[col2]
    return interaction_name

def bin_numeric_column(column: str, bins: int = 5):
    """Create bins for a numeric column.
    
    Args:
        column (str): Name of the numeric column
        bins (int): Number of bins to create
        
    Returns:
        str: Name of the new binned column
    """
    global df
    binned_name = f"{column}_binned"
    df[binned_name] = pd.qcut(df[column], q=bins, labels=[f"bin_{i+1}" for i in range(bins)])
    return binned_name
