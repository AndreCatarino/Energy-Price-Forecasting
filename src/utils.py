import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def load_energy_data(file_path:str="../data/raw/energy_dataset.csv", date:str="time") -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=[date])
    return df

def load_weather_data(file_path:str="../data/raw/weather_features.csv", date:str="dt_iso") -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=[date])
    return df

def load_preprocessed_data(file_path:str="../data/processed/processed_merged_data.pkl") -> pd.DataFrame: 
    df = pd.read_pickle(file_path)
    return df

def id_outliers(df:pd.DataFrame)-> pd.DataFrame:
    """
    Identify outliers for each column of a dataframe
    :param df: dataframe
    :return: dataframe with lower and upper bound and number of outliers
    """
    result_data = []
    for col_name in df.columns:
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = len(df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)])
        result_data.append([lower_bound, upper_bound, n_outliers])
    outliers = pd.DataFrame(result_data, columns=['lower_bound', 'upper_bound', 'n_outliers'], index=df.columns)
    return outliers

def filtered_heatmap(df:pd.DataFrame, absthreshold:int=0) -> pd.DataFrame:
    """
    Filter a correlation matrix by absolute value threshold
    :param df: correlation matrix
    :param absthreshold: absolute value threshold
    :return: filtered correlation matrix
    """
    passed = set()
    for (r,c) in combinations(df.columns, 2):
        if (abs(df.loc[r,c]) >= absthreshold) and (r != c):
            passed.add(r)
            passed.add(c)
    passed = sorted(passed)
    return df.loc[passed,passed]

def compare_metrics(eval_df:pd.DataFrame, stat:str, metrics:list) -> None:
    """
    Compare the mean or standard deviation of each metric for each model
    :param eval_df: dataframe with evaluation metrics for each model
    :param stat: 'mean' or 'std'
    :param metrics: list of metrics to compare
    :return: None
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(eval_df['Model'], eval_df[metric].apply(lambda x: eval(f'np.{stat}(x)')), color='skyblue')
        ax.set_title(metric)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_xticklabels(eval_df['Model'], rotation=45, ha='right')
        ax.yaxis.grid()

    # Adjust the layout to avoid overlapping titles
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'../plots/metrics_{stat}.png')

    # Show the plot
    plt.show()