import pandas as pd
from itertools import combinations

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
