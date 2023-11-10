import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional

class DataProcessor:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructor for DataProcessor class
        :param data: Dataframe to be processed
        :return: None
        """
        self.data = data
        self.X = None
        self.y = None
    
    def merge(self, df: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        Merge two dataframes
        :param df: Dataframe to be merged
        :param on: Column name to merge on
        :return: Merged dataframe
        """
        if self.data is None:
            self.data = df
        self.data = pd.merge(self.data, df, on=on)
        return self.data

    def create_feature_matrix_and_target_vector(self, target_column: str) -> tuple:
        """
        Create feature matrix and target vector
        :param target_column: Name of the target column
        :return: Feature matrix and target vector
        """
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]
        return self.X, self.y

    def scale(self, data:pd.DataFrame=Optional) -> pd.DataFrame:
        """ 
        Scale the data using StandardScaler
        :return: Scaled features dataframe
        """
        scaler = StandardScaler()
        if data is not pd.DataFrame and self.data is not None:
            scaled_data = scaler.fit_transform(self.data)
            self.data = pd.DataFrame(scaled_data, columns=self.data.columns)
            return self.data
        elif data is not None:
            scaled_X = scaler.fit_transform(data)
        else:
            raise Exception("Feature matrix not found. Run create_feature_matrix_and_target_vector() first.")
        return pd.DataFrame(scaled_X, columns=self.data.columns)
    
    def set_index_datetime(self, idx_column:str) -> None:
        """
        Set index of dataframe to datetime
        :param column: Column name to set as index
        :return: None
        """
        self.data[idx_column] = pd.to_datetime(self.data[idx_column], utc=True)
        self.data.set_index(idx_column, inplace=True)
    
    def remove_zero_variance(self) -> pd.DataFrame:
        """
        Remove columns with zero variance
        :return: Dataframe with columns with zero variance removed
        """
        self.data = self.data.loc[:, self.data.var() != 0]
        return self.data
    
    def clean_severe_missing(self, threshold:float=0.5) -> pd.DataFrame:
        """
        Remove columns with severe missing values
        :param threshold: Threshold for missing values
        :return: Dataframe with columns with severe missing values removed
        """
        self.data = self.data.loc[:, self.data.isnull().mean() < threshold]
        return self.data
    
    def label_encoding(self) -> pd.DataFrame:
        """
        Label encode all categorical columns
        :param column: Column name to be encoded
        :return: Encoded dataframe
        """
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype('category')
                self.data[col] = self.data[col].cat.codes
        return self.data
    
    def check_time_monotonicity(self, groupby:str=Optional) -> bool:
        """
        Check if the time series is monotonic
        :param groupby: Column name to groupby
        :return: True if monotonic, False if not
        """
        if groupby in self.data.columns:
            groups = self.data.groupby(groupby)
            return True if all(group.index.is_monotonic_increasing for _, group in groups) else False
        else:
            return self.data.index.is_monotonic_increasing
