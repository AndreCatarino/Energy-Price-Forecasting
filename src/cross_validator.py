from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deeplearning_build import deepL
from sklearn.model_selection import train_test_split

class TimeSeriesCrossValidator:
    def __init__(self, model_name:str, df:pd.DataFrame, n_splits:int=5) -> None:
        """
        Cross validate a model using TimeSeriesSplit
        :param model_name: name of the model to be used
        :param df: dataframe with the data
        :param n_splits: number of splits to be used in TimeSeriesSplit
        """
        self.model_name = model_name
        self.dl = None
        self.data = df
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def validate(self) -> tuple:
        """
        Validate the model using TimeSeriesSplit
        """
        self.dl = deepL()
        
        mae_scores = []
        mse_scores = []
        rmse_scores = []
        r2_scores = []

        for train_index, test_index in self.tscv.split(self.data):
            
            df_train, df_test = self.data.iloc[train_index], self.data.iloc[test_index]
            # Split train into train and validation
            df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=False)

            self.dl.input_data(df_train, df_val)

            self.dl.train(self.model_name, plot_history=True)
            mae, mse, rmse, r2 = self.dl.evaluate(self.model_name, df_test)

            mae_scores.append(mae)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        return mae_scores, mse_scores, rmse_scores, r2_scores
    