from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class TimeSeriesCrossValidator:
    def __init__(self, model, X, y, n_splits=5):
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def validate(self):
        scores = []
        for train_index, test_index in self.tscv.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            score = mean_squared_error(y_test, y_pred)  
            scores.append(score)

        return scores