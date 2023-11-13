import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tcn import TCN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class deepL():
    def __init__(self) -> None:
        """
        Initialize the class
        """
        self.models = {
            "LSTM": self.lstm_model,
            "GRU": self.gru_model,
            "BiLSTM": self.bilstm_model,
            "CNN-LSTM": self.cnn_lstm_model,
            "CNN-GRU": self.cnn_gru_model,
            "CNN-BiLSTM": self.cnn_bilstm_model,
            "CNN-BiLSTM-Attention": self.cnn_bilstm_attention_model,
            "TCN": self.tcn_model,
        }
        self.train_set = None
        self.valid_set = None
        self.trained_models = {}

    def input_data(self, X_train:pd.DataFrame, X_val:pd.DataFrame, window_size:int=15) -> None:
        """
        Prepare input data
        :param X_train: Training data
        :param X_val: Validation data
        :param window_size: Size of the window
        """
        self.n_features = X_train.shape[-1]
        self.train_set = self.prepare_sequential_window(X_train, window_size, classification=False)
        self.valid_set = self.prepare_sequential_window(X_val, window_size, classification=False)

    def train(self, model_name:str, plot_history:bool=False) -> keras.models.Sequential:
        """
        Train the model
        :param model_name: Name of the model
        :param X_train: Training data
        :param X_val: Validation data
        :return: Trained model
        """
        model = self.models.get(model_name)
        if model is not None:
            return self.models.get(model_name)(plot_history)
        elif self.train_set is None and self.valid_set is None:
            raise ValueError("Input data first.")
        else:
            raise ValueError("Model not supported.")
        
    def predict(self, model_name:str, X_test:pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable
        :param model_name: Name of the model
        :param X_test: Test data
        :return: Predictions
        """
        model = self.trained_models.get(model_name)
        if model is not None:
            y_pred = model.predict(self.prepare_sequential_window(X_test, window_size=15, classification=False))
            y_pred = y_pred.reshape(-1, 1)
            return y_pred
        else:
            raise ValueError("Model not trained yet.")
        
    def evaluate(self, model_name:str, X_test:pd.DataFrame, plt_pred:bool=False) -> tuple:
        """
        Evaluate the model
        :param model_name: Name of the model
        :return: Evaluation metrics
        """
        model = self.trained_models.get(model_name)
        if model is not None:
            test_set = self.prepare_sequential_window(X_test, window_size=15, classification=False)
            y_pred = model.predict(test_set)
            y_pred = y_pred.reshape(-1, 1)

            y_batch_list = []
            for _, y_batch in test_set:
                y_batch_list.append(y_batch.numpy())

            y_batch_list = np.array(y_batch_list)
            y_batch_list = y_batch_list.reshape(-1, 1)

            # load scaler
            scaler = joblib.load("../artifacts/scaler.pkl")
            pred = scaler.inverse_transform(y_pred)
            original_target = scaler.inverse_transform(y_batch_list)

            mae = mean_absolute_error(original_target, pred)
            mse = mean_squared_error(original_target, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(original_target, pred)

            if plt_pred:
                plt = self.plot_predictions(model_name, pred, original_target)
                return mae, mse, rmse, r2, plt

            return mae, mse, rmse, r2

        else:
            raise ValueError("Invalid model name.")
        
    def load_model(self, model_name:str) -> keras.models.Sequential:
        """
        Load the model
        :param model_name: Name of the model
        :return: Loaded model
        """
        if model_name == "TCN":
            with keras.utils.custom_object_scope({'TCN': TCN}):
                model = keras.models.load_model("../artifacts/{}.h5".format(model_name))
        else:
            model = keras.models.load_model("../artifacts/{}.h5".format(model_name))
        self.trained_models[model_name] = model
        return model

    def prepare_sequential_window(self, series:pd.DataFrame, window_size:int, classification:bool=False) -> tf.data.Dataset:
        """
        Prepare sequential window for time series data
        :param series: Time series data
        :param window_size: Size of the window
        :param classification: Whether the problem is classification or not
        :return: Sequential window
        """
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
        ds = ds.flat_map(lambda window: window.batch(window_size + 1))
        if classification:
            ds = ds.map(
                lambda window: (
                    window[:-1],
                    tf.expand_dims(
                        tf.where(window[1:, -1] > window[:-1, -1], 1.0, 0.0), axis=-1
                    ),
                )
            )
        else:
            ds = ds.map(
                lambda window: (window[:-1], tf.expand_dims(window[1:, -1], axis=-1))
            )
        return ds.batch(1).prefetch(1)
    
    def plot_history(self, model_name:str, history:keras.callbacks.History) -> None:
        """
        save training and validation loss plot without actually plotting it
        :param history: History of the model
        """
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='train')
        ax.plot(history.history['val_loss'], label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('{} Training and Validation Loss'.format(model_name))
        ax.legend()
        fig.savefig("../plots/{}_history.png".format(model_name))
        plt.close(fig)
    
    def plot_predictions(self, model_name:str, y_pred:np.ndarray, y_test:np.ndarray) -> plt:
        """
        Plot predictions
        :param model_name: Name of the model
        :param y_pred: Predictions
        :param y_test: Actual values
        :return: Figure
        """
        plt.figure(figsize=(16, 10))
        plt.plot(y_pred, label="predictions")
        plt.plot(y_test, label="actual")
        plt.legend()
        plt.title("{} Predictions vs Actuals".format(model_name))
        plt.savefig("../plots/{}_predictions.png".format(model_name))
        return plt

    def lstm_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build LSTM model
        :param plot_history: Whether to plot the training and validation loss
        :return: LSTM model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
        [
            keras.layers.LSTM(
                100,
                return_sequences=True,
                stateful=True,
                batch_input_shape=[1, None, self.n_features],
            ),
            keras.layers.LSTM(100, return_sequences=True, stateful=True),
            keras.layers.LSTM(100, return_sequences=True, stateful=True),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear"),
        ]   
        )

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/lstm.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["LSTM"] = model

        if plot_history:
            self.plot_history("LSTM" , history)

        return model
    
    def gru_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build GRU model
        :param plot_history: Whether to plot the training and validation loss
        :return: GRU model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
        [
            keras.layers.GRU(
                100,
                return_sequences=True,
                stateful=True,
                batch_input_shape=[1, None, self.n_features],
            ),
            keras.layers.GRU(100, return_sequences=True, stateful=True),
            keras.layers.GRU(100, return_sequences=True, stateful=True),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear"),
        ]   
        )

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/GRU.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["GRU"] = model

        if plot_history:
            self.plot_history("GRU" , history)

        return model    

    def bilstm_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build BiLSTM model
        :param plot_history: Whether to plot the training and validation loss
        :return: BiLSTM model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential()
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(100, return_sequences=True, stateful=True), batch_input_shape=[1, None, self.n_features]
            )
        )
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(100, return_sequences=True, stateful=True)
            )
        )
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(100, return_sequences=True, stateful=True)
            )
        )
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=1, activation="linear"))

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/BiLSTM.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["BiLSTM"] = model

        if plot_history:
            self.plot_history("BiLSTM" , history)

        return model

    def cnn_lstm_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build CNN-LSTM model
        :param plot_history: Whether to plot the training and validation loss
        :return: CNN-LSTM model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
        [
        # 1D-Conv layer will slide filters across one-dimension (time axis) of the input; kernel:filter size
        keras.layers.Conv1D(
            filters=20,
            kernel_size=4,
            strides=1,
            padding="causal",
            activation="relu",
            batch_input_shape=[1,None, self.n_features],
        ),
        #keras.layers.MaxPooling1D(pool_size=2), # downsample the input representation by taking the maximum value over the window defined by pool_size
        keras.layers.LSTM(100, return_sequences=True, stateful=True),
        keras.layers.LSTM(100, return_sequences=True, stateful=True),
        # dropout layer to prevent overfitting
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="linear"),
        ])

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/cnn-lstm.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["CNN-LSTM"] = model

        if plot_history:
            self.plot_history("cnn-lstm" , history)

        return model    

    def cnn_gru_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build CNN-GRU model
        :param plot_history: Whether to plot the training and validation loss
        :return: CNN-GRU model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
        [
        keras.layers.Conv1D(
            filters=20,
            kernel_size=4,
            strides=1,
            padding="causal",
            activation="relu",
            batch_input_shape=[1,None, self.n_features],
        ),
        keras.layers.GRU(100, return_sequences=True, stateful=True),
        keras.layers.GRU(100, return_sequences=True, stateful=True),
        # dropout layer to prevent overfitting
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="linear"),
        ])

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/CNN-GRU.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["CNN-GRU"] = model

        if plot_history:
            self.plot_history("CNN-GRU", history)

        return model

    def cnn_bilstm_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build CNN-BiLSTM model 
        :param plot_history: Whether to plot the training and validation loss
        :return: CNN-BiLSTM model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
    [
        keras.layers.Conv1D(
            filters=20,
            kernel_size=4,
            strides=1,
            padding="causal",
            activation="relu",
            batch_input_shape=[1,None, self.n_features],
        ),
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
        # dropout layer to prevent overfitting
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="linear"),
    ])

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/CNN-BiLSTM.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["CNN-BiLSTM"] = model

        if plot_history:
            self.plot_history("CNN-BiLSTM", history)

        return model   

    def cnn_bilstm_attention_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build CNN-BiLSTM-Attention model
        :param plot_history: Whether to plot the training and validation loss
        :return: CNN-BiLSTM-Attention model
        """
        keras.backend.clear_session()
        
        # attention layer
        class Attention(keras.layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.W1 = keras.layers.Dense(units)
                self.W2 = keras.layers.Dense(units)
                self.V = keras.layers.Dense(1)

            def call(self, features):
                score = tf.nn.tanh(self.W1(features) + self.W2(features))
                attention_weights = tf.nn.softmax(self.V(score), axis=1)
                context_vector = attention_weights * features
                context_vector = tf.reduce_sum(context_vector, axis=1)
                return context_vector, attention_weights

        # Input layer
        input_layer = keras.layers.Input(shape=(None, self.n_features))

        # CNN-BiLSTM-Attention model
        x = keras.layers.Conv1D(
            filters=20,
            kernel_size=4,
            strides=1,
            padding="causal",
            activation="relu"
        )(input_layer)

        x = keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True))(x)
        # add dropout to control overfitting
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True))(x)

        context_vector, attention_weights = Attention(100)(x)
        output = keras.layers.Dense(1, activation="linear")(x)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/CNN-BiLSTM-Attention.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["CNN-BiLSTM-Attention"] = model

        if plot_history:
            self.plot_history("CNN-BiLSTM-Attention",history)

        return model   

    def tcn_model(self, plot_history:bool) -> keras.models.Sequential:
        """
        Build TCN model
        :param plot_history: Whether to plot the training and validation loss
        :return: TCN model
        """
        keras.backend.clear_session()
        
        model = keras.models.Sequential(
    [
        TCN(
            batch_input_shape=(1, None, self.n_features),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            padding='causal',
            use_skip_connections=True,
            dropout_rate=0.2,
            return_sequences=True,
            use_batch_norm=False,
            use_weight_norm=False,
            use_layer_norm=False,
        ),
        keras.layers.Dense(1, activation="linear"),
    ])
        
        model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae", "mape"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
        )
        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            mode="min",
            verbose=1,
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            "../artifacts/tcn.h5", save_best_only=True, monitor="val_loss", mode="min"
        )

        history = model.fit(
            self.train_set,
            epochs=50,
            batch_size=32,
            validation_data=self.valid_set,
            verbose=1,
            shuffle=False,
            callbacks=[early_stopping, lr_schedule, checkpoint]
        )

        self.trained_models["TCN"] = model

        if plot_history:
            self.plot_history("TCN",history)

        return model   
