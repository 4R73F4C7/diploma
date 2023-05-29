import math
from typing import Optional, List
import pandas as pd
import numpy as np
from keras.layers import GRU
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
from keras import callbacks
import json
import config
import utils

RANDOM_STATE = 228


class Model:
    """
    Base class for different models.
    """

    def __init__(self, df, filters=[]):
        """
        Initializes the Model instance.

        Args:
        - df: DataFrame containing the data.
        - filters: List of column names to filter the data.
        """
        self.data = df[filters]
        self.filters = filters
        self.X_train = None
        self.y_train = None
        self.X_Test = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.test_mae = 0
        self.test_rmse = 0
        self.train_mae = 0
        self.train_rmse = 0
        self.train_predict = None
        self.test_predict = None

    def fit(self):
        """
        Placeholder method for fitting the model.
        """
        pass

    def process_data(self, lookback, train_size, scaler=StandardScaler):
        """
        Preprocesses the data for training and testing.

        Args:
        - lookback: Number of time steps to look back.
        - train_size: Proportion of the data to be used for training.
        - scaler: Scaler object used for scaling the data.
        """
        data = self.data.copy()

        train_data, test_data = train_test_split(data, test_size=1 - train_size, shuffle=False)
        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        X_train, y_train = [], []
        for i in range(len(train_data) - lookback):
            X_train.append(train_data[i:i + lookback])
            y_train.append(train_data[i + lookback, 0])
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = [], []
        for i in range(len(test_data) - lookback):
            X_test.append(test_data[i:i + lookback])
            y_test.append(test_data[i + lookback, 0])
        self.X_Test, self.y_test = np.array(X_test), np.array(y_test)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], len(self.filters)))
        self.X_Test = self.X_Test.reshape((self.X_Test.shape[0], self.X_Test.shape[1], len(self.filters)))

    def draw_train_graph(self, collection=None, ax=None):
        """
        Draws the training graph.

        Args:
        - collection: Collection name (optional).
        - ax: Matplotlib axis object (optional).
        """
        Xt = self.model.predict(self.X_train)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        if len(self.filters) > 1:
            df_actual_keys += ['null_val']
        df_actual_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        df_actual_data = {"Actual": self.y_train}
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_train)

        df_predicted_keys = ["Predicted"]

        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')

        df_predicted_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        df_predicted_data = {"Predicted": Xt}
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(df_actual[df_actual_keys])

        df_predicted = pd.DataFrame(df_predicted_data)
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(df_predicted[df_predicted_keys])
        self.train_predict = df_predicted.Predicted

        ax.plot(df_actual.Actual, label="Actual")
        ax.plot(self.train_predict, label="Predicted")
        ax.legend()

        ax.set_title(f"Train Dataset [{self.__class__.__name__}] [{self.filters}] [{collection}]")
        self.train_rmse = math.sqrt(mean_squared_error(df_actual.Actual, self.train_predict))
        self.train_mae = mean_absolute_error(df_actual.Actual, self.train_predict)
        print(f"[{self.filters}] [{self.__class__.__name__}] Train RMSE =", self.train_rmse)
        print(f"[{self.filters}] [{self.__class__.__name__}] Train MAE =", self.train_mae)

    def draw_test_graph(self, collection=None, ax=None):
        """
        Draws the testing graph.

        Args:
        - collection: Collection name (optional).
        - ax: Matplotlib axis object (optional).
        """
        Xt = self.model.predict(self.X_Test)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        df_actual_data = {"Actual": self.y_test}
        if len(self.filters) > 1:
            df_actual_keys.append('null_val')
        df_actual_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_test)

        df_predicted_keys = ["Predicted"]
        df_predicted_data = {"Predicted": Xt}

        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')
        df_predicted_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(df_actual[df_actual_keys])

        df_predicted = pd.DataFrame(df_predicted_data)
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(df_predicted[df_predicted_keys])
        self.test_predict = df_predicted.Predicted
        ax.plot(df_actual.Actual, label="Actual")
        ax.plot(self.test_predict, label="Predicted")
        ax.legend()

        ax.set_title(f"Test Dataset [{self.__class__.__name__}] [{self.filters}] [{collection}]")
        self.test_rmse = math.sqrt(mean_squared_error(df_actual.Actual, self.test_predict))
        self.test_mae = mean_absolute_error(df_actual.Actual, self.test_predict)
        print(f"[{self.filters}] [{self.__class__.__name__}] Test RMSE =", self.test_rmse)
        print(f"[{self.filters}] [{self.__class__.__name__}] Test MAE =", self.test_mae)

    def plot_multiple_graphs(self, collection=None, show: bool = False):
        """
        Plots multiple graphs.

        Args:
        - collection: Collection name (optional).
        - show: Whether to display the plot or not (optional).
        """
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        self.draw_train_graph(collection, axs[0])
        self.draw_test_graph(collection, axs[1])

        plt.tight_layout()
        plt.savefig(f"data/analysis/graphs/{collection}_{self.__class__.__name__}_{'_'.join(self.filters)}.png")
        if show:
            plt.show()
        plt.close()

    def export(self, metric_params: Optional[List] = None):
        """
        Exports the model's metrics and predictions.

        Args:
        - metric_params: List of metric parameters to export (optional).

        Returns:
        - result: Dictionary containing the exported metrics and predictions.
        """
        if not metric_params:
            result = {
                "test_rmse": float(self.test_rmse),
                "test_mae": float(self.test_mae),
                "train_rmse": float(self.train_rmse),
                "train_mae": float(self.train_mae),
                "test_predict": list(map(float, self.test_predict.tolist())),
            }
            return result
        result = {}
        for metric_param in metric_params:
            result[metric_param] = getattr(self, metric_param)
        return result


class LSTM_MODEL(Model):
    def __init__(self, df, filters):
        """
        Initializes an LSTM_MODEL object.

        Args:
            df: The input dataframe.
            filters: List of filters to apply on the dataframe.
        """
        super().__init__(df, filters=filters)

    def fit(self, lookback=config.LOOKBACK, train_size=config.TRAIN_SIZE, scaler=StandardScaler, epochs=300):
        """
        Fits the LSTM model to the data.

        Args:
            lookback: Number of previous time steps to consider for making predictions.
            train_size: Percentage of data to use for training.
            scaler: Scaler to preprocess the data.
            epochs: Number of epochs for training.

        Note:
            This function internally processes the data, initializes the model, compiles it,
            and fits it to the training data.

        Returns:
            None
        """
        self.process_data(lookback, train_size, scaler)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.model = Sequential()

        self.model.add(LSTM(units=256, input_shape=(lookback, len(self.filters))))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mse')

        earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                patience=25, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                      validation_data=(self.X_Test, self.y_test),
                                      shuffle=False, callbacks=[earlystopping])


class GRU_MODEL(Model):
    def __init__(self, df, filters):
        """
        Initializes a GRU_MODEL object.

        Args:
            df: The input dataframe.
            filters: List of filters to apply on the dataframe.
        """
        super().__init__(df, filters=filters)

    def fit(self, lookback=config.LOOKBACK, train_size=config.TRAIN_SIZE, scaler=StandardScaler, epochs=300):
        """
        Fits the GRU model to the data.

        Args:
            lookback: Number of previous time steps to consider for making predictions.
            train_size: Percentage of data to use for training.
            scaler: Scaler to preprocess the data.
            epochs: Number of epochs for training.

        Note:
            This function internally processes the data, initializes the model, compiles it,
            and fits it to the training data.

        Returns:
            None
        """
        self.process_data(lookback, train_size, scaler)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.model = Sequential()
        self.model.add(GRU(units=256, input_shape=(lookback, len(self.filters))))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                patience=25, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                      validation_data=(self.X_Test, self.y_test),
                                      shuffle=False, callbacks=[earlystopping])


class CollectionAnalyzer:
    def __init__(self, collection: str = "mutant-ape-yacht-club", df: pd.DataFrame = None):
        """
        Initializes a CollectionAnalyzer object.

        Args:
            collection: The name of the collection.
            df: The input DataFrame.

        Note:
            If `df` is not provided, it can be set later using the `df` attribute.
        """
        self.collection = collection
        self.df = df


def main():
    """
    Main function that performs the analysis.

    Note:
        This function reads data from a CSV file, performs analysis on different collections,
        and saves the results and models.
    """
    analyzers = []
    data_df = pd.read_csv('data/data.csv', parse_dates=True)
    dfs = data_df.groupby(by='Collection')
    for collection, df in dfs:
        print("Collection Name: ", collection)
        df_calls = utils.add_indicators_and_decisions(df, "FloorEthPrice")
        df_calls.to_csv(f"data/analysis/actions/actual/{collection}.csv", index=False)
        analyzers.append(CollectionAnalyzer(f"{collection}", df))
    params = [
        ['FloorEthPrice'],
        ['FloorEthPrice', 'VolumeInEth'],
        ['FloorEthPrice', 'Sales'],
        ['FloorEthPrice', 'VolumeInEth', 'Sales'],
    ]
    results_models = {}
    try:
        with open("data/analysis/results.json", 'r', encoding="utf-8") as f:
            export_data = json.loads(f.read())
    except:
        export_data = {}
    models = [LSTM_MODEL, GRU_MODEL]
    for analyzer in analyzers:
        results_models[analyzer.collection] = {}
        if analyzer.collection not in export_data.keys():
            export_data[analyzer.collection] = {}
        for param in params:
            current_param = '_'.join(param)
            results_models[analyzer.collection][current_param] = {}

            if current_param not in export_data[analyzer.collection].keys():
                export_data[analyzer.collection][current_param] = {}
            results_models[analyzer.collection][current_param]['params'] = param[:]
            results_models[analyzer.collection][current_param]['models'] = {}
            export_data[analyzer.collection][current_param]['params'] = param[:]

            if 'models' not in export_data[analyzer.collection][current_param].keys():
                export_data[analyzer.collection][current_param]['models'] = {}
            for model in models:
                print(model.__name__)
                results_models[analyzer.collection][current_param]['models'][model.__name__] = {
                    "model": model(analyzer.df.copy(), param)
                }
                results_models[analyzer.collection][current_param]['models'][model.__name__]['model'].fit(
                    scaler=MaxAbsScaler,
                    epochs=300)
                results_models[analyzer.collection][current_param]['models'][model.__name__][
                    'model'].plot_multiple_graphs(
                    analyzer.collection)
                export_data[analyzer.collection][current_param]['models'][model.__name__] = {
                    'stats': results_models[analyzer.collection][current_param]['models'][model.__name__][
                        'model'].export()
                }
                if model.__name__ != "KNNModel":
                    results_models[analyzer.collection][current_param]['models'][model.__name__][
                        'model'].model.save(
                        f"data/analysis/models/{analyzer.collection}_{model.__name__}_{current_param}.h5")
        with open(f"data/analysis/results/{analyzer.collection}.json", 'w', encoding="utf-8") as f:
            f.write(json.dumps(export_data))
        results_models[analyzer.collection] = {}
    with open("data/analysis/results.json", 'w', encoding="utf-8") as f:
        data = export_data.copy()
        f.write(json.dumps(export_data))


if __name__ == '__main__':
    main()
