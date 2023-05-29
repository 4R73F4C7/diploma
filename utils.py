import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import talib as ta

import config


def draw_collection(df: pd.DataFrame, collection: str, type: str = "actual", show: bool = True,
                    save: bool = True) -> None:
    """
    Draw a collection's floor price data with actions (buy/sell) indicated on the plot.

    Args:
        df (pd.DataFrame): DataFrame containing the floor price data.
        collection (str): Name of the NFT collection.
        type (str, optional): Type of data to draw (actual or predicted). Defaults to "actual".
        show (bool, optional): Whether to display the plot. Defaults to True.
        save (bool, optional): Whether to save the plot as an image. Defaults to True.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    if type != "predicted":
        replace_start_index = int(len(df) * config.TRAIN_SIZE)
        remove_end_index = replace_start_index + config.LOOKBACK
        df.drop(range(replace_start_index, remove_end_index), inplace=True)
        df = df[:-config.LOOKBACK]

    ax.plot(df['TimeStamp'], df['FloorEthPrice'])

    df = df[df['action'] != 0]

    color_map = {
        -1: 'red',
        1: 'green',
    }

    if type != "prices":
        ax.scatter(df['TimeStamp'], df['FloorEthPrice'], c=df['action'].map(color_map), s=10)

    if type == "predicted":
        delta = pd.Timedelta(days=10)
        prediction_start = df[df['Prediction_Start']]['TimeStamp']
        for timestamp in prediction_start:
            ax.axvline(x=timestamp, color='blue', linestyle='--')
            ax.text(timestamp - delta, 0.2, 'Prediction Start', rotation=90,
                    transform=ax.get_xaxis_text1_transform(0)[0], color='blue')

    ax.set_title(f'{collection.capitalize().replace("-", "")} Calls [{type.capitalize()}]')
    ax.set_xlabel('Date')
    ax.set_ylabel('Floor Price (ETH)')

    legend_elements = [
        plt.Line2D([0], [0], linestyle='-', label='Floor Price')
    ]

    if type != "prices":
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Sell'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=5, label='Buy'))

    if type == "predicted":
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='blue', label='Prediction Start'))

    ax.legend(handles=legend_elements)

    if save:
        plt.savefig(f'data/analysis/visualizations/{type}/{collection}.png')

    if show:
        plt.show()

    plt.close()


def add_indicators_and_decisions(df: pd.DataFrame, column: str = "FloorEthPrice", lower_iqr: float = 0.5,
                                 upper_iqr: float = 2, sma_period: int = 20, rsi_period: int = 5,
                                 rsi_threshold: float = 50) -> pd.DataFrame:
    """
    Add technical indicators and decision signals to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the floor price data.
        column (str, optional): Name of the column to calculate indicators on. Defaults to "FloorEthPrice".
        lower_iqr (float, optional): Lower IQR multiplier for outlier removal. Defaults to 0.5.
        upper_iqr (float, optional): Upper IQR multiplier for outlier removal. Defaults to 2.
        sma_period (int, optional): Simple Moving Average period. Defaults to 20.
        rsi_period (int, optional): RSI period. Defaults to 5.
        rsi_threshold (float, optional): RSI threshold for buy/sell decisions. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame with added indicators and decision signals.
    """

    Q1 = df['FloorEthPrice'].quantile(0.25)
    Q3 = df['FloorEthPrice'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - lower_iqr * IQR
    upper_bound = Q3 + upper_iqr * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df['SMA'] = ta.SMA(df[column], timeperiod=sma_period)
    df['RSI'] = ta.RSI(df[column], timeperiod=rsi_period)

    df['buy_signal'] = np.where((df['RSI'] > rsi_threshold) & (df['FloorEthPrice'] > df['SMA']), 1, 0)
    df['sell_signal'] = np.where((df['RSI'] < rsi_threshold) & (df['FloorEthPrice'] < df['SMA']), 1, 0)

    df = df.dropna()

    df['action'] = df['sell_signal'] - df['buy_signal']

    return df
