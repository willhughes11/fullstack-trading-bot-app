import logging
import random
from typing import Union

import numpy as np
import pandas as pd
from pyhhmm.gaussian import GaussianHMM
from ta import add_trend_ta

logger = logging.getLogger()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


def engineer_features(data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
    df = data.copy()
    df["returns"] = (df["close"] / df["close"].shift(1)) - 1
    df["range"] = (df["close"] / df["low"]) - 1
    df.dropna(inplace=True)
    df_trend_ta = add_trend_ta(
        df.copy(), high="high", low="low", close="close", fillna=True
    )

    return df, df_trend_ta


def add_moving_averages(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df[f"ma_{window}"] = df["close"].rolling(window=window).mean()

    return df


def pick_features(df_trend_ta: pd.DataFrame, n_samples: int):
    features = ["returns", "range"]
    df_trends_only = df_trend_ta.copy()
    columns_to_remove = ["open", "high", "low", "close", "volume", "returns", "range"]
    df_trends_only.drop(columns=columns_to_remove, inplace=True)
    column_names = df_trends_only.columns.tolist()
    random_items = random.sample(column_names, n_samples - len(features))
    features.extend(random_items)

    return features


def train_and_test_split(
    df: pd.DataFrame,
    df_trend_ta: pd.DataFrame,
    train_split: int,
    test_split: int,
    indictators: list[str],
) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = df_trend_ta[indictators].iloc[:train_split]
    test_dataset = df.iloc[test_split:]
    X_test = df_trend_ta[indictators].iloc[test_split:]

    return X_train, test_dataset, X_test


def train_model(
    X_train: pd.DataFrame,
    n_states: int,
    covariance_type: str,
    n_emissions: int,
) -> GaussianHMM:
    print("-" * 100)
    print(X_train.columns, n_states, n_emissions)
    print("-" * 100)
    model = GaussianHMM(
        n_states=n_states, covariance_type=covariance_type, n_emissions=len(X_train.columns)
    )
    model.train([np.array(X_train.values)])

    return model


def setup_test(test_dataset: pd.DataFrame) -> pd.DataFrame:
    df_main = test_dataset.copy()
    df_main.drop(columns=["high", "low"], inplace=True)

    return df_main


def test_model(
    model: GaussianHMM, X_test: pd.DataFrame, df_main: pd.DataFrame
) -> pd.DataFrame:
    hmm_results = model.predict([X_test.values])[0]
    df_main["hmm"] = hmm_results

    return df_main


def add_ma_signals(
    df_main: pd.DataFrame, l_window_column: str, s_window_column: str
) -> pd.DataFrame:
    df_main.loc[df_main[s_window_column] > df_main[l_window_column], "ma_signal"] = 1
    df_main.loc[df_main[s_window_column] <= df_main[l_window_column], "ma_signal"] = 0

    return df_main


def add_hmm_signals(df_main: pd.DataFrame, favourable_states: list[int] = [0, 1]):
    hmm_values = df_main["hmm"].values
    hmm_values = [1 if x in favourable_states else 0 for x in hmm_values]
    df_main["hmm_signals"] = hmm_values

    return df_main


def add_combined_signals(df_main: pd.DataFrame):
    df_main["main_signal"] = 0
    df_main.loc[
        (df_main["ma_signal"] == 1) & (df_main["hmm_signals"] == 1), "main_signal"
    ] = 1
    df_main["main_signal"] = df_main["main_signal"].shift(1)

    return df_main


def calculate_pnl_drawdown(df: pd.DataFrame) -> Union[float, float]:
    df = df[df["main_signal"] != 0].copy()

    df["pnl"] = df["close"].pct_change(fill_method=None) * df["main_signal"].shift(1)

    df["cum_pnl"] = df["pnl"].cumsum()
    df["max_cum_pnl"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["max_cum_pnl"] - df["cum_pnl"]

    return df["pnl"].sum(), df["drawdown"].max()


def backtest(
    data: pd.DataFrame,
    train_split: int,
    test_split: int,
    ma_l_window: int = 21,
    ma_s_window: int = 12,
    n_states: int = 4,
    covariance_type: str = "full",
    n_emissions: int = 4,
):
    try:
        df, df_trend_ta = engineer_features(data)
        df = add_moving_averages(df, ma_l_window)
        df = add_moving_averages(df, ma_s_window)
        features = pick_features(df_trend_ta, n_states)
        X_train, test_dataset, X_test = train_and_test_split(
            df, df_trend_ta, train_split, test_split, features
        )
        model = train_model(X_train, n_states, covariance_type, n_emissions)
        df_main = setup_test(test_dataset)
        df_main = test_model(model, X_test, df_main)
        df_main = add_ma_signals(df_main, f"ma_{ma_l_window}", f"ma_{ma_s_window}")
        df_main = add_hmm_signals(df_main)
        df_main = add_combined_signals(df_main)
        pnl, drawdown = calculate_pnl_drawdown(df_main)
        return pnl, drawdown, features
    except Exception as e:
        logger.error(e)
        return 0, 0, []
