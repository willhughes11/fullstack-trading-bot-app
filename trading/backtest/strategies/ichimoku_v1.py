import numpy as np
import pandas as pd
import mplfinance as mpf
from pandas import DataFrame, Series


class IchimokuCloud:
    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def get_tenkan_sen(self):
        window = 9
        tenkan_high = self.df["High"].rolling(window).max()
        tenkan_low = self.df["Low"].rolling(window).min()
        tenkan_sen = Series((tenkan_high + tenkan_low) / 2)

        return tenkan_sen

    def get_kijun_sen(self, window: int = 26):
        kijun_high = self.df["High"].rolling(window).max()
        kijun_low = self.df["Low"].rolling(window).min()
        kijun_sen = Series((kijun_high + kijun_low) / 2)

        return kijun_sen

    def get_senkou_span_a(self, df: DataFrame):
        senkou_span_a = Series(((df["tenkan_sen"] + df["kijun_sen"]) / 2))

        return senkou_span_a

    def get_senkou_span_b(self, window: int = 52):
        senkou_b_high = self.df["High"].rolling(window).max()
        senkou_b_low = self.df["Low"].rolling(window).min()
        senkou_span_b = Series(((senkou_b_high + senkou_b_low) / 2))

        return senkou_span_b

    def get_chikou_span(self, ahead: int = -24):
        chikou_span = Series(self.df["Close"].shift(ahead))

        return chikou_span

    def get_ichimoku_df(self, shift: bool = False) -> DataFrame:
        selected_columns = ["Datetime", "Open", "High", "Low", "Close"]
        new_df = self.df[selected_columns]
        indicator_df = DataFrame(new_df)
        indicator_df["tenkan_sen"] = self.get_tenkan_sen()
        indicator_df["kijun_sen"] = self.get_kijun_sen()
        indicator_df["senkou_span_a"] = self.get_senkou_span_a(indicator_df)
        indicator_df["senkou_span_b"] = self.get_senkou_span_b()
        indicator_df["chikou_span"] = self.get_chikou_span()

        if shift:
            return self.shift_senkou_clouds(indicator_df)

        return indicator_df

    def shift_senkou_clouds(self, df: DataFrame, shift_periods: int = 26):
        columns_to_shift = ["senkou_span_a", "senkou_span_b"]

        # Create a new DataFrame with NaN values to hold the shifted columns
        shifted_df = pd.DataFrame(
            np.nan, index=range(shift_periods), columns=columns_to_shift
        )

        # Concatenate the original DataFrame with the DataFrame containing NaN values
        df_extended = pd.concat([df, shifted_df], ignore_index=True)

        # Shift the specified columns
        df_extended[columns_to_shift] = df_extended[columns_to_shift].shift(
            periods=shift_periods
        )

        df_extended.fillna(np.nan, inplace=True)

        df_extended["Datetime"] = pd.to_datetime(df_extended["Datetime"])
        date_range = pd.date_range(
            start=df_extended["Datetime"].min(),
            end=df_extended["Datetime"].max(),
            freq="H",
        )
        df_extended["Datetime"] = df_extended["Datetime"].fillna(pd.Series(date_range))

        return df_extended

    def kumo_cloud_plot(self, df: DataFrame):
        plot_df = df.set_index("Datetime")
        lead_span = [
            mpf.make_addplot(plot_df["tenkan_sen"], label="tenkan_sen", color="red"),
            mpf.make_addplot(plot_df["kijun_sen"], label="kijun_sen", color="blue"),
            mpf.make_addplot(
                plot_df["chikou_span"], label="chikou_span", color="purple"
            ),
            mpf.make_addplot(
                plot_df["senkou_span_a"], label="senkou_span_a", color="green"
            ),
            mpf.make_addplot(
                plot_df["senkou_span_b"], label="senkou_span_b", color="orange"
            ),
        ]

        mpf.plot(
            plot_df,
            addplot=lead_span,
            type="candle",
            fill_between=dict(
                y1=plot_df["senkou_span_a"].values, y2=plot_df["senkou_span_b"].values
            ),
        )

    def price_senkou_span_cross(self, df: DataFrame):
        last_price_index = df["Close"].last_valid_index()
        open = df.iloc[last_price_index]["Open"]
        high = df.iloc[last_price_index]["High"]
        low = df.iloc[last_price_index]["Low"]
        close = df.iloc[last_price_index]["Close"]
        senkou_span_a = df.iloc[-1]["senkou_span_a"]
        senkou_span_b = df.iloc[-1]["senkou_span_b"]

        candle = [open, high, low, close]
        senkou_span = [senkou_span_a, senkou_span_b]

        if all(price > max(senkou_span) for price in candle):
            return 1
        elif all(price < min(senkou_span) for price in candle):
            return -1
        else:
            return 0

    def kijun_sen_direction(self, df: DataFrame, window: int = 26):
        last_kijun_sen_index = df["kijun_sen"].last_valid_index()
        if last_kijun_sen_index != None:
            kijun_sen = df.iloc[
                max(0, last_kijun_sen_index - window) : last_kijun_sen_index + 1
            ]["kijun_sen"]
            data = np.array(kijun_sen.tolist())
            slope_linreg = np.polyfit(range(len(data)), data, 1)[0]
            if slope_linreg > 0:
                return 1
            elif slope_linreg < 0:
                return -1
            else:
                return 0
        else:
            return 0

    def senkou_span_cloud_color(self, df: DataFrame):
        if df["senkou_span_a"].iloc[-1] > df["senkou_span_b"].iloc[-1]:
            return 1, 5
        elif df["senkou_span_a"].iloc[-1] < df["senkou_span_b"].iloc[-1]:
            return -1, 5
        else:
            return 0, 0

    def price_kijun_sen_cross(self, df: DataFrame):
        last_price_index = df["Close"].last_valid_index()
        if last_price_index != None:
            close = df.iloc[last_price_index]["Close"]
            kijun_sen = df.iloc[last_price_index]["kijun_sen"]

            if close > kijun_sen:
                return 1, 5
            elif close < kijun_sen:
                return -1, 5
            else:
                return 0, 0
        else:
            return 0, 0

    def tenkan_sen_kijun_sen_cross(self, df: DataFrame):
        last_price_index = df["Close"].last_valid_index()
        if last_price_index != None:
            tenkan_sen = df.iloc[last_price_index]["tenkan_sen"]
            kijun_sen = df.iloc[last_price_index]["kijun_sen"]

            if tenkan_sen > kijun_sen:
                return 1, 4
            elif tenkan_sen < kijun_sen:
                return -1, 4
            else:
                return 0, 0
        else:
            return 0, 0

    def price_chikou_span_cross(self, df: DataFrame):
        last_chikou_span_index = df["chikou_span"].last_valid_index()
        if last_chikou_span_index != None:
            close = df.iloc[last_chikou_span_index]["Close"]
            chikou_span = df.iloc[last_chikou_span_index]["chikou_span"]

            if close > chikou_span:
                return 1, 4
            elif close < chikou_span:
                return -1, 4
            else:
                return 0, 0
        else:
            return 0, 0
