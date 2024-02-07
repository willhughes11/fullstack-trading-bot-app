import yfinance as yfin
from pandas_datareader import data as pdr
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols


def get_stock_data(start_date: str, end_date: str):
    yfin.pdr_override()
    symbols = get_nasdaq_symbols()
    symbols = symbols[
        (symbols["ETF"] == True) & (symbols["Market Category"] == "G")
    ]  # G = NASDAQ GLOBAL MARKET
    symbols = list(symbols.index.values)
    data = pdr.get_data_yahoo(symbols, start=start_date, end=end_date)["Adj Close"]
    return data
