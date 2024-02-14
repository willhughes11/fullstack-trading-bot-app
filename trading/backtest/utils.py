import datetime
from ctypes import *

import pandas as pd

TF_EQUIV = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1H",
    "4h": "4H",
    "12h": "12H",
    "1d": "D",
}

STRAT_PARAMS = {
    "hmm": {
        "ma_l_window": {"name": "MA Large Window", "type": int, "min": 20, "max": 200},
        "ma_s_window": {"name": "MA Small Window", "type": int, "min": 12, "max": 100},
        "n_states": {"name": "N States", "type": int, "min": 2, "max": 5},
    },
}


def ms_to_dt(ms: int) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ms / 1000)


def resample_timeframe(data: pd.DataFrame, tf: str) -> pd.DataFrame:
    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )