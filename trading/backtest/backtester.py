from ctypes import *

from database import Hdf5Client

from utils import resample_timeframe, STRAT_PARAMS
import strategies.hmm


def run(
    exchange: str, symbol: str, strategy: str, tf: str, from_time: int, to_time: int
):
    params_des = STRAT_PARAMS[strategy]

    params = dict()

    for p_code, p in params_des.items():
        while True:
            try:
                params[p_code] = p["type"](input(p["name"] + ": "))
                break
            except ValueError:
                continue

    if strategy == "hmm":
        h5_db = Hdf5Client(exchange)
        data = h5_db.get_data(symbol, from_time, to_time)
        data = resample_timeframe(data, tf)

        data_len = len(data)
        train_split = int(data_len / 2)
        test_split = data_len - train_split

        pnl, max_drawdown = strategies.hmm.backtest(
            data,
            train_split,
            test_split,
            ma_l_window=params["ma_l_window"],
            ma_s_window=params["ma_s_window"],
            n_states=params["n_states"],
            covariance_type="full",
            n_emissions=params["n_emissions"]
        )

        return pnl, max_drawdown
