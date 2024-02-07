import os
from datetime import datetime, timedelta

import pandas as pd


def df_to_csv(data: pd.Series, file_path: str):
    return data.to_csv(file_path)


def is_file_recent(file_path: str, hours: int = 24):
    current_time = datetime.now()

    file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))

    time_difference = current_time - file_creation_time

    return time_difference.total_seconds() <= hours * 3600


def does_file_exist(file_path: str):
    if os.path.exists(file_path):
        if is_file_recent(file_path, hours=96):
            return True
        else:
            return False
    else:
        return False
