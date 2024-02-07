import warnings
from datetime import datetime, timedelta

import pandas as pd
from data_sources.yahoo_finance import get_stock_data
from db import Base, engine
from engineered_stats.pairs import find_pairs
from utility_functions import df_to_csv, does_file_exist

warnings.simplefilter(action='ignore', category=FutureWarning)


def main() -> None:
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)
    end_date = today.strftime("%Y-%m-%d")
    start_date = thirty_days_ago.strftime("%Y-%m-%d")

    file_path = "data/raw_data_etf.csv"
    if does_file_exist(file_path):
        stocks = pd.read_csv(file_path)
    else:
        stocks = get_stock_data(start_date, end_date)
        df_to_csv(stocks, file_path)

    print(stocks)
    
    file_path = "data/raw_data_coint_pairs.csv"
    stocks.dropna(axis=1, inplace=True)
    stocks = stocks.set_index("Date")
    pairs = find_pairs(stocks, 255, file_path)
    print(pairs)

if __name__ == "__main__":
    main()

# # Create tables
# Base.metadata.create_all(engine)

# # Insert data
# Session = sessionmaker(bind=engine)
# session = Session()

# new_user = User(username='john_doe', email='john.doe@example.com')
# session.add(new_user)
# session.commit()

# # Query data
# users = session.query(User).all()
# for user in users:
#     print(f"User ID: {user.id}, Username: {user.username}, Email: {user.email}")
