import datetime
import logging
import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse

import backtester
import optimizer
from data_collector import collect_all
from exchanges.binance import BinanceClient
from utils import TF_EQUIV

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("info.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest.py - Data Collection, Backtesting and Optimization"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Choose the program mode (data / backtest / optimize)",
    )
    parser.add_argument("-e", "--exchange", type=str, help="Choose an exchange")
    parser.add_argument("-sy", "--symbol", type=str, help="Choose a symbol")
    parser.add_argument("-st", "--strategy", type=str, help="Choose a strategy")
    parser.add_argument("-tf", "--timeframe", type=str, help="Choose a timeframe")
    parser.add_argument("-bt_f", "--backtest_from", type=str, help="Backtest from")
    parser.add_argument("-bt_to", "--backtest_to", type=str, help="Backtest to")
    parser.add_argument(
        "-p", "--population", type=int, help="Choose a populaltion size"
    )
    parser.add_argument(
        "-g", "--generations", type=int, help="Choose a number of generations"
    )
    args = parser.parse_args()

    if args.mode:
        mode = args.mode
    else:
        mode = input("Choose the program mode (data / backtest / optimize): ").lower()

    if args.exchange:
        exchange = args.exchange
    else:
        while True:
            exchange = input("Choose an exchange: ").lower()

            if exchange in ["binance"]:
                break

    if exchange == "binance":
        client = BinanceClient(True)
        # candles = client.get_historical_data("BTCUSDT")

    if args.symbol:
        symbol = str(args.symbol).upper()
    else:
        while True:
            symbol = input("Choose a symbol: ").upper()

            if symbol in client.symbols:
                break

    if mode == "data":
        collect_all(client, exchange, symbol)
    elif mode in ["backtest", "optimize"]:
        available_strategies = ["hmm"]
        if args.strategy:
            strategy = args.strategy
        else:
            while True:
                strategy = input(
                    f"Choose a strategy: ({', '.join(available_strategies)}) "
                ).lower()
                if strategy in available_strategies:
                    break

        if args.timeframe:
            tf = args.timeframe
        else:
            while True:
                tf = input(
                    f"Choose a timeframe: ({', '.join(TF_EQUIV.keys())}) "
                ).lower()
                if tf in TF_EQUIV.keys():
                    break

        if args.backtest_from:
            from_time = args.backtest_from
            if from_time == "":
                from_time = 0
        else:
            while True:
                from_time = input(
                    f"Backtest from: (yyyy-mm-dd or Press Enter) "
                ).lower()
                if from_time == "":
                    from_time = 0
                    break

                try:
                    from_time = int(
                        datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp()
                        * 1000
                    )
                    break
                except ValueError:
                    continue

        if args.backtest_to:
            to_time = args.backtest_to
            if to_time == "":
                to_time = int(datetime.datetime.now().timestamp() * 1000)
        else:
            while True:
                to_time = input(f"Backtest to: (yyyy-mm-dd or Press Enter) ").lower()
                if to_time == "":
                    to_time = int(datetime.datetime.now().timestamp() * 1000)
                    break

                try:
                    to_time = int(
                        datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp()
                        * 1000
                    )
                    break
                except ValueError:
                    continue

        if mode == "backtest":
            print(backtester.run(exchange, symbol, strategy, tf, from_time, to_time))
        elif mode == "optimize":
            if args.population:
                pop_size = args.population
            else:
                while True:
                    try:
                        pop_size = int(input(f"Choose a populaltion size: "))
                        break
                    except ValueError:
                        continue

            if args.generations:
                generations = args.generations
            else:
                while True:
                    try:
                        generations = int(input(f"Choose a number of generations: "))
                        break
                    except ValueError:
                        continue

            nsga2 = optimizer.Nsga2(
                exchange, symbol, strategy, tf, from_time, to_time, pop_size
            )

            start_time = time.time()
            p_population = nsga2.create_initial_population()
            end_time = time.time() - start_time
            logger.info("create_initial_population() - %s\n", end_time)

            start_time = time.time()
            p_population = nsga2.evaluate_population(p_population)
            end_time = time.time() - start_time
            logger.info("evaluate_population() - %s\n", end_time)

            start_time = time.time()
            p_population = nsga2.crowding_distance(p_population)
            end_time = time.time() - start_time
            logger.info("crowding_distance() - %s\n", end_time)

            g = 0
            while g < generations:
                start_time = time.time()
                q_population = nsga2.create_offspring_population(p_population)
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                start_time = time.time()
                q_population = nsga2.evaluate_population(q_population)
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                start_time = time.time()
                r_population = p_population + q_population
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                start_time = time.time()
                nsga2.population_params.clear()
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                i = 0
                population = dict()
                for bt in r_population:
                    bt.reset_results()
                    nsga2.population_params.append(bt.parameters)
                    population[i] = bt
                    i += 1

                start_time = time.time()
                fronts = nsga2.non_dominated_sorting(population)
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                for j in range(len(fronts)):
                    fronts[j] = nsga2.crowding_distance(fronts[j])

                start_time = time.time()
                p_population = nsga2.create_new_population(fronts)
                end_time = time.time() - start_time
                logger.info("crowding_distance() - %s\n", end_time)

                print(f"\r{int((g + 1) / generations * 100)}%", end="")

                g += 1

            print("\n")

            for individual in p_population:
                print(individual)


if __name__ == "__main__":
    main()
