import requests
import logging

logger = logging.getLogger()


class BinanceClient:
    def __init__(self, futures: bool = False) -> None:
        self.futures = futures
        self._base_url = (
            "https://testnet.binancefuture.com"
            if self.futures
            else "https://testnet.binance.vision"
        )

        self.symbols = self._get_symbols()

    def _make_request(self, endpoint: str, query_parameters: dict):
        try:
            url = f"{self._base_url}/{endpoint}"
            response = requests.get(url, params=query_parameters)
        except Exception as e:
            logger.error("Connection error while making request to %s: %s", url)
            return None

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(
                "Error while making request to %s: %s (status code = %s)",
                url,
                response.json(),
                response.status_code,
            )
            return None

    def _get_symbols(self) -> list[str] | None:
        endpoint = "fapi/v1/exchangeInfo"

        data = self._make_request(endpoint, {})

        if data is not None:
            symbols = [x["symbol"] for x in data["symbols"]]

            return symbols

    def get_historical_data(
        self, symbol: str, start_time: int = None, end_time: int = None
    ):
        endpoint = "fapi/v1/klines"
        params = dict()

        params["symbol"] = symbol
        params["interval"] = "1m"
        params["limit"] = 1500

        if start_time is not None:
            params["startTime"] = start_time

        if end_time is not None:
            params["endTime"] = end_time

        raw_candles = self._make_request(endpoint, params)

        candles = []
        if raw_candles is not None:
            for c in raw_candles:
                candles.append(
                    (
                        float(c[0]),
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]),
                    )
                )

            return candles
        else:
            return None
