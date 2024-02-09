import numpy as np
import pandas as pd
import statsmodels.api as sm
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
from utility_functions import df_to_csv, does_file_exist


def find_returns_and_volatility(data: pd.DataFrame, trading_days: int):
    df_returns = pd.DataFrame(
        data.pct_change().mean() * trading_days, columns=["Returns"]
    )
    df_returns["Volatility"] = data.pct_change().std() * np.sqrt(255)

    return df_returns


def scale_features(data: pd.DataFrame):
    scaler = StandardScaler()
    scaler = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaler, columns=data.columns, index=data.index)
    df_scaled = scaled_data

    return df_scaled


def fit_k_means_model(df_scaled: pd.DataFrame):
    X = df_scaled.copy()
    K = range(1, 15)
    distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    kl = KneeLocator(K, distortions, curve="convex", direction="decreasing")
    c = kl.elbow
    k_means = KMeans(n_clusters=c)
    k_means.fit(X)
    prediction = k_means.predict(df_scaled)

    return X, k_means


def return_clustered_series(X: pd.DataFrame, k_means: KMeans):
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series = clustered_series[clustered_series != -1]

    return clustered_series, clustered_series_all


def clean_clustered_series(clustered_series: pd.DataFrame):
    clusters_clean = clustered_series[clustered_series < 3]
    return clusters_clean


def calculate_cointegration(series_1: pd.Series, series_2: pd.Series):
    coint_flag = 0
    coint_res = coint(series_1, series_2)
    coint_t = coint_res[0]
    p_value = coint_res[1]
    critical_value = coint_res[2][1]
    model = sm.OLS(series_1, series_2).fit()
    hedge_ratio = model.params[0]
    coint_flag = 1 if p_value < 0.05 and coint_t < critical_value else 0
    return coint_flag, hedge_ratio


def find_conintegrated_pairs(
    clusters_clean: pd.DataFrame, data: pd.DataFrame, file_path: str
):
    tested_pairs = []
    cointegrated_pairs = []

    for base_asset in clusters_clean.index:
        base_label = clusters_clean[base_asset]

        for compare_asset in clusters_clean.index:
            compare_label = clusters_clean[compare_asset]

            test_pair = base_asset + compare_asset
            test_pair = "".join(sorted(test_pair))
            is_tested = test_pair in tested_pairs
            tested_pairs.append(test_pair)

            if (
                compare_asset != base_asset
                and base_label == compare_label
                and not is_tested
            ):
                series_1 = data[base_asset].values.astype(float)
                series_2 = data[compare_asset].values.astype(float)
                coint_flag, _ = calculate_cointegration(series_1, series_2)
                if coint_flag == 1:
                    cointegrated_pairs.append(
                        {
                            "base_asset": base_asset,
                            "compare_asset": compare_asset,
                            "label": base_label,
                        }
                    )

                    print(
                        {
                            "base_asset": base_asset,
                            "compare_asset": compare_asset,
                            "label": base_label,
                        }
                    )

    df_coint = pd.DataFrame(cointegrated_pairs)
    return df_coint


def get_unique_cointegrated_assets(df_coint: pd.DataFrame):
    coint_assets = [df_coint["base_asset"].values]
    coint_assets.append(df_coint["compare_asset"].values)
    coint_unique_assets = np.unique(coint_assets)

    return coint_unique_assets


def find_pairs(data: pd.DataFrame, trading_days: int, file_path: str):
    if not does_file_exist(file_path):
        df_returns = find_returns_and_volatility(data, trading_days)
        df_scaled = scale_features(df_returns)
        X, k_means = fit_k_means_model(df_scaled)
        clustered_series, clustered_series_all = return_clustered_series(X, k_means)
        clusters_clean = clean_clustered_series(clustered_series)
        df_coint = find_conintegrated_pairs(clustered_series, data, file_path)
        df_to_csv(df_coint, file_path)
    else:
        df_coint = pd.read_csv(file_path)

    return df_coint
