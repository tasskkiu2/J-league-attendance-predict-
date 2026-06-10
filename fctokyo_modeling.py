"""Utilities for FC Tokyo attendance modeling.

The notebooks in this repository are useful for exploration, but the repeated
data preparation and evaluation logic is easier to maintain here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MATCH_DATA_PATH = RAW_DATA_DIR / "J1_tokyo_home_2015-2024.csv"
WEATHER_DATA_PATH = RAW_DATA_DIR / "weather_tokyo_2015-2024.csv"
TARGET_COLUMN = "入場者数"


@dataclass(frozen=True)
class ModelResult:
    name: str
    r2: float
    rmse: float


def load_match_data(path: str | Path = MATCH_DATA_PATH) -> pd.DataFrame:
    """Load raw FC Tokyo home-match data."""
    return pd.read_csv(path)


def load_weather_data(path: str | Path = WEATHER_DATA_PATH) -> pd.DataFrame:
    """Load Tokyo daily weather data and keep only columns used by the model."""
    weather = pd.read_csv(path, header=4)
    weather = weather.iloc[:, [0, 1, 4]].copy()
    weather.columns = ["date", "temperature", "rain"]
    weather["date"] = pd.to_datetime(weather["date"], format="%Y/%m/%d")
    weather["temperature"] = pd.to_numeric(weather["temperature"], errors="coerce")
    weather["rain"] = pd.to_numeric(weather["rain"], errors="coerce").fillna(0)
    return weather


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags and date-related fields shared across analyses."""
    df = df.copy()
    df["コロナ禍ダミー"] = df["シーズン"].isin([2020, 2021]).astype(int)
    df["国立フラグ"] = df["スタジアム"].str.contains("国立", na=False).astype(int)
    df["試合日"] = df["試合日"].astype(str)
    df["曜日"] = df["試合日"].str.extract(r"[（(]([^)）]+)[)）]")
    df["休日フラグ"] = df["曜日"].str.contains("土|日|祝", na=False).astype(int)
    df["date"] = pd.to_datetime(
        df["試合日"].str.extract(r"(\d{2}/\d{2}/\d{2})")[0],
        format="%y/%m/%d",
    )
    df["datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["K/O時刻"].astype(str),
        errors="coerce",
    )
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    return df


def add_weather_features(
    matches: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge weather data and derive weather-related categorical flags."""
    weather = load_weather_data() if weather is None else weather
    df = matches.merge(weather, on="date", how="left")
    df["rain_flag"] = (df["rain"].fillna(0) > 0).astype(int)
    df["temp_zone"] = pd.cut(
        df["temperature"],
        bins=[0, 10, 15, 20, 25, 30, 35, 40],
        labels=["寒い", "やや寒い", "快適", "快適2", "暑い", "猛暑", "酷暑"],
    )
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-match and rolling attendance features."""
    df = df.copy()
    attendance = df[TARGET_COLUMN]
    df["lag1"] = attendance.shift(1)
    df["lag2"] = attendance.shift(2)
    df["rolling_mean_2"] = attendance.rolling(window=2).mean()
    df["rolling_mean_3"] = attendance.rolling(window=3).mean()
    df["rolling_mean_5"] = attendance.rolling(window=5).mean()
    df["rolling_mean_7"] = attendance.rolling(window=7).mean()

    scores = df["スコア"].str.split("-", expand=True).astype(float)
    df["home_score"] = scores[0]
    df["away_score"] = scores[1]
    df["result_numeric"] = (df["home_score"] > df["away_score"]).astype(int)
    df["lag1_result_numeric"] = df["result_numeric"].shift(1)
    return df


def prepare_dataset(
    match_path: str | Path = MATCH_DATA_PATH,
    weather_path: str | Path = WEATHER_DATA_PATH,
) -> pd.DataFrame:
    """Build the modeling dataset from raw match and weather CSV files."""
    matches = add_basic_features(load_match_data(match_path))
    weather = load_weather_data(weather_path)
    df = add_weather_features(matches, weather)
    return add_time_series_features(df)


def build_feature_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a statsmodels/sklearn-ready X/y pair with aligned rows."""
    categorical_features = categorical_features or []
    parts = [df[numeric_features].astype(float)]

    for column in categorical_features:
        parts.append(pd.get_dummies(df[column], drop_first=True).astype(float))

    x = pd.concat(parts, axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = df[target].astype(float)
    valid_index = x.dropna().index.intersection(y.dropna().index)
    return x.loc[valid_index], y.loc[valid_index]


def split_by_test_season(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    test_season: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Use one season as the holdout set."""
    season = df.loc[x.index, "シーズン"].astype(int)
    train_mask = season != test_season
    test_mask = season == test_season
    return x[train_mask], x[test_mask], y[train_mask], y[test_mask]


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Return R2 and RMSE for a prediction vector."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def fit_ols(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ModelResult, sm.regression.linear_model.RegressionResultsWrapper]:
    model = sm.OLS(y_train, x_train).fit()
    r2, rmse = score_predictions(y_test, model.predict(x_test))
    return ModelResult("OLS", r2, rmse), model


def fit_random_forest(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[ModelResult, RandomForestRegressor]:
    x_train = x_train.drop(columns=["const"], errors="ignore")
    x_test = x_test.drop(columns=["const"], errors="ignore")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    r2, rmse = score_predictions(y_test, model.predict(x_test))
    return ModelResult("Random Forest", r2, rmse), model


def run_baseline_comparison(test_season: int = 2024) -> list[ModelResult]:
    """Run the baseline models used most often in the notebooks."""
    df = prepare_dataset()
    numeric_features = [
        "コロナ禍ダミー",
        "国立フラグ",
        "休日フラグ",
        "rolling_mean_3",
        "rain_flag",
    ]
    x, y = build_feature_matrix(
        df,
        numeric_features=numeric_features,
        categorical_features=["アウェイ"],
    )
    x_train, x_test, y_train, y_test = split_by_test_season(df, x, y, test_season)
    ols_result, _ = fit_ols(x_train, x_test, y_train, y_test)
    rf_result, _ = fit_random_forest(x_train, x_test, y_train, y_test)
    return [ols_result, rf_result]


if __name__ == "__main__":
    for result in run_baseline_comparison():
        print(f"{result.name}: R2={result.r2:.3f}, RMSE={result.rmse:,.0f}")
