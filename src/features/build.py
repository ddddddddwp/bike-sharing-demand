"""Feature engineering for Bike Sharing Demand.

The builder is deliberately fold-safe: `fit` only sees training rows and stores
target aggregates; `transform` can be applied to validation/test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "count"
DROP = {"casual", "registered", "count", "datetime"}
CAT_COLS = ["season", "weather", "weekday", "month"]


def _parse_dt(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    raise KeyError("dataframe must contain `datetime`")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = _parse_dt(out)
    out["year"] = dt.dt.year.astype("int16")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["hour"] = dt.dt.hour.astype("int8")
    out["weekday"] = dt.dt.weekday.astype("int8")
    out["dayofyear"] = dt.dt.dayofyear.astype("int16")
    out["weekofyear"] = dt.dt.isocalendar().week.astype("int8")
    out["is_weekend"] = (out["weekday"] >= 5).astype("int8")
    out["is_rush_hour"] = out["hour"].isin([7, 8, 9, 17, 18, 19]).astype("int8")
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["hour_x_workingday"] = out["hour"].astype(str) + "_" + out["workingday"].astype(str)
    out["temp_x_season"] = out["temp"] * out["season"]
    out["humidity_x_weather"] = out["humidity"] * out["weather"]
    return out


class FeatureBuilder:
    """Fold-safe feature builder with smoothed target aggregates."""

    def __init__(self, use_target_aggregates: bool = True, smooth: float = 20.0):
        self.use_target_aggregates = bool(use_target_aggregates)
        self.smooth = float(smooth)
        self._agg: dict[str, pd.Series] = {}
        self._global = 0.0
        self._names: list[str] = []

    def _keys(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "hour_mean": df["hour"].astype(str),
            "weekday_mean": df["weekday"].astype(str),
            "month_mean": df["month"].astype(str),
            "season_mean": df["season"].astype(str),
            "weather_mean": df["weather"].astype(str),
            "workingday_mean": df["workingday"].astype(str),
            "hour_workingday_mean": df["hour_x_workingday"].astype(str),
        }

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        base = add_time_features(df)
        y = base[TARGET].astype("float64")
        self._global = float(y.mean())
        self._agg = {}
        if self.use_target_aggregates:
            for name, key in self._keys(base).items():
                g = y.groupby(key).agg(["sum", "count"])
                # smoothed mean: (sum + m*global) / (count + m)
                smoothed = (g["sum"] + self.smooth * self._global) / (g["count"] + self.smooth)
                self._agg[name] = smoothed.astype("float64")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = add_time_features(df)
        if self.use_target_aggregates:
            for name, key in self._keys(x).items():
                x[name] = key.map(self._agg[name]).astype("float64").fillna(self._global)
        drop = [c for c in DROP if c in x.columns]
        x = x.drop(columns=drop)
        # keep category dtype for tree models
        for c in CAT_COLS:
            if c in x.columns:
                x[c] = x[c].astype("category")
        self._names = list(x.columns)
        return x

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)
