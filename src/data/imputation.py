# Leakage-safe imputation strategies for time-series features
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class LeakageSafeImputer:
    # Preprocessing imputer ensuring statistics are fitted strictly on proper training data
    # (Excludes both validation/calibration and test periods from fitting)
    def __init__(self, method: str = "ffill_train_median", train_end_date: str = "2018-03-13"):
        self.method = method
        self.train_end_date = pd.to_datetime(train_end_date)
        self.fit_stats_: Dict = {}

    def fit(self, df: pd.DataFrame, pollutant_cols: List[str]):
        # Compute summary statistics strictly from records with Date <= train_end_date (proper training period)
        train_df = df[df["Date"] <= self.train_end_date].copy()
        if len(train_df) == 0:
            raise ValueError(f"No training data available on or before {self.train_end_date} to fit imputer.")
        stats = {
            "city_medians": {},
            "city_means": {},
            "city_month_medians": {},
            "global_medians": train_df[pollutant_cols].median().to_dict(),
        }

        for city, cgroup in train_df.groupby("City"):
            stats["city_medians"][city] = cgroup[pollutant_cols].median().to_dict()
            stats["city_means"][city] = cgroup[pollutant_cols].mean().to_dict()

            cgroup_month = cgroup.copy()
            cgroup_month["Month"] = cgroup_month["Date"].dt.month
            stats["city_month_medians"][city] = (
                cgroup_month.groupby("Month")[pollutant_cols].median().to_dict()
            )

        self.fit_stats_ = stats
        return self

    def transform(self, df: pd.DataFrame, pollutant_cols: List[str]) -> pd.DataFrame:
        # Impute missing values using strictly causal or training-fitted statistics
        df_out = df.copy()
        df_out = df_out.sort_values(["City", "Date"]).reset_index(drop=True)

        if self.method == "ffill_train_median":
            # Method 1: Causal forward fill within city, initial missing filled with train median
            out_chunks = []
            for city, cgroup in df_out.groupby("City", sort=False):
                cgroup = cgroup.copy()
                city_med = self.fit_stats_["city_medians"].get(city, self.fit_stats_["global_medians"])
                for col in pollutant_cols:
                    if col in cgroup.columns:
                        cgroup[col] = cgroup[col].ffill()
                        fallback = city_med.get(col, self.fit_stats_["global_medians"].get(col, 0.0))
                        cgroup[col] = cgroup[col].fillna(fallback if pd.notna(fallback) else 0.0)
                out_chunks.append(cgroup)
            df_out = pd.concat(out_chunks, ignore_index=True)

        elif self.method == "train_season_median":
            # Method 2: City-Month seasonal median fitted strictly on training data
            out_chunks = []
            for city, cgroup in df_out.groupby("City", sort=False):
                cgroup = cgroup.copy()
                cgroup["Month"] = cgroup["Date"].dt.month
                city_med = self.fit_stats_["city_medians"].get(city, self.fit_stats_["global_medians"])
                month_dict = self.fit_stats_["city_month_medians"].get(city, {})

                for col in pollutant_cols:
                    if col in cgroup.columns:
                        col_month_dict = month_dict.get(col, {})
                        cgroup[f"{col}_month_med"] = cgroup["Month"].map(col_month_dict)
                        fallback = city_med.get(col, self.fit_stats_["global_medians"].get(col, 0.0))
                        cgroup[f"{col}_month_med"] = cgroup[f"{col}_month_med"].fillna(fallback)
                        cgroup[col] = cgroup[col].fillna(cgroup[f"{col}_month_med"])
                        cgroup.drop(columns=[f"{col}_month_med"], inplace=True)
                cgroup.drop(columns=["Month"], inplace=True)
                out_chunks.append(cgroup)
            df_out = pd.concat(out_chunks, ignore_index=True)

        elif self.method == "causal_rolling_mean":
            # Method 3: Strictly causal historical rolling mean (shift 1) + training mean fallback
            out_chunks = []
            for city, cgroup in df_out.groupby("City", sort=False):
                cgroup = cgroup.copy()
                city_mean = self.fit_stats_["city_means"].get(city, self.fit_stats_["global_medians"])
                for col in pollutant_cols:
                    if col in cgroup.columns:
                        causal_roll = cgroup[col].shift(1).rolling(window=7, min_periods=1).mean()
                        fallback = city_mean.get(col, self.fit_stats_["global_medians"].get(col, 0.0))
                        causal_roll = causal_roll.fillna(fallback if pd.notna(fallback) else 0.0)
                        cgroup[col] = cgroup[col].fillna(causal_roll)
                        cgroup[col] = cgroup[col].fillna(fallback if pd.notna(fallback) else 0.0)
                out_chunks.append(cgroup)
            df_out = pd.concat(out_chunks, ignore_index=True)

        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        return df_out
