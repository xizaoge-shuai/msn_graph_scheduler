from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TelecomColumns:
    timestamp: str = "timestamp"
    user_id: str = "user_id"
    base_station: str = "base_station_id"
    latitude: str = "latitude"
    longitude: str = "longitude"


def load_generic_telecom_csv(path: str, columns: TelecomColumns = TelecomColumns()) -> pd.DataFrame:
    """Load a user-supplied telecom trace without assuming a proprietary schema.

    Rename source columns before calling, or pass a TelecomColumns mapping.
    """
    df = pd.read_csv(path)
    required = [columns.timestamp, columns.user_id, columns.base_station]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Telecom trace missing columns: {missing}")
    out = df.rename(
        columns={
            columns.timestamp: "timestamp",
            columns.user_id: "user_id",
            columns.base_station: "base_station_id",
            columns.latitude: "latitude",
            columns.longitude: "longitude",
        }
    ).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "user_id", "base_station_id"])
    return out.sort_values(["user_id", "timestamp"])


def active_users_per_station(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    x = df.set_index("timestamp")
    return (
        x.groupby("base_station_id")["user_id"]
        .resample(freq)
        .nunique()
        .rename("active_users")
        .reset_index()
    )


def request_intensity(active_users: np.ndarray, lambda0: float, kappa: float) -> np.ndarray:
    return lambda0 + kappa * np.asarray(active_users, dtype=float)
