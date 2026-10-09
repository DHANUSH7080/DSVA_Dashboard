"""Forecasting models, simplest first, plus series preparation.

Candidates:
  naive           the last observed value carried forward
  seasonal_naive  the same month last year
  ets             damped-trend exponential smoothing
  ets_seasonal    damped-trend exponential smoothing with additive yearly seasonality
Each takes a 1-D numpy array (oldest first, one value per month) and a horizon, returns an array.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def f_naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def f_seasonal_naive(y: np.ndarray, h: int, m: int = 12) -> np.ndarray:
    n = len(y)
    if n < m + 1:
        return f_naive(y, h)
    # forecast step k targets position n+k-1; the same month last year is position n+k-1-m
    pos = [n + k - 1 - m for k in range(1, h + 1)]
    return np.array([y[p] if 0 <= p < n else y[-1] for p in pos], dtype=float)


def _ets(y: np.ndarray, h: int, seasonal: bool) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw = dict(trend="add", damped_trend=True, initialization_method="estimated")
            if seasonal:
                kw.update(seasonal="add", seasonal_periods=12)
            fc = np.asarray(ExponentialSmoothing(y, **kw).fit().forecast(h), dtype=float)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return fc
    except Exception:
        return f_naive(y, h)


def f_ets(y: np.ndarray, h: int) -> np.ndarray:
    return _ets(y, h, seasonal=False)


def f_ets_seasonal(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) < 30:  # needs a couple of full seasons to be meaningful
        return _ets(y, h, seasonal=False)
    return _ets(y, h, seasonal=True)


MODELS = {
    "naive": f_naive,
    "seasonal_naive": f_seasonal_naive,
    "ets": f_ets,
    "ets_seasonal": f_ets_seasonal,
}
BASELINES = ("naive", "seasonal_naive")

MODEL_LABELS = {
    "naive": "Naive (last value)",
    "seasonal_naive": "Seasonal naive (same month last year)",
    "ets": "Exponential smoothing (damped trend)",
    "ets_seasonal": "Exponential smoothing + seasonality",
}


def prepare_series(
    df_centre: pd.DataFrame, target: str, min_history: int = 24, max_gap: int = 3
) -> pd.Series | None:
    """Monthly series on a regular index, using the centre's latest unbroken run of data.

    Short gaps (up to ``max_gap`` months) are filled by linear interpolation. A longer gap ends the run,
    so the model learns from the data after it (for example after the 2020-21 suspensions) and never
    bridges a break with invented values. Returns None if the final run is shorter than ``min_history``.
    """
    s = df_centre.dropna(subset=[target]).drop_duplicates("month", keep="last").set_index("month")[target].sort_index()
    if s.empty:
        return None
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    s = s.interpolate(limit=max_gap, limit_area="inside")
    valid = s.notna().to_numpy()
    end = len(s)
    start = end
    while start > 0 and valid[start - 1]:
        start -= 1
    run = s.iloc[start:end]
    if len(run) < min_history:
        return None
    return run.astype(float)


def series_scale(y: np.ndarray, floor: float = 0.5) -> float:
    """Mean absolute one-step change: the yardstick for scaled errors (as in MASE)."""
    if len(y) < 2:
        return floor
    return float(max(np.mean(np.abs(np.diff(y))), floor))
