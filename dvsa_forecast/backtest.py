"""Rolling-origin backtest, error metrics, selection, and prediction ranges.

Why rolling origin: a single train/test split can flatter or punish a model by luck. Here every model
forecasts from many past months and is scored against what actually happened, at each horizon.

Prediction ranges are split-conformal: the half-width at each horizon is the chosen quantile of past
scaled absolute errors at that horizon, multiplied by the centre's own scale. They make no assumption
about the shape of the errors, and their real coverage is checked on later origins than the ones used
to set them.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd

from .models import BASELINES, MODELS, series_scale

_COLUMNS = ["centre", "model", "origin", "horizon", "target_month", "actual", "forecast", "scale"]


def _backtest_one(task: tuple, models: dict[str, Callable] | None = None) -> list[tuple]:
    centre, index, y, horizons, n_origins, min_train, bounds = task
    try:  # keep each worker single-threaded so processes do not fight over cores
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass
    models = models or MODELS
    idx = pd.DatetimeIndex(index)
    n = len(y)
    rows: list[tuple] = []
    # an origin t means: train on y[:t], then forecast the next `horizons` months
    first_origin = max(min_train, n - horizons - n_origins + 1)
    for t in range(first_origin, n - horizons + 1):
        train = y[:t]
        scale = series_scale(train)
        for name, fn in models.items():
            fc = fn(train, horizons)
            if bounds:
                fc = np.clip(fc, *bounds)
            for h in range(1, horizons + 1):
                pos = t + h - 1
                if pos < n:
                    rows.append((centre, name, idx[t - 1], h, idx[pos], float(y[pos]), float(fc[h - 1]), scale))
    return rows


def rolling_backtest(
    series: dict[str, pd.Series],
    horizons: int = 3,
    n_origins: int = 18,
    min_train: int = 24,
    bounds: tuple[float, float] | None = (0.0, 100.0),
    models: dict[str, Callable] | None = None,
    progress: Callable[[float], None] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """One row per centre, model, origin and horizon."""
    tasks = [
        (c, s.index.to_numpy(), s.to_numpy(dtype=float), horizons, n_origins, min_train, bounds)
        for c, s in series.items()
    ]
    rows: list[tuple] = []
    workers = n_jobs if n_jobs is not None else min(os.cpu_count() or 1, 8)
    if models is not None or workers <= 1 or len(tasks) < 4:
        for i, task in enumerate(tasks):
            rows.extend(_backtest_one(task, models))
            if progress:
                progress((i + 1) / len(tasks))
    else:
        # "spawn" is the default on Windows and macOS and avoids fork/thread deadlocks everywhere
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_backtest_one, t) for t in tasks]
            for i, f in enumerate(as_completed(futures)):
                rows.extend(f.result())
                if progress:
                    progress((i + 1) / len(tasks))
    bt = pd.DataFrame(rows, columns=_COLUMNS)
    bt["error"] = bt["forecast"] - bt["actual"]
    bt["abs_err"] = bt["error"].abs()
    bt["scaled_abs_err"] = bt["abs_err"] / bt["scale"]
    return bt.sort_values(["centre", "model", "origin", "horizon"]).reset_index(drop=True)


# ---- metrics -----------------------------------------------------------------------------------------


def summarise(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """MAE, RMSE, MASE (mean scaled error; below 1 beats a naive one-step forecast) and bias."""
    work = bt.assign(sq_err=bt["error"] ** 2)
    g = work.groupby(by)
    out = g.agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        mse=("sq_err", "mean"),
        mase=("scaled_abs_err", "mean"),
        bias=("error", "mean"),
    ).reset_index()
    out["rmse"] = np.sqrt(out.pop("mse"))
    return out[by + ["n", "mae", "rmse", "mase", "bias"]]


def pooled_baseline(bt: pd.DataFrame) -> str:
    sub = bt[bt["model"].isin(BASELINES)]
    return str(sub.groupby("model")["scaled_abs_err"].mean().idxmin())


def champion(bt: pd.DataFrame) -> str:
    return str(bt.groupby("model")["scaled_abs_err"].mean().idxmin())


def per_centre_choice(bt: pd.DataFrame, champ: str, baseline: str) -> pd.DataFrame:
    """Use the champion for a centre only if it had lower MAE than the baseline for that centre."""
    pivot = bt[bt["model"].isin({champ, baseline})].groupby(["centre", "model"])["abs_err"].mean().unstack()
    rows = []
    for centre, r in pivot.iterrows():
        if champ == baseline or champ not in r or baseline not in r:
            use, why = baseline, "baseline is the best overall model"
        elif r[champ] < r[baseline]:
            use, why = champ, "beat the baseline for this centre in the backtest"
        else:
            use, why = baseline, "did not beat the baseline for this centre in the backtest"
        rows.append({"centre": centre, "model_used": use, "why": why,
                     "mae_champion": r.get(champ, np.nan), "mae_baseline": r.get(baseline, np.nan)})
    return pd.DataFrame(rows)


def split_origins(bt: pd.DataFrame, frac: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split of origins: earlier ones to choose models and set ranges, later ones to judge."""
    origins = np.sort(bt["origin"].unique())
    cut = origins[max(1, int(len(origins) * frac)) - 1]
    return bt[bt["origin"] <= cut], bt[bt["origin"] > cut]


def evaluate_selection(bt: pd.DataFrame) -> pd.DataFrame:
    """Honest comparison on later origins: baseline, one overall champion, and per-centre selection.

    Models are chosen on the EARLIER origins only, then scored on the LATER ones, so the numbers are
    not flattered by choosing the winner on the same data it is judged on.
    """
    early, late = split_origins(bt)
    if late.empty or early.empty:
        return pd.DataFrame()
    base = pooled_baseline(early)
    champ = champion(early)
    choice = per_centre_choice(early, champ, base).set_index("centre")["model_used"]

    def score(frame: pd.DataFrame, label: str) -> dict:
        return {"approach": label, "n": len(frame), "mae": frame["abs_err"].mean(),
                "mase": frame["scaled_abs_err"].mean(), "bias": frame["error"].mean()}

    rows = [score(late[late["model"] == base], f"Baseline only ({base})")]
    if champ != base:
        rows.append(score(late[late["model"] == champ], f"Overall champion only ({champ})"))
    sel = late.merge(choice.rename("chosen"), left_on="centre", right_index=True)
    sel = sel[sel["model"] == sel["chosen"]]
    rows.append(score(sel, "Per-centre selection (ships only if it beats the baseline)"))
    out = pd.DataFrame(rows)
    out["skill_vs_baseline_pct"] = 100 * (1 - out["mase"] / out["mase"].iloc[0])
    return out


# ---- prediction ranges --------------------------------------------------------------------------------


def conformal_quantiles(bt_model: pd.DataFrame, level: float) -> dict[int, float]:
    """Per-horizon quantile of scaled absolute errors (finite-sample corrected)."""
    q: dict[int, float] = {}
    for h, g in bt_model.groupby("horizon"):
        e = np.sort(g["scaled_abs_err"].dropna().to_numpy())
        n = len(e)
        if n == 0:
            continue
        k = min(n, int(np.ceil((n + 1) * level)))
        q[int(h)] = float(e[k - 1])
    return q


def coverage_check(bt_model: pd.DataFrame, level: float) -> pd.DataFrame:
    """Set ranges on earlier origins, then measure how often they contained the truth on later ones."""
    early, late = split_origins(bt_model)
    q = conformal_quantiles(early, level)
    rows = []
    for h, g in late.groupby("horizon"):
        if int(h) not in q:
            continue
        half = q[int(h)] * g["scale"]
        covered = (g["abs_err"] <= half).mean()
        rows.append({"horizon": int(h), "nominal": level, "observed": covered,
                     "mean_half_width": half.mean(), "n": len(g)})
    return pd.DataFrame(rows)
