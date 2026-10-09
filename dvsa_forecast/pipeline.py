"""End-to-end run: tidy data in, forecasts, error metrics, intervals and ranking out (CSV files)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest as bt_mod
from .decision import DecisionParams, default_threshold, rank_centres
from .models import BASELINES, MODEL_LABELS, MODELS, prepare_series, series_scale
from .quality import centre_flags, run_checks
from .schema import DEFAULT_TARGET, PERCENT_LIKE


def run_pipeline(
    tidy: pd.DataFrame,
    out_dir: str | Path,
    target: str = DEFAULT_TARGET,
    horizons: int = 3,
    n_origins: int = 18,
    level: float = 0.80,
    max_centres: int | None = None,
    start: str | None = None,
    synthetic: bool = False,
    source: str = "",
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda msg, frac: None)

    if target not in tidy:
        raise ValueError(f"Target column {target!r} is not in the data. Columns: {list(tidy.columns)}")

    say("Checking data quality", 0.02)
    checks = run_checks(tidy)
    flags = centre_flags(tidy)
    hist = tidy.sort_values(["centre", "month"]).drop_duplicates(["centre", "month"], keep="last").copy()
    bounds = (0.0, 100.0) if target in PERCENT_LIKE else None
    if bounds:  # impossible values are excluded from modelling, and reported by the checks above
        hist.loc[(hist[target] < 0) | (hist[target] > 100), target] = np.nan

    # only centres still reporting are forecast: a closed centre has nothing to forecast or to rank
    last_seen = hist.groupby("centre")["month"].max()
    active = set(last_seen[last_seen >= hist["month"].max() - pd.DateOffset(months=3)].index)
    n_inactive = len(last_seen) - len(active)
    hist = hist[hist["centre"].isin(active)].copy()
    # modelling window: DVSA changed the booking window from 18 to 24 weeks in 2021, so earlier
    # availability figures measure something different
    model_hist = hist[hist["month"] >= pd.Timestamp(start)] if start else hist

    centres = sorted(hist["centre"].unique())
    if max_centres:
        centres = centres[:max_centres]

    series: dict[str, pd.Series] = {}
    too_short = []
    for c in centres:
        s = prepare_series(model_hist[model_hist["centre"] == c], target)
        if s is None:
            too_short.append(c)
        else:
            series[c] = s
    if not series:
        raise ValueError("No centre has enough unbroken history (24+ months) to model.")

    say("Running rolling backtest", 0.10)
    bt = bt_mod.rolling_backtest(
        series, horizons=horizons, n_origins=n_origins, bounds=bounds,
        progress=lambda f: say("Running rolling backtest", 0.10 + 0.70 * f),
    )

    say("Scoring models", 0.82)
    by_model_h = bt_mod.summarise(bt, ["model", "horizon"])
    by_model = bt_mod.summarise(bt, ["model"])
    by_centre = bt_mod.summarise(bt, ["centre", "model"])
    region_map = hist.groupby("centre")["region"].last() if "region" in hist else pd.Series(dtype=str)
    bt_r = bt.merge(region_map.rename("region"), left_on="centre", right_index=True, how="left")
    by_region = bt_mod.summarise(bt_r, ["region", "model"])
    selection_eval = bt_mod.evaluate_selection(bt)

    baseline = bt_mod.pooled_baseline(bt)
    champ = bt_mod.champion(bt)
    choice = bt_mod.per_centre_choice(bt, champ, baseline).set_index("centre")

    coverage = pd.concat(
        [bt_mod.coverage_check(bt[bt["model"] == m], level).assign(model=m) for m in MODELS], ignore_index=True
    )
    quant = {m: bt_mod.conformal_quantiles(bt[bt["model"] == m], level) for m in MODELS}

    say("Forecasting", 0.88)
    rows = []
    for c, s in series.items():
        used = choice.at[c, "model_used"] if c in choice.index else baseline
        y = s.to_numpy()
        fc = MODELS[used](y, horizons)
        if bounds:
            fc = np.clip(fc, *bounds)
        scale = series_scale(y)
        note = f"{MODEL_LABELS[used]} ({choice.at[c, 'why']})" if c in choice.index else MODEL_LABELS[used]
        for h in range(1, horizons + 1):
            half = quant[used].get(h, np.nan) * scale
            lo, hi = fc[h - 1] - half, fc[h - 1] + half
            if bounds:
                lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
            rows.append({"centre": c, "horizon": h, "target_month": s.index[-1] + pd.DateOffset(months=h),
                         "mean": float(fc[h - 1]), "lo": float(lo), "hi": float(hi),
                         "model_used": used, "model_note": note})
    forecast = pd.DataFrame(rows)

    say("Ranking centres", 0.95)
    params = DecisionParams(threshold=default_threshold(hist, target))
    ranking = rank_centres(hist, forecast, flags, params, target)

    notes = pd.DataFrame(
        {"check": ["Centres excluded: under 24 months of unbroken history",
                   "Centres excluded: no data in the last 3 months (closed or renamed)",
                   "Modelling window start"],
         "severity": ["info" if too_short else "ok", "info" if n_inactive else "ok", "info" if start else "ok"],
         "count": [len(too_short), n_inactive, 0],
         "detail": ["; ".join(too_short[:8]), "",
                    f"Months before {start} are shown but not modelled (booking window changed 18 -> 24 weeks)" if start else ""]}
    )
    checks = pd.concat([checks, notes], ignore_index=True)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": synthetic, "source": source, "target": target, "horizons": horizons,
        "n_origins": n_origins, "interval_level": level, "centres_modelled": len(series),
        "centres_excluded": len(too_short), "centres_closed_or_stale": n_inactive, "model_start": start, "last_data_month": str(hist["month"].max().date()),
        "baseline": baseline, "champion": champ, "default_threshold": params.threshold,
    }
    results = {
        "history": hist, "backtest": bt, "metrics_by_model_horizon": by_model_h, "metrics_by_model": by_model,
        "metrics_by_centre": by_centre, "metrics_by_region": by_region, "selection_eval": selection_eval,
        "coverage": coverage, "forecast": forecast, "priorities": ranking, "quality": checks, "flags": flags,
    }
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    say("Done", 1.0)
    results["meta"] = pd.DataFrame([meta])
    return results
