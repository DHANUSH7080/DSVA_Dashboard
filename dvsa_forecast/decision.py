"""From forecasts to a ranked, explainable shortlist.

Everything here is a transparent rule with a visible weight. The weights, thresholds and cut-offs are
ASSUMPTIONS made by the analyst, not DVSA policy, and are exposed in the dashboard so they can be
challenged and changed. The output is a prompt for a person to review, never an instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class DecisionParams:
    threshold: float = 10.0  # "tight" = availability below this many % of the 24-week window
    w_shortfall: float = 0.45  # how far below the threshold the forecast is
    w_persistence: float = 0.30  # share of the last 12 months below the threshold
    w_deterioration: float = 0.15  # forecast fall versus latest value
    w_volume: float = 0.10  # centre size (standard test slots offered in the window, 12-month sum)
    review_cut: float = 60.0  # score at or above: review for extra capacity
    watch_cut: float = 35.0  # score at or above: watch closely
    persistent_share: float = 0.75  # share of months tight to call "persistently tight"
    trend_move: float = 0.20  # forecast move (as a fraction of threshold) to call tightening/improving

    def as_dict(self) -> dict:
        return asdict(self)


def default_threshold(history: pd.DataFrame, target: str = "availability_pct") -> float:
    """Data-driven default: the 25th percentile of centre-months over the latest 12 months."""
    last = history["month"].max()
    recent = history[history["month"] > last - pd.DateOffset(months=12)][target].dropna()
    return float(np.round(recent.quantile(0.25), 1)) if len(recent) else 10.0


def _centre_features(history: pd.DataFrame, target: str, thr: float) -> pd.DataFrame:
    last = history["month"].max()
    window = history[history["month"] > last - pd.DateOffset(months=12)]
    rows = []
    for centre, g in window.groupby("centre"):
        g = g.sort_values("month")
        v = g[target].dropna()
        if v.empty:
            continue
        tail = v.iloc[-6:].to_numpy()
        slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else 0.0
        rows.append(
            {
                "centre": centre,
                "region": g["region"].dropna().iloc[-1] if "region" in g and g["region"].notna().any() else "Unknown",
                "latest": float(v.iloc[-1]),
                "mean_12m": float(v.mean()),
                "std_12m": float(v.std(ddof=0)),
                "months_tight_12m": int((v < thr).sum()),
                "months_seen_12m": int(len(v)),
                "trend_pp_per_month": slope,
                "slots_12m": float(g["slots_total"].sum()) if "slots_total" in g else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_centres(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    flags: pd.DataFrame | None = None,
    params: DecisionParams | None = None,
    target: str = "availability_pct",
) -> pd.DataFrame:
    """Return one row per centre, ranked by priority score (1 = most pressing)."""
    p = params or DecisionParams()
    thr = p.threshold
    feat = _centre_features(history, target, thr)
    if feat.empty:
        return feat

    fc = forecast.groupby("centre").agg(
        fc_mean=("mean", "mean"), fc_lo=("lo", "mean"), fc_hi=("hi", "mean"),
        model_used=("model_used", "first"), model_note=("model_note", "first"),
    ).reset_index()
    df = feat.merge(fc, on="centre", how="inner")
    if flags is not None and not flags.empty:
        df = df.merge(flags, on="centre", how="left")
    else:
        df["data_flags"] = ""
    df["data_flags"] = df["data_flags"].fillna("")

    df["share_tight"] = df["months_tight_12m"] / df["months_seen_12m"].clip(lower=1)
    shortfall = ((thr - df["fc_mean"]) / thr).clip(0, 1)
    deterioration = ((df["latest"] - df["fc_mean"]) / thr).clip(0, 1)
    if df["slots_12m"].notna().any() and df["slots_12m"].max() > 0:
        volume = (df["slots_12m"] / df["slots_12m"].max()).fillna(0.5)
    else:
        volume = pd.Series(0.5, index=df.index)

    wsum = p.w_shortfall + p.w_persistence + p.w_deterioration + p.w_volume
    df["priority_score"] = 100 * (
        p.w_shortfall * shortfall + p.w_persistence * df["share_tight"]
        + p.w_deterioration * deterioration + p.w_volume * volume
    ) / (wsum if wsum else 1)

    p75 = df["std_12m"].quantile(0.75)

    def pattern(r) -> str:
        if r.share_tight >= p.persistent_share:
            return "Persistently tight"
        if (r.fc_mean - r.latest) <= -p.trend_move * thr:
            return "Tightening"
        if (r.fc_mean - r.latest) >= p.trend_move * thr and r.share_tight >= 0.25:
            return "Improving"
        if r.std_12m >= p75 and r.std_12m > 0:
            return "Volatile"
        if r.months_tight_12m == 0:
            return "Comfortable"
        return "Mixed"

    df["pattern"] = df.apply(pattern, axis=1)

    width = (df["fc_hi"] - df["fc_lo"])
    clear = (df["fc_hi"] < thr) | (df["fc_lo"] > thr)

    def confidence(i) -> str:
        if df.at[i, "data_flags"]:
            return "Low"
        if clear[i] and width[i] <= thr:
            return "High"
        if width[i] > 1.5 * thr:
            return "Low"
        return "Medium"

    df["confidence"] = [confidence(i) for i in df.index]

    def action(r) -> str:
        if r.priority_score >= p.review_cut:
            return "Check data first" if r.confidence == "Low" and r.data_flags else "Review for extra examiner capacity"
        if r.priority_score >= p.watch_cut:
            return "Watch closely"
        return "Monitor"

    df["action"] = df.apply(action, axis=1)

    def reasons(r) -> str:
        parts = [
            f"below {thr:.1f}% in {r.months_tight_12m} of the last {r.months_seen_12m} months",
            f"latest {r.latest:.1f}%",
            f"forecast average {r.fc_mean:.1f}% (range {r.fc_lo:.1f} to {r.fc_hi:.1f}%)",
            f"trend {r.trend_pp_per_month:+.2f} points a month",
            f"model: {r.model_note}",
        ]
        if r.data_flags:
            parts.append(f"data flags: {r.data_flags}")
        return "; ".join(parts)

    df["reasons"] = df.apply(reasons, axis=1)
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    cols = ["rank", "centre", "region", "pattern", "priority_score", "action", "confidence", "latest",
            "fc_mean", "fc_lo", "fc_hi", "months_tight_12m", "months_seen_12m", "trend_pp_per_month",
            "slots_12m", "model_used", "model_note", "data_flags", "reasons"]
    return df[cols].round({"priority_score": 1, "latest": 2, "fc_mean": 2, "fc_lo": 2, "fc_hi": 2,
                           "trend_pp_per_month": 3})
