"""Automated data checks. Every check reports; none silently fixes or deletes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import PERCENT_LIKE


def run_checks(df: pd.DataFrame, recent_months: int = 3) -> pd.DataFrame:
    """Return one row per check: check, severity, count, detail."""
    rows: list[dict] = []

    def add(check: str, severity: str, count: int, detail: str = "") -> None:
        rows.append({"check": check, "severity": severity if count else "ok", "count": int(count), "detail": detail})

    dups = df.duplicated(["centre", "month"], keep=False)
    add("Duplicate centre-month rows", "error", dups.sum(), "; ".join(sorted(df.loc[dups, "centre"].unique())[:5]))

    for col in PERCENT_LIKE:
        if col in df:
            bad = df[col].notna() & ((df[col] < 0) | (df[col] > 100))
            add(f"{col} outside 0-100", "error", bad.sum(), "; ".join(sorted(df.loc[bad, "centre"].unique())[:5]))
    for col in ("slots_total", "slots_booked", "slots_available", "overtime_slots", "median_wait_weeks", "wait_weeks_to_10pct"):
        if col in df:
            bad = df[col].notna() & (df[col] < 0)
            add(f"{col} negative", "error", bad.sum())

    share_cols = [c for c in ("share_le_7w", "share_7_14w", "share_14_21w", "share_gt_21w") if c in df]
    if len(share_cols) == 4:
        tot = df[share_cols].sum(axis=1, min_count=4)
        bad = tot.notna() & ((tot - 100).abs() > 1.5)
        add("Wait bands do not sum to ~100%", "warning", bad.sum())

    # centres that stop reporting: possibly closed or renamed
    last_all = df["month"].max()
    last_by_centre = df.groupby("centre")["month"].max()
    cutoff = last_all - pd.DateOffset(months=recent_months)
    gone = last_by_centre[last_by_centre < cutoff]
    add("Centres with no recent data (closed or renamed?)", "warning", len(gone), "; ".join(gone.index[:8]))

    # same name differing only by case or spacing
    norm = df["centre"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    variants = df.assign(n=norm).groupby("n")["centre"].nunique()
    add("Centre names differing only by case or spacing", "warning", int((variants > 1).sum()))

    # gaps inside a centre's history
    gap_centres = 0
    for _, g in df.groupby("centre"):
        span = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
        if len(span.difference(g["month"].drop_duplicates())) > 0:
            gap_centres += 1
    add("Centres with missing months inside their history", "info", gap_centres,
        "Often the 2020-21 test suspensions; the modelling uses each centre's latest unbroken run of data")

    # big one-month jumps in the main measure
    if "availability_pct" in df:
        d = df.sort_values(["centre", "month"]).groupby("centre")["availability_pct"].diff().abs()
        add("One-month change over 40 points", "warning", int((d > 40).sum()))

    if "availability_pct" in df:
        add("Missing availability values", "info", int(df["availability_pct"].isna().sum()))

    return pd.DataFrame(rows)


def centre_flags(df: pd.DataFrame, recent_months: int = 3) -> pd.DataFrame:
    """Per-centre data flags used to lower confidence in the decision layer."""
    last_all = df["month"].max()
    cutoff = last_all - pd.DateOffset(months=recent_months)
    flags = {}
    for centre, g in df.groupby("centre"):
        f = []
        if g["month"].max() < cutoff:
            f.append("no recent data")
        if g.duplicated("month").any():
            f.append("duplicate months")
        for col in PERCENT_LIKE:
            if col in g and (g[col].dropna().gt(100) | g[col].dropna().lt(0)).any():
                f.append(f"{col} out of range")
        flags[centre] = "; ".join(f)
    return pd.Series(flags, name="data_flags").rename_axis("centre").reset_index()
