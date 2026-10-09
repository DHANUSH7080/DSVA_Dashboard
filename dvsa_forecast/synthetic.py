"""Synthetic data in the tidy contract, for tests and the demo.

THIS IS NOT DVSA DATA. Centre names are invented, and every number is simulated. It exists so
the pipeline and dashboard can be exercised without the real files, and is always labelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = [f"Region {c}" for c in "ABCDEFGHI"]


def make_synthetic(
    n_centres: int = 40,
    start: str = "2019-01-01",
    end: str = "2026-08-01",
    seed: int = 7,
    inject_issues: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    n = len(months)
    month_num = months.month.values
    rows = []
    for c in range(n_centres):
        name = f"Synthetic Centre {c + 1:03d}"
        region = REGIONS[c % len(REGIONS)]
        level = rng.uniform(4, 24)
        amp = rng.uniform(0.0, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0.6, 0.9)
        sd = rng.uniform(0.5, 2.0)
        size = rng.uniform(80, 900)
        drift = rng.uniform(-0.03, 0.04)

        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + rng.normal(0, sd)
        t_idx = np.arange(n)
        y = level + drift * t_idx + amp * np.sin(2 * np.pi * month_num / 12 + phase) + noise

        post_pandemic = months >= "2020-09-01"
        y = np.where(post_pandemic, y - rng.uniform(0, 5), y)  # post-suspension backlog
        y = np.where(months >= "2026-05-01", y + rng.uniform(0, 3), y)  # a rules-change step
        y = np.clip(y, 0.3, 60.0)

        tests = rng.poisson(size * (1 + 0.1 * np.sin(2 * np.pi * month_num / 12)))
        pass_rate = np.clip(rng.normal(48, 3, n), 30, 70)

        for i, m in enumerate(months):
            if "2020-04-01" <= str(m.date()) <= "2020-08-01":
                continue  # suspension: no data published
            rows.append(
                {
                    "centre": name,
                    "region": region,
                    "month": m,
                    "availability_pct": round(float(y[i]), 2),
                    "tests_conducted": int(tests[i]),
                    "pass_rate": round(float(pass_rate[i]), 1),
                }
            )
    df = pd.DataFrame(rows)

    if inject_issues:
        dup = df.iloc[[10]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        df.loc[df.index[25], "availability_pct"] = 140.0  # impossible value
        closed = df["centre"].eq("Synthetic Centre 003") & (df["month"] >= "2026-03-01")
        df = df.loc[~closed].reset_index(drop=True)  # a centre that stops reporting
    return df.sort_values(["centre", "month"]).reset_index(drop=True)
