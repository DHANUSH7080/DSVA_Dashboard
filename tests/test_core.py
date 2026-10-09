from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvsa_forecast import backtest as B
from dvsa_forecast.decision import DecisionParams, rank_centres
from dvsa_forecast.ingest import load_centre_table
from dvsa_forecast.models import f_seasonal_naive, prepare_series
from dvsa_forecast.pipeline import run_pipeline
from dvsa_forecast.quality import run_checks
from dvsa_forecast.synthetic import make_synthetic


# ---- models --------------------------------------------------------------------------------------------


def test_seasonal_naive_uses_same_month_last_year():
    y = np.arange(1.0, 31.0)  # 30 months
    fc = f_seasonal_naive(y, 3)
    # next months are positions 30,31,32; a year earlier = positions 18,19,20
    assert list(fc) == [y[18], y[19], y[20]]


def test_prepare_series_fills_short_gaps_and_breaks_on_long_ones():
    months = pd.date_range("2019-01-01", periods=60, freq="MS")
    df = pd.DataFrame({"month": months, "v": np.arange(60.0)})
    df = df.drop(index=[10, 11])  # short gap: filled
    s = prepare_series(df, "v")
    assert len(s) == 60 and s.notna().all()
    df2 = pd.DataFrame({"month": months, "v": np.arange(60.0)}).drop(index=range(20, 26))  # 6-month gap
    s2 = prepare_series(df2, "v")
    assert s2.index.min() == months[26]  # starts after the break; nothing bridged
    assert len(s2) == 34


def test_prepare_series_too_short_returns_none():
    months = pd.date_range("2025-01-01", periods=10, freq="MS")
    assert prepare_series(pd.DataFrame({"month": months, "v": np.arange(10.0)}), "v") is None


# ---- backtest -------------------------------------------------------------------------------------------


def _toy_series(n=60, seed=1, sd=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.Series(10 + np.cumsum(rng.normal(0, sd, n)) * 0.2 + rng.normal(0, sd, n), index=idx)


def test_backtest_has_no_look_ahead():
    s = _toy_series()
    bt = B.rolling_backtest({"A": s}, horizons=3, n_origins=10, n_jobs=1)
    assert (bt["target_month"] > bt["origin"]).all()
    # actual equals the series value at the target month
    chk = bt.merge(s.rename("y"), left_on="target_month", right_index=True)
    assert np.allclose(chk["actual"], chk["y"])

    # a model that can only see the training data must not beat one that peeks: the naive forecast at
    # horizon 1 equals the value at the origin month, never the target month
    n1 = bt[(bt["model"] == "naive") & (bt["horizon"] == 1)].merge(s.rename("o"), left_on="origin", right_index=True)
    assert np.allclose(n1["forecast"], np.clip(n1["o"], 0, 100))


def test_metrics_and_selection_run():
    series = {f"C{i}": _toy_series(seed=i) for i in range(6)}
    bt = B.rolling_backtest(series, horizons=3, n_origins=12, n_jobs=1)
    m = B.summarise(bt, ["model"])
    assert set(m.columns) >= {"mae", "rmse", "mase", "bias"} and (m["rmse"] >= m["mae"] - 1e-9).all()
    sel = B.evaluate_selection(bt)
    assert not sel.empty and sel["skill_vs_baseline_pct"].iloc[0] == 0


def test_conformal_ranges_cover_roughly_the_nominal_rate_on_stable_data():
    series = {f"C{i}": _toy_series(n=80, seed=100 + i) for i in range(25)}
    bt = B.rolling_backtest(series, horizons=2, n_origins=24, n_jobs=1, models={"naive": __import__("dvsa_forecast.models", fromlist=["x"]).f_naive})
    cov = B.coverage_check(bt[bt["model"] == "naive"], 0.8)
    assert cov["observed"].between(0.68, 0.92).all(), cov


# ---- quality --------------------------------------------------------------------------------------------


def test_quality_checks_find_injected_problems():
    df = make_synthetic(10, inject_issues=True)
    chk = run_checks(df).set_index("check")
    assert chk.loc["Duplicate centre-month rows", "count"] >= 2
    assert chk.loc["availability_pct outside 0-100", "count"] >= 1
    assert chk.loc["Centres with no recent data (closed or renamed?)", "count"] >= 1


# ---- ingest ---------------------------------------------------------------------------------------------


REAL_HEADER = [
    "Country", "Region", "Driving test centre", "Month",
    "Test slots provided through overtime [note 3]",
    "Weeks to 10%+ weekly appointment availability [note 4]",
    "Average (median) waiting time between booking and taking the test (weeks) [note 5]",
    "Tests taken within 7 weeks of booking (Percentage) [note 6]",
    "Tests taken in more than 7 and up to 14 weeks of booking (Percentage) [note 7]",
    "Tests taken in more than 14 and up to 21 weeks of booking (Percentage) [note 8]",
    "Tests taken in more than 21 weeks of booking (Percentage)[note 9]",
    "Total standard tests made available within the booking window [note 10]",
    "Standard tests booked within the booking window [note 11]",
    "Standard tests still available to book within the booking window [note 12]",
    "Standard tests still available to book within the booking window (Percentage) [note 13]",
    "Notes",
]


def _write_ods(path, frac=False):
    """A file laid out like DVSA's DRT122F: title rows, then the header, [x] markers, closed centres."""
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    rows = []
    for c in ["Alpha", "[z] Beta"]:
        for i, m in enumerate(months):
            pct = 0.12 + i / 100 if frac else 12.0 + i
            total, booked = 1000 + i, 880 - i
            rows.append(["England", "North", c, m, "[x]", 4.5 + i, "[x]", "[x]", "[x]", "[x]", "[x]",
                         total, booked, total - booked, "[x]" if (c == "Alpha" and i == 2) else pct, "[z]"])
    titles = [["Car driving test availability by driving test centre"] + [None] * 15,
              ["This worksheet contains one table."] + [None] * 15,
              ["Shorthand: [x] = not available. Centres prefixed by [z] are now closed."] + [None] * 15]
    cover = pd.DataFrame([["Cover sheet"]])
    notes = pd.DataFrame([["Note", "Meaning"], ["1", "x"]])
    with pd.ExcelWriter(path, engine="odf") as w:
        cover.to_excel(w, sheet_name="Cover_sheet", header=False, index=False)
        notes.to_excel(w, sheet_name="Notes", header=False, index=False)
        pd.DataFrame(titles + [REAL_HEADER] + rows).to_excel(w, sheet_name="DRT122F", header=False, index=False)


def test_loader_maps_real_layout_and_cleans_markers(tmp_path):
    f = tmp_path / "t.ods"
    _write_ods(f)
    tidy, rep = load_centre_table(f)
    assert rep.sheet == "DRT122F" and rep.header_row == 3
    # the percentage column, not the slot COUNT column, must be the availability measure
    assert rep.mapped["availability_pct"].endswith("(Percentage) [note 13]")
    assert rep.mapped["slots_available"].endswith("[note 12]")
    assert rep.mapped["slots_total"].endswith("[note 10]")
    assert len(tidy) == 12 and tidy["centre"].nunique() == 2
    assert sorted(tidy["centre"].unique()) == ["Alpha", "Beta"]  # "[z] " prefix removed
    assert tidy.groupby("centre")["closed"].all().to_dict() == {"Alpha": False, "Beta": True} or \
        tidy.groupby("centre")["closed"].any().to_dict() == {"Alpha": False, "Beta": True}
    assert tidy["availability_pct"].isna().sum() == 1  # the [x] marker
    assert tidy["share_gt_21w"].isna().all()
    assert tidy["slots_total"].iloc[0] == 1000


def test_loader_converts_fractions_to_percent(tmp_path):
    f = tmp_path / "t.ods"
    _write_ods(f, frac=True)
    tidy, rep = load_centre_table(f)
    assert tidy["availability_pct"].max() > 10 and any("fraction" in n for n in rep.notes)


def test_loader_explains_failure(tmp_path):
    f = tmp_path / "bad.ods"
    with pd.ExcelWriter(f, engine="odf") as w:
        pd.DataFrame([["a", "b"], [1, 2]]).to_excel(w, header=False, index=False)
    with pytest.raises(ValueError, match="header row"):
        load_centre_table(f)


# ---- decision + pipeline --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    tidy = make_synthetic(12, inject_issues=False)
    return run_pipeline(tidy, out, synthetic=True, n_origins=10), out


def test_pipeline_outputs_are_sane(results):
    r, out = results
    f = r["forecast"]
    assert (f["lo"] <= f["mean"] + 1e-9).all() and (f["mean"] <= f["hi"] + 1e-9).all()
    assert f[["lo", "hi", "mean"]].min().min() >= 0 and f[["lo", "hi", "mean"]].max().max() <= 100
    assert (out / "meta.json").exists() and (out / "priorities.csv").exists()
    assert r["priorities"]["rank"].tolist() == list(range(1, len(r["priorities"]) + 1))


def test_ranking_orders_tight_above_comfortable_and_responds_to_threshold(results):
    r, _ = results
    ranking = r["priorities"]
    top, bottom = ranking.iloc[0], ranking.iloc[-1]
    assert top["priority_score"] >= bottom["priority_score"]
    lo = rank_centres(r["history"], r["forecast"], r["flags"], DecisionParams(threshold=1.0))
    hi = rank_centres(r["history"], r["forecast"], r["flags"], DecisionParams(threshold=30.0))
    assert hi["priority_score"].mean() > lo["priority_score"].mean()


def test_app_starts_on_pipeline_output(results, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    _, out = results
    monkeypatch.setenv("DVSA_DATA", str(out))
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5


@pytest.mark.skipif(not __import__("pathlib").Path("data/DRT122F.ods").exists(), reason="real file not present")
def test_loader_on_real_drt122f():
    tidy, rep = load_centre_table("data/DRT122F.ods", "DRT122F")
    assert rep.mapped["availability_pct"].endswith("(Percentage) [note 13]")
    assert tidy["centre"].nunique() > 300 and tidy["month"].max() >= pd.Timestamp("2026-01-01")
    assert tidy["availability_pct"].between(0, 100).all()
