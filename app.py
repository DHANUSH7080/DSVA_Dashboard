"""Streamlit dashboard: driving test availability, forecasts with ranges, error metrics, ranked shortlist.

Run:  streamlit run app.py -- --data outputs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st  # type: ignore[import-not-found]

from dvsa_forecast.decision import DecisionParams, rank_centres
from dvsa_forecast.models import MODEL_LABELS

INK, BLUE, ORANGE, GREY = "#0F2A3D", "#2F6F9F", "#B4540A", "#8A97A3"

st.set_page_config(page_title="Driving test pressure outlook", layout="wide")


# ---- data ---------------------------------------------------------------------------------------------


def _data_dir() -> Path | None:
    cands = []
    if "--data" in sys.argv:
        cands.append(sys.argv[sys.argv.index("--data") + 1])
    if os.environ.get("DVSA_DATA"):
        cands.append(os.environ["DVSA_DATA"])
    cands += ["outputs", "outputs_demo"]
    for c in cands:
        if (Path(c) / "meta.json").exists():
            return Path(c)
    return None


@st.cache_data(show_spinner=False)
def load(path: str) -> dict:
    d = Path(path)
    dates = {"history": ["month"], "backtest": ["origin", "target_month"], "forecast": ["target_month"]}
    out = {}
    for f in d.glob("*.csv"):
        out[f.stem] = pd.read_csv(f, parse_dates=dates.get(f.stem, []))
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


data_dir = _data_dir()
if data_dir is None:
    st.error(
        "No results found. Create them first:\n\n"
        "`python -m dvsa_forecast.cli demo` (synthetic) or "
        "`python -m dvsa_forecast.cli run DRT122F.ods --out outputs` (real DVSA file)."
    )
    st.stop()

D = load(str(data_dir))
meta = D["meta"]
target = meta["target"]
hist, fc, bt = D["history"], D["forecast"], D["backtest"]
hist["region"] = hist["region"].fillna("Unknown") if "region" in hist else "Unknown"
flags = D["flags"].fillna({"data_flags": ""})
level_pct = int(round(meta["interval_level"] * 100))

# ---- header -------------------------------------------------------------------------------------------

st.title("Driving test availability: pressure outlook by test centre")
if meta["synthetic"]:
    st.error("SYNTHETIC DEMO DATA. Centres and numbers are invented to show how the tool works. This is not DVSA data.")
else:
    st.caption(
        f"Built from DVSA's published statistics (Open Government Licence v3.0), data to {meta['last_data_month'][:7]}. "
        "Independent project; not an official DVSA product."
    )

# ---- sidebar: decision settings -----------------------------------------------------------------------

with st.sidebar:
    st.header("Decision settings")
    st.caption("These are the analyst's assumptions, not DVSA policy. Change them and the ranking updates.")
    thr = st.number_input(
        "'Tight' threshold (% of appointments available)", 0.0, 100.0, float(meta["default_threshold"]), 0.5,
        help="Default is the 25th percentile of centre-months over the latest 12 months.",
    )
    st.subheader("Score weights")
    w1 = st.slider("Forecast shortfall below threshold", 0.0, 1.0, 0.45, 0.05)
    w2 = st.slider("Months tight in the last 12", 0.0, 1.0, 0.30, 0.05)
    w3 = st.slider("Forecast fall versus latest", 0.0, 1.0, 0.15, 0.05)
    w4 = st.slider("Centre size (test slots offered)", 0.0, 1.0, 0.10, 0.05)
    st.subheader("Action cut-offs")
    review_cut = st.slider("Review for extra capacity at score >=", 10, 100, 60, 5)
    watch_cut = st.slider("Watch closely at score >=", 5, 95, 35, 5)
    if watch_cut >= review_cut:
        st.warning("Watch cut-off should be below the review cut-off.")

params = DecisionParams(threshold=thr, w_shortfall=w1, w_persistence=w2, w_deterioration=w3, w_volume=w4,
                        review_cut=float(review_cut), watch_cut=float(watch_cut))
ranking = rank_centres(hist, fc, flags, params, target)

tabs = st.tabs(["Priority ranking", "Centre outlook", "Model quality", "Data checks", "About and limits"])

# ---- tab 1: ranking -----------------------------------------------------------------------------------

with tabs[0]:
    n_review = int((ranking["action"] == "Review for extra examiner capacity").sum())
    n_watch = int((ranking["action"] == "Watch closely").sum())
    n_check = int((ranking["action"] == "Check data first").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Centres modelled", meta["centres_modelled"], help=f"{meta['centres_excluded']} excluded for short history")
    c2.metric("Review for extra capacity", n_review)
    c3.metric("Watch closely", n_watch)
    c4.metric("Check data first", n_check)

    f1, f2, f3 = st.columns(3)
    regions = f1.multiselect("Region", sorted(ranking["region"].dropna().unique()))
    patterns = f2.multiselect("Pattern", sorted(ranking["pattern"].unique()))
    actions = f3.multiselect("Suggested action", sorted(ranking["action"].unique()))
    view = ranking.copy()
    if regions:
        view = view[view["region"].isin(regions)]
    if patterns:
        view = view[view["pattern"].isin(patterns)]
    if actions:
        view = view[view["action"].isin(actions)]

    show = view[["rank", "centre", "region", "pattern", "priority_score", "action", "confidence",
                 "latest", "fc_mean", "fc_lo", "fc_hi", "months_tight_12m"]].rename(columns={
        "priority_score": "score", "fc_mean": "forecast avg", "fc_lo": f"low ({level_pct}%)",
        "fc_hi": f"high ({level_pct}%)", "months_tight_12m": "months tight (of 12)"})
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f")},
    )
    st.download_button("Download full ranking with reasons (CSV)", ranking.to_csv(index=False).encode(),
                       "centre_priority_ranking.csv", "text/csv")

    if not view.empty:
        pick = st.selectbox("Why is a centre ranked here?", view["centre"].tolist())
        row = ranking[ranking["centre"] == pick].iloc[0]
        st.info(f"**{row['action']}** (confidence: {row['confidence']}). {row['reasons']}.")
    st.caption(
        "A shortlist to prompt a human decision. It does not account for local factors such as examiner "
        "leave, planned recruitment or site constraints, which the service team will know and this model cannot."
    )

    with st.expander("Average availability by region"):
        last = hist["month"].max()
        reg = (hist[hist["month"] > last - pd.DateOffset(months=36)]
               .groupby(["region", "month"])[target].mean().reset_index())
        fig = go.Figure()
        for r, g in reg.groupby("region"):
            fig.add_trace(go.Scatter(x=g["month"], y=g[target], name=str(r), mode="lines"))
        fig.update_layout(template="plotly_white", height=360, yaxis_title=target.replace("_", " "),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

# ---- tab 2: one centre --------------------------------------------------------------------------------

with tabs[1]:
    centre = st.selectbox("Test centre", ranking["centre"].tolist(), key="centre_pick")
    row = ranking[ranking["centre"] == centre].iloc[0]
    h = hist[hist["centre"] == centre].sort_values("month")
    f = fc[fc["centre"] == centre].sort_values("horizon")
    months_shown = st.slider("Months of history shown", 12, 84, 36, 6)
    h = h[h["month"] > h["month"].max() - pd.DateOffset(months=months_shown)]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Latest", f"{row['latest']:.1f}%")
    k2.metric("Forecast average", f"{row['fc_mean']:.1f}%")
    k3.metric(f"{level_pct}% range", f"{row['fc_lo']:.1f} to {row['fc_hi']:.1f}%")
    k4.caption("Pattern")
    k4.markdown(f"**{row['pattern']}**")
    k5.caption("Suggested action")
    k5.markdown(f"**{row['action']}**  \nConfidence: {row['confidence']}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["month"], y=h[target], name="Actual", mode="lines+markers",
                             line=dict(color=INK, width=2), marker=dict(size=5)))
    anchor = pd.DataFrame({"target_month": [h["month"].max()], "mean": [h[target].iloc[-1]],
                           "lo": [h[target].iloc[-1]], "hi": [h[target].iloc[-1]]})
    band = pd.concat([anchor, f[["target_month", "mean", "lo", "hi"]]], ignore_index=True)
    fig.add_trace(go.Scatter(x=band["target_month"], y=band["hi"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band["target_month"], y=band["lo"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(47,111,159,0.22)", name=f"{level_pct}% range"))
    fig.add_trace(go.Scatter(x=band["target_month"], y=band["mean"], name="Forecast", mode="lines+markers",
                             line=dict(color=BLUE, width=2, dash="dash")))
    used = f["model_used"].iloc[0] if len(f) else None
    one = bt[(bt["centre"] == centre) & (bt["model"] == used) & (bt["horizon"] == 1)]
    one = one[one["target_month"] > h["month"].min()]
    if not one.empty:
        fig.add_trace(go.Scatter(x=one["target_month"], y=one["forecast"], mode="markers",
                                 name="What the model said one month ahead", marker=dict(symbol="x", size=8, color=ORANGE)))
    fig.add_hline(y=thr, line=dict(color=GREY, dash="dot"), annotation_text=f"tight threshold {thr:.1f}%",
                  annotation_position="top left")
    fig.update_layout(template="plotly_white", height=460, yaxis_title=target.replace("_", " ") + " (%)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.write(f"**Why:** {row['reasons']}.")

    st.subheader("How well did the models do for this centre?")
    mc = D["metrics_by_centre"]
    mc = mc[mc["centre"] == centre].copy()
    mc["model"] = mc["model"].map(MODEL_LABELS)
    mc["used for forecast"] = np.where(mc["model"] == MODEL_LABELS.get(used, ""), "yes", "")
    st.dataframe(mc.drop(columns="centre").round(3).rename(columns={"n": "forecasts scored"}), hide_index=True,
                 width="stretch")
    st.caption("MAE and RMSE are in percentage points. MASE below 1 means better than carrying the previous month forward.")

# ---- tab 3: model quality -----------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Does the model beat a simple baseline?")
    sel = D["selection_eval"]
    if not sel.empty:
        s = sel.copy()
        s.columns = ["Approach", "Forecasts scored", "MAE (points)", "MASE", "Bias (points)", "Improvement on baseline (%)"]
        st.dataframe(s.round(3), hide_index=True, width="stretch")
        st.caption(
            "Models were chosen on earlier months only and scored on later months, so these numbers are not flattered "
            "by choosing the winner on the same data it is judged on. A positive improvement means it beat the baseline."
        )

    st.subheader("Error by model and horizon")
    metric = st.radio("Metric", ["mae", "rmse", "mase", "bias"], horizontal=True,
                      format_func=lambda m: {"mae": "MAE", "rmse": "RMSE", "mase": "MASE", "bias": "Bias"}[m])
    mh = D["metrics_by_model_horizon"].copy()
    fig = go.Figure()
    palette = {"naive": GREY, "seasonal_naive": "#C9A66B", "ets": BLUE, "ets_seasonal": INK}
    for m, g in mh.groupby("model"):
        fig.add_trace(go.Bar(x=[f"{int(x)} month ahead" if x == 1 else f"{int(x)} months ahead" for x in g["horizon"]],
                             y=g[metric], name=MODEL_LABELS.get(m, m), marker_color=palette.get(m)))
    fig.update_layout(barmode="group", template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title=metric.upper())
    st.plotly_chart(fig, width="stretch")
    t = mh.copy()
    t["model"] = t["model"].map(MODEL_LABELS)
    st.dataframe(t.round(3), hide_index=True, width="stretch")

    st.subheader(f"Do the {level_pct}% ranges hold up?")
    cov = D["coverage"].copy()
    model_pick = st.selectbox("Model", list(MODEL_LABELS), index=list(MODEL_LABELS).index(meta["champion"])
                              if meta["champion"] in MODEL_LABELS else 0, format_func=lambda m: MODEL_LABELS[m])
    cv = cov[cov["model"] == model_pick][["horizon", "nominal", "observed", "mean_half_width", "n"]]
    cv = cv.rename(columns={"nominal": "target coverage", "observed": "actual coverage on later months",
                            "mean_half_width": "average half-width (points)", "n": "forecasts scored"})
    st.dataframe(cv.round(3), hide_index=True, width="stretch")
    gap = (cov[cov["model"] == model_pick]["nominal"] - cov[cov["model"] == model_pick]["observed"]).mean()
    if pd.notna(gap) and gap > 0.05:
        st.warning(f"Ranges covered the real result about {gap * 100:.0f} points less often than intended on later months. "
                   "Treat them as optimistic: conditions changed after the ranges were set.")
    elif pd.notna(gap):
        st.success("Ranges covered the real result about as often as intended on later months.")

    st.subheader("Error by region")
    reg = D["metrics_by_region"]
    reg = reg[reg["model"] == model_pick].drop(columns="model").round(3)
    st.dataframe(reg, hide_index=True, width="stretch")
    with st.expander("What these measures mean"):
        st.markdown(
            "- **MAE**: average miss, in percentage points.\n"
            "- **RMSE**: like MAE but punishes big misses more.\n"
            "- **MASE**: miss divided by the centre's typical month-to-month change. Below 1 = better than carrying last month forward.\n"
            "- **Bias**: average of forecast minus actual. Negative = tends to forecast too low.\n"
            "- **Coverage**: how often the real result fell inside the range."
        )

# ---- tab 4: data checks -------------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Automated data checks")
    q = D["quality"].copy().fillna("")
    st.dataframe(q, hide_index=True, width="stretch")
    st.caption("Checks report problems; they never silently delete or repair data. Impossible values are left out of "
               "modelling and listed here.")
    fl = D["flags"].fillna("")
    fl = fl[fl["data_flags"] != ""]
    st.write(f"**Centres with data flags: {len(fl)}** (these get low confidence and 'check data first')")
    if len(fl):
        st.dataframe(fl, hide_index=True, width="stretch")

# ---- tab 5: about -------------------------------------------------------------------------------------

with tabs[4]:
    st.markdown(
        f"""
### What this is
An independent open-data project that forecasts driving test availability for each test centre a few months ahead,
shows how wrong the forecasts have been in the past, and ranks centres to prompt a human decision about where
extra examiner capacity might help. Data source: DVSA published statistics (Open Government Licence v3.0).

### How it works
1. **Data model**: one tidy row per centre per month, with automated checks.
2. **Forecast**: four models, simplest first. A new model only ships for a centre if it beat the baseline in the backtest.
3. **Error**: a rolling backtest forecasts from many past months and compares with what happened.
4. **Ranges**: based on the model's own past errors at each horizon ({level_pct}% target), with real coverage checked.
5. **Ranking**: a transparent score from forecast shortfall, months tight, forecast fall and centre size. Weights are in the sidebar.

### Limits
- Public data is monthly and aggregated by centre. It cannot show where learners live versus where they test,
  when slots are released, examiner rosters, or why people miss tests.
- Every centre was subject to the 2026 booking-rule changes, so there is no untouched comparison group.
  Availability can rise because more tests are provided, not only because behaviour changed.
- The threshold, weights and cut-offs are assumptions to be agreed with the service, not DVSA policy.
- A person should review every recommendation.

### Run details
Created {meta['created_utc']}. Target: `{target}`. Horizon: {meta['horizons']} months.
Baseline: `{meta['baseline']}`. Best overall model: `{meta['champion']}`.
Backtest origins: {meta['n_origins']}. Source: {meta['source']}.
"""
    )
