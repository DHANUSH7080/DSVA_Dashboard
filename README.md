# Driving test pressure outlook (open-data project)

Forecasts driving test availability for each test centre a few months ahead, shows how wrong the forecasts
have been in the past, and ranks centres to prompt a human decision about where extra examiner capacity might help.

Built only on statistics DVSA publishes under the Open Government Licence v3.0. An independent project, not an
official DVSA product, and not DVSA policy.

## Status

| Part | State |
|---|---|
| Loader for DVSA's DRT122F `.ods` | Verified against the real file (Jan 2019 to Aug 2026, 29,814 rows, 369 centres). Prints its column mapping on every run |
| Forecasting, backtest, error metrics, prediction ranges, ranking, dashboard | Built; 14 automated tests |
| Real run (`outputs/`) | Done on DRT122F: 314 centres modelled from June 2021, 3 months ahead, 18 backtest origins each |

## What the real run shows (read before quoting anything)

- **A simple "same as last month" forecast is hard to beat.** Naive MAE is 2.1 / 2.9 / 3.4 points at 1 / 2 / 3 months ahead.
  Damped-trend smoothing is close (2.4 / 3.1 / 3.6) and slightly better for some centres. Same-month-last-year is much
  worse (MAE about 9) because availability moved a lot between years.
- **Per-centre selection beat the baseline by about 8% (MASE 0.82 vs 0.89)** when judged on later months only. Modest, not dramatic.
- **The 80% ranges are slightly too narrow.** On later months they covered about 71% / 70% / 77% of outcomes at 1 / 2 / 3 months.
  The dashboard shows this rather than hiding it.
- **The ranking is dominated by small remote Scottish centres** that sit at or near 0% available every month (30 of the 35
  "Review" centres). That reflects a long-standing structural pattern in the data, not a new development, and with
  a floor at 0% those ranges are degenerate. Treat it as a shortlist to question, not a finding.
- Median availability across centres is about 5% (Aug 2026), so the data-driven "tight" threshold (25th percentile) is 2.3%.

## Quick start

```bash
pip install -r requirements.txt

# 1. Download from the GOV.UK page "Driving test and theory test data: cars":
#    DRT122F  (waiting times and availability by driving test centre, monthly)
#    DRT121G  (the same by country and region, optional)

# 2. Look at the file's layout
python -m dvsa_forecast.cli inspect DRT122F.ods

# 3. Run. It prints which column it mapped to each field.
python -m dvsa_forecast.cli run DRT122F.ods --out outputs

# 4. Dashboard
streamlit run app.py -- --data outputs
```

A quick trial: add `--max-centres 15`. A full run takes about 5 minutes. `--start` sets the first month modelled (default 2021-06-01).

If a column is mapped wrongly or missed, override it with the exact header text from the file:

```bash
python -m dvsa_forecast.cli run DRT122F.ods --map '{"availability_pct": "Exact header text here"}'
```

To forecast a different measure, use `--target` (it must be one of the fields in `dvsa_forecast/schema.py`).

Try it with no data: `python -m dvsa_forecast.cli demo` then `streamlit run app.py -- --data outputs_demo`.

## What it does

1. **Data model.** One tidy row per centre per month (`schema.py`), with automated checks (`quality.py`):
   duplicates, impossible values, centres that stop reporting, odd jumps. Checks report; they never silently repair.
2. **Models, simplest first** (`models.py`): naive, same month last year, damped-trend exponential smoothing,
   and the same with yearly seasonality. Each centre uses its latest unbroken run of data, so breaks such as
   the 2020-21 suspensions are not bridged with invented values. Only months from June 2021 are modelled, because
   DVSA's booking window changed from 18 to 24 weeks in 2021, so earlier availability means something different.
   Closed centres (prefixed `[z]` in DVSA's file) and centres with no data in the last 3 months are not forecast.
3. **Rolling backtest** (`backtest.py`). Every model forecasts from many past months and is scored against what
   happened, at 1, 2 and 3 months ahead: MAE, RMSE, MASE and bias.
4. **A model ships only if it earns it.** For each centre the better model is used only if it beat the baseline in
   the backtest. Selection is judged honestly: models are chosen on earlier months and scored on later ones.
5. **Prediction ranges.** Split-conformal ranges from the model's own past errors at each horizon. Their real
   coverage is checked on later months and shown, including when it falls short of the target.
6. **Decision layer** (`decision.py`). A transparent score from forecast shortfall, months tight in the last 12,
   forecast fall, and centre size. Each row gets a pattern, a confidence level, a suggested action and plain-English
   reasons. Centre size uses the standard slots offered. Weights, threshold and cut-offs are in the dashboard sidebar.

## Assumptions to challenge

- The "tight" threshold defaults to the 25th percentile of centre-months in the latest 12 months. It is an analyst's
  choice, not DVSA policy.
- Score weights (0.45 / 0.30 / 0.15 / 0.10) and the action cut-offs (60 / 35) are judgement, and should be agreed
  with the people who make deployment decisions.
- MASE scaling uses each centre's average month-to-month change, with a floor of 0.5 points.

## Limits

- Public data is monthly and aggregated by centre. It cannot show where learners live versus where they test, when
  slots are released, examiner rosters, or why people miss tests.
- Every centre was subject to the 2026 booking-rule changes, so there is no untouched comparison group. A rise in
  availability can reflect more tests being provided, not only changed booking behaviour.
- The output is a prompt for a person, not an instruction. Local knowledge the model cannot see should override it.
- With DVSA's internal data, the same pipeline could estimate true local demand and evaluate interventions such as
  reminder messages against a proper comparison group.

## Layout

```
dvsa_forecast/   schema, ingest, quality, models, backtest, decision, pipeline, synthetic, cli
app.py           Streamlit dashboard
tests/           14 tests: models, no look-ahead, interval coverage, loader, ranking, pipeline, app start-up
outputs_demo/    SYNTHETIC demo results (invented data)
outputs/         results from the real DRT122F file
```

Run the tests with `python -m pytest`.
