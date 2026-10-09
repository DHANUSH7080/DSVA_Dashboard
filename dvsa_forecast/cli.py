"""Command line: inspect a DVSA workbook, run the pipeline, or run it on synthetic demo data.

  python -m dvsa_forecast.cli inspect DRT122F.ods
  python -m dvsa_forecast.cli run DRT122F.ods --out outputs
  python -m dvsa_forecast.cli demo --out outputs_demo
"""

from __future__ import annotations

import argparse
import sys

from .ingest import inspect_workbook, load_centre_table, parse_overrides
from .pipeline import run_pipeline
from .schema import DEFAULT_TARGET
from .synthetic import make_synthetic


def _progress(msg: str, frac: float) -> None:
    sys.stderr.write(f"\r{msg:<28s} {frac * 100:5.1f}%")
    if frac >= 1:
        sys.stderr.write("\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dvsa_forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("inspect", help="print every sheet's size and first rows")
    p_in.add_argument("file")

    p_run = sub.add_parser("run", help="load a DVSA .ods and run the full pipeline")
    p_run.add_argument("file")
    p_run.add_argument("--out", default="outputs")
    p_run.add_argument("--sheet", default=None)
    p_run.add_argument("--map", default=None, help="JSON string or file: {tidy_field: 'exact header text'}")
    p_run.add_argument("--target", default=DEFAULT_TARGET)
    p_run.add_argument("--horizons", type=int, default=3)
    p_run.add_argument("--origins", type=int, default=18)
    p_run.add_argument("--level", type=float, default=0.80)
    p_run.add_argument("--max-centres", type=int, default=None, help="quick trial on the first N centres")
    p_run.add_argument("--start", default="2021-06-01",
                       help="first month to model (booking window changed 18->24 weeks in 2021); '' = all")

    p_demo = sub.add_parser("demo", help="run on SYNTHETIC data (not DVSA data)")
    p_demo.add_argument("--out", default="outputs_demo")
    p_demo.add_argument("--centres", type=int, default=40)

    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        print(inspect_workbook(a.file))
        return 0

    if a.cmd == "run":
        tidy, report = load_centre_table(a.file, a.sheet, parse_overrides(a.map))
        print(report.text())
        print(f"\nLoaded {len(tidy):,} rows, {tidy['centre'].nunique()} centres, "
              f"{tidy['month'].min().date()} to {tidy['month'].max().date()}")
        run_pipeline(tidy, a.out, a.target, a.horizons, a.origins, a.level, a.max_centres, a.start or None,
                     synthetic=False, source=str(a.file), progress=_progress)
        print(f"Wrote results to {a.out}/. Start the dashboard with: streamlit run app.py -- --data {a.out}")
        return 0

    tidy = make_synthetic(a.centres, inject_issues=True)
    run_pipeline(tidy, a.out, synthetic=True, source="synthetic", progress=_progress)
    print(f"Wrote SYNTHETIC demo results to {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
