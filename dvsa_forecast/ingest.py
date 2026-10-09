"""Read DVSA's published .ods tables into the tidy contract.

Verified against DRT122F (car driving test availability by driving test centre, Jan 2019 - Aug 2026).
The header row is detected, "[note N]" suffixes are stripped, and columns are mapped by keyword; the
mapping is printed on every run. If DVSA renames a column, pass ``--map`` (see README) or run ``inspect``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import MEASURES


# ---- header -> field rules, evaluated in order; the first unclaimed match wins ---------------------


def _has(h: str, *words: str) -> bool:
    return all(w in h for w in words)


RULES = [
    ("share_gt_21w", lambda h: "more than 21" in h),
    ("share_14_21w", lambda h: "more than 14" in h and "21" in h),
    ("share_7_14w", lambda h: "more than 7" in h and "14" in h),
    ("share_le_7w", lambda h: "within 7 weeks" in h),
    ("wait_weeks_to_10pct", lambda h: "weeks to 10%" in h),
    ("median_wait_weeks", lambda h: "median" in h),
    ("overtime_slots", lambda h: "overtime" in h),
    # the three slot counts and the percentage share share most of their wording: test the specific ones first
    ("availability_pct", lambda h: "still available" in h and "percentage" in h),
    ("slots_available", lambda h: "still available" in h),
    ("slots_booked", lambda h: "tests booked" in h),
    ("slots_total", lambda h: "made available" in h),
    ("country", lambda h: h == "country"),
    ("region", lambda h: h == "region"),
    ("centre", lambda h: "centre" in h or "center" in h),
    ("month", lambda h: h in ("month", "date", "period")),
]


def _norm(x) -> str:
    """Lower-case, collapse spaces and drop footnote tags such as ' [note 13]' or '[notes 1, 2]'."""
    t = re.sub(r"\[notes?[^\]]*\]", "", str(x), flags=re.I)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass
class LoadReport:
    sheet: str = ""
    header_row: int = -1
    mapped: dict = field(default_factory=dict)
    unmapped: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def text(self) -> str:
        lines = [f"Sheet: {self.sheet!r}, header on row {self.header_row + 1}"]
        lines += [f"  {k:22s} <- {v!r}" for k, v in self.mapped.items()]
        if self.unmapped:
            lines.append("  Unmapped columns: " + ", ".join(map(repr, self.unmapped)))
        lines += [f"  NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


# ---- inspection ---------------------------------------------------------------------------------------


def read_all_sheets(path: str | Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, header=None, engine="odf")


def inspect_workbook(path: str | Path, rows: int = 8) -> str:
    """Human-readable summary of every sheet: size and the first rows."""
    out = []
    for name, raw in read_all_sheets(path).items():
        out.append(f"=== sheet {name!r}: {raw.shape[0]} rows x {raw.shape[1]} columns ===")
        out.append(raw.head(rows).to_string(max_colwidth=40))
        out.append("")
    return "\n".join(out)


# ---- header detection --------------------------------------------------------------------------------


def _find_header_row(raw: pd.DataFrame, scan: int = 60) -> int | None:
    best, best_score = None, 0
    for i in range(min(scan, len(raw))):
        cells = [_norm(c) for c in raw.iloc[i].tolist() if isinstance(c, str) and c.strip()]
        if len(cells) < 3:
            continue
        has_centre = any("centre" in c or "center" in c for c in cells)
        has_time = any(c in ("month", "date", "period") for c in cells)
        score = len(cells) + 5 * has_centre + 5 * has_time
        if has_centre and has_time and score > best_score:
            best, best_score = i, score
    return best


def _map_columns(headers: list[str], overrides: dict[str, str] | None) -> tuple[dict, list]:
    mapped: dict[str, str] = {}
    used: set[str] = set()
    if overrides:
        for fld, hdr in overrides.items():
            if hdr in headers:
                mapped[fld] = hdr
                used.add(hdr)
    for hdr in headers:
        if hdr in used:
            continue
        h = _norm(hdr)
        for fld, test in RULES:
            if fld not in mapped and test(h):
                mapped[fld] = hdr
                used.add(hdr)
                break
    unmapped = [h for h in headers if h not in used]
    return mapped, unmapped


# ---- cleaning ----------------------------------------------------------------------------------------


def _to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype("string").str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(txt, errors="coerce")  # suppression markers such as [x] become NaN


def _to_month(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        dt = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
    return dt.dt.to_period("M").dt.to_timestamp()


def load_centre_table(
    path: str | Path,
    sheet: str | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Return (tidy dataframe, report). Raises ValueError with guidance if nothing usable is found."""
    sheets = read_all_sheets(path)
    candidates = {sheet: sheets[sheet]} if sheet else sheets

    best = None
    for name, raw in candidates.items():
        hdr = _find_header_row(raw)
        if hdr is None:
            continue
        body_rows = len(raw) - hdr - 1
        if best is None or body_rows > best[2]:
            best = (name, hdr, body_rows)
    if best is None:
        raise ValueError(
            "Could not find a header row containing both a 'centre' column and a month/date column in any sheet.\n"
            "Run `python -m dvsa_forecast.cli inspect <file>` and check the layout. "
            "If the table is wide (one column per month), reshape it to long form first."
        )
    name, hdr, _ = best
    raw = sheets[name]
    headers = [str(c).strip() if pd.notna(c) else f"unnamed_{i}" for i, c in enumerate(raw.iloc[hdr].tolist())]
    body = raw.iloc[hdr + 1 :].copy()
    body.columns = headers
    body = body.dropna(how="all")

    mapped, unmapped = _map_columns(headers, overrides)
    report = LoadReport(sheet=name, header_row=hdr, mapped=mapped, unmapped=unmapped)
    if "centre" not in mapped or "month" not in mapped:
        raise ValueError("Mapped columns lack centre and/or month.\n" + report.text())

    out = pd.DataFrame(index=body.index)
    out["centre"] = body[mapped["centre"]].astype("string").str.strip()
    out["month"] = _to_month(body[mapped["month"]])
    out["region"] = body[mapped["region"]].astype("string").str.strip() if "region" in mapped else "Unknown"
    if "country" in mapped:
        out["country"] = body[mapped["country"]].astype("string").str.strip()
    # DVSA marks closed centres by prefixing the name with "[z]"; keep the flag, clean the name
    out["closed"] = out["centre"].str.startswith("[z]", na=False)
    out["centre"] = out["centre"].str.replace(r"^\[z\]\s*", "", regex=True).str.strip()
    for fld in MEASURES:
        if fld in mapped:
            out[fld] = _to_number(body[mapped[fld]])

    # footnote rows and blank lines: no usable centre or month
    out = out.dropna(subset=["centre", "month"])
    out = out[out["centre"].str.len() > 0]

    # percentages given as fractions (0-1) are converted to 0-100
    for fld in ("availability_pct", "share_le_7w", "share_7_14w", "share_14_21w", "share_gt_21w"):
        if fld in out and out[fld].notna().any() and out[fld].max() <= 1.0:
            out[fld] = out[fld] * 100
            report.notes.append(f"{fld} looked like a fraction (max <= 1) and was multiplied by 100")

    n_closed = out.loc[out["closed"], "centre"].nunique()
    if n_closed:
        report.notes.append(f"{n_closed} centres are marked closed ([z]); the flag is kept in the 'closed' column")
    missing = [m for m in ("availability_pct",) if m not in out]
    if missing:
        report.notes.append("No availability column was mapped; set it with --map or choose another --target")
    return out.reset_index(drop=True), report


def parse_overrides(text: str | None) -> dict[str, str] | None:
    """Accept a JSON string or a path to a JSON file mapping tidy field -> exact header text."""
    if not text:
        return None
    p = Path(text)
    return json.loads(p.read_text()) if p.exists() else json.loads(text)
