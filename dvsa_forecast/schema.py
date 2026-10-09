"""The tidy data contract every part of the project relies on.

One row per test centre per month. Only ``centre`` and ``month`` are mandatory; the
forecasting target (default ``availability_pct``) must be present for a run.
"""

from __future__ import annotations

ID_COLUMNS = ["centre", "month"]

MEASURES: dict[str, str] = {
    "availability_pct": (
        "Share of appointments in the 24-week booking window that were still bookable (%). "
        "Booking window was 18 weeks to May 2021 and 24 weeks after, so only later months are comparable. Higher means less pressure."
    ),
    "wait_weeks_to_10pct": (
        "Weeks until at least 10% of weekly appointments were available (DVSA's long-standing measure)."
    ),
    "median_wait_weeks": "Median weeks between booking and taking the test, for tests taken in the month.",
    "slots_total": "Standard tests made available within the booking window (count, month-end snapshot).",
    "slots_booked": "Standard tests already booked within the booking window (count).",
    "slots_available": "Standard tests still available to book within the booking window (count).",
    "overtime_slots": "Test slots provided through examiner overtime (published from December 2023).",
    "share_le_7w": "Share of tests taken within 7 weeks of booking.",
    "share_7_14w": "Share of tests taken more than 7 and up to 14 weeks after booking.",
    "share_14_21w": "Share of tests taken more than 14 and up to 21 weeks after booking.",
    "share_gt_21w": "Share of tests taken more than 21 weeks after booking.",
}

OPTIONAL_COLUMNS = ["country", "region", "closed"] + [m for m in MEASURES]

DEFAULT_TARGET = "availability_pct"

# Value limits used by the validation checks.
PERCENT_LIKE = ["availability_pct", "share_le_7w", "share_7_14w", "share_14_21w", "share_gt_21w"]
