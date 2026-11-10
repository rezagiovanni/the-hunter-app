#!/usr/bin/env python3
"""
cron_e: backfill_deep.py
Deep backfill for pairs that entered the top 20 AFTER the original
2026-01-01 backfill.

Logic:
  - For every CSV in data/ohlcv/, read its FIRST bar timestamp.
  - If first bar > 2026-01-01 (the deep floor), the pair is "shallow" —
    it was never part of the original deep backfill. Fetch from
    2026-01-01 → now and merge.
  - Pairs already at/below the floor are skipped (zero API cost).
  - A pair listed later than 2026-01-01 gets data from its listing date
    (the API simply returns nothing before that) — the floor is fixed at
    2026-01-01 regardless.

Schedule: 30 2 * * * (daily 02:30)
Chunking: 7-day windows (history_v2 silently returns WRONG DATA above
  673 bars — verified pitfall; 7d = 672 bars, just under the cap).
Output report: data/reports/cron_e_report.json + data/reports/cron_e_latest.md

First run will hit all currently-shallow pairs (~28 × ~39 chunks).
Steady state: only genuinely new pairs, ~1-3 per day.
"""

import csv
import datetime as dt
import glob
import json
import os
import tempfile
import time
import requests

BASE_URL = "https://indodax.com"
# Paths from paths.py - see that module for why this is centralised.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, OHLCV_DIR, REPORTS_DIR as REPORT_DIR

import config as _config  # noqa: E402

# CSV reading/writing lives in ohlcv_io so there is exactly one implementation.
# It was copied into five modules and four of the copies wrote straight over the
# live file, which would silently truncate a pair's entire history on a kill.
from ohlcv_io import CSV_COLUMNS, read_csv, write_csv  # noqa: F401

# The deep floor is ROLLING, not a fixed date. It used to be the literal
# "2021-01-01", which meant the fetch window silently widened every year: by
# 2026 it was pulling 5.8 years of history for a product that only needs 2, and
# the gap grew forever.
#
# It is also deliberately deeper than the prune cutoff by _config's
# backfill_slack_days. If the two matched exactly, prune would trim to the floor
# and backfill would immediately re-fetch the same bars, every run, for every
# pair.
_DATA_CFG = _config.current()
RETENTION_YEARS = float(_config.get(_DATA_CFG, "data.retention_years", 2.0))
_SLACK_DAYS = float(_config.get(_DATA_CFG, "data.backfill_slack_days", 30))


def deep_floor_ts(now=None):
    """Unix seconds: the oldest bar worth fetching."""
    import datetime as _dt
    now = now if now is not None else _dt.datetime.now(_dt.timezone.utc)
    return int((now - _dt.timedelta(days=RETENTION_YEARS * 365.25 + _SLACK_DAYS))
               .timestamp())


def deep_floor_date(now=None):
    """Human-readable form of deep_floor_ts()."""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(deep_floor_ts(now), _dt.timezone.utc).strftime("%Y-%m-%d")


# Kept for any caller that imported the old name.
DEEP_FLOOR_DATE = deep_floor_date()
TIMEFRAME = "15"
CHUNK_DAYS = 7                    # 672 bars — under the 673-bar silent cap
CHUNK_SECONDS = CHUNK_DAYS * 86400
SLEEP_BETWEEN_REQUESTS = 0.35




def fetch_bars(symbol: str, from_ts: int, to_ts: int) -> list[dict]:
    """Fetch bars in 7-day chunks. Returns list of row dicts."""
    all_rows = []
    cursor = from_ts
    while cursor < to_ts:
        chunk_end = min(cursor + CHUNK_SECONDS, to_ts)
        url = (
            f"{BASE_URL}/tradingview/history_v2"
            f"?symbol={symbol}&tf={TIMEFRAME}&from={cursor}&to={chunk_end}"
        )
        try:
            resp = requests.get(url, timeout=20)
            if not resp.ok or resp.text.strip().startswith("error"):
                print(f"  [WARN] {symbol} {cursor}-{chunk_end}: {resp.status_code} {resp.text[:60]}")
                cursor = chunk_end
                time.sleep(1)
                continue
            bars = resp.json()
            if isinstance(bars, list):
                for b in bars:
                    all_rows.append({
                        "timestamp": int(b["Time"]),
                        "open":      float(b["Open"]),
                        "high":      float(b["High"]),
                        "low":       float(b["Low"]),
                        "close":     float(b["Close"]),
                        "volume":    float(b["Volume"]),
                    })
        except Exception as e:
            print(f"  [WARN] {symbol} {cursor}-{chunk_end}: {e}")
        cursor = chunk_end
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return all_rows


def main():
    floor_ts = deep_floor_ts()
    floor_label = deep_floor_date()
    now_ts = int(time.time())

    csv_files = sorted(glob.glob(os.path.join(OHLCV_DIR, "*.csv")))
    if not csv_files:
        print("[cron_e] no CSV files found — nothing to do")
        return

    print(f"[cron_e] scanning {len(csv_files)} CSVs (deep floor {floor_label})...")

    pair_stats = []
    processed = 0
    total_added = 0

    # Best-effort progress reporting. A two-year backfill runs for tens of
    # minutes and used to show nothing but "running" for the whole time; the
    # panel has a progress bar, and nothing ever filled it. Reporting is
    # deliberately non-fatal: this script must still work when run by cron,
    # where setup_state may not be importable at all.
    def _report(done):
        try:
            import setup_state
            setup_state.set_progress("backfill-deep", done, len(csv_files),
                                     f"{done}/{len(csv_files)} pairs")
        except Exception:
            pass

    _report(0)

    for _idx, csv_path in enumerate(csv_files, 1):
        # Reported at the top of the iteration for the pair just finished, so
        # every `continue` path below is covered without repeating the call.
        _report(_idx - 1)
        symbol = os.path.splitext(os.path.basename(csv_path))[0]
        existing = read_csv(csv_path)

        if not existing:
            print(f"[cron_e] {symbol}: empty CSV, skipping")
            pair_stats.append({
                "symbol": symbol, "status": "EMPTY", "old_bars": 0,
                "new_bars": 0, "first_bar": None, "bars_added": 0,
            })
            continue

        first_ts = existing[0]["timestamp"]  # CSVs are sorted ascending
        first_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(first_ts))

        if first_ts <= floor_ts:
            # Already deep enough — zero API cost, skip.
            continue

        # Shallow pair: fetch from the deep floor → now.
        print(f"[cron_e] {symbol}: shallow (first bar {first_str}) — backfilling from {floor_label}")
        fetched = fetch_bars(symbol, floor_ts, now_ts)
        old_count = len(existing)

        if not fetched:
            print(f"  0 bars fetched (kept {old_count})")
            pair_stats.append({
                "symbol": symbol, "status": "NO_DATA", "old_bars": old_count,
                "new_bars": old_count, "first_bar": first_str, "bars_added": 0,
            })
            continue

        merged = {r["timestamp"]: r for r in existing}
        for r in fetched:
            merged[r["timestamp"]] = r
        write_csv(csv_path, list(merged.values()))

        added = len(merged) - old_count
        new_first = time.strftime("%Y-%m-%d %H:%M", time.gmtime(min(merged.keys())))
        total_added += added
        processed += 1
        print(f"  {old_count} → {len(merged)} bars (+{added}), history from {new_first}")
        pair_stats.append({
            "symbol": symbol, "status": "DEEP_BACKFILLED", "old_bars": old_count,
            "new_bars": len(merged), "first_bar": new_first, "bars_added": added,
        })

    # Report artifact (user requirement: repair jobs leave a verifiable file)
    _report(len(csv_files))
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_ts)),
        "deep_floor": floor_label,
        "summary": {
            "total_csvs": len(csv_files),
            "pairs_processed": processed,
            "bars_added": total_added,
        },
        "pairs": pair_stats,
    }
    with open(os.path.join(REPORT_DIR, "cron_e_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    lines = [
        "# Cron E: Deep Backfill Report",
        f"Generated at: {report_data['generated_at']}",
        f"Deep floor: {floor_label} | Pairs processed: {processed} | Bars added: {total_added}",
        "",
        "| Symbol | Status | Old Bars | New Bars | First Bar | Added |",
        "|---|---|---|---|---|---|",
    ]
    for p in pair_stats:
        lines.append(
            f"| {p['symbol']} | {p['status']} | {p['old_bars']} | {p['new_bars']} | {p['first_bar']} | {p['bars_added']} |"
        )
    lines.append("")
    with open(os.path.join(REPORT_DIR, "cron_e_latest.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"[cron_e] done — {processed} pair(s) backfilled, +{total_added} bars")
    print(f"[cron_e] report saved: {REPORT_DIR}/cron_e_latest.md")


if __name__ == "__main__":
    main()
