#!/usr/bin/env python3
"""
cron_d: backfill_stale.py
Scans ALL OHLCV CSVs in data/ohlcv/ and repairs data completeness.

Two jobs:
1. GAP REPAIR   — any missing 15m bar INSIDE the data (>900s between consecutive
                  timestamps). A gap occurs when a pair drops out of the top 20
                  and cron_a stops fetching it.
2. TAIL EXTENSION — fetch all bars from the last stored bar up to "now" for pairs
                  that fell out of the top 20 (cron_a no longer updates them),
                  so their data stays fresh even outside the top-20 universe.

Schedule: 0 0 * * * (daily 00:00)
Scope: every CSV in data/ohlcv/ — not just current top 20
Chunk: <=240 days per request (history_v2 silent cap bug: window MUST be <673 bars)
Output report: data/reports/cron_d_report.json + data/reports/cron_d_latest.md
"""

import csv
import glob
import json
import os
import time
import requests

BASE_URL = "https://indodax.com"
# Paths from paths.py - see that module for why this is centralised.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, OHLCV_DIR, REPORTS_DIR as REPORT_DIR

TIMEFRAME = "15"
BAR_SECONDS = 900          # 15 minutes
CHUNK_DAYS = 240
CHUNK_SECONDS = CHUNK_DAYS * 86400
SLEEP_BETWEEN_REQUESTS = 0.35
CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def read_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": int(row["timestamp"]),
                "open":      float(row["open"]),
                "high":      float(row["high"]),
                "low":       float(row["low"]),
                "close":     float(row["close"]),
                "volume":    float(row["volume"]),
            })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_sorted)


def find_gaps(rows: list[dict]) -> list[tuple[int, int]]:
    """Return list of (gap_start, gap_end) unix timestamp pairs."""
    gaps = []
    for i in range(len(rows) - 1):
        diff = rows[i + 1]["timestamp"] - rows[i]["timestamp"]
        if diff > BAR_SECONDS:
            gaps.append((rows[i]["timestamp"] + BAR_SECONDS, rows[i + 1]["timestamp"]))
    return gaps


def fetch_bars(symbol: str, from_ts: int, to_ts: int) -> list[dict]:
    """Fetch OHLCV bars in <=240-day chunks. Returns list of row dicts."""
    all_rows = []
    cursor = from_ts
    while cursor < to_ts:
        chunk_end = min(cursor + CHUNK_SECONDS, to_ts)
        url = (
            f"{BASE_URL}/tradingview/history_v2"
            f"?symbol={symbol}&tf={TIMEFRAME}&from={cursor}&to={chunk_end}"
        )
        try:
            resp = requests.get(url, timeout=15)
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


def process_file(csv_path: str, now_ts: int) -> dict:
    symbol = os.path.splitext(os.path.basename(csv_path))[0]
    stats = {
        "symbol": symbol,
        "total_bars": 0,
        "first_bar": None,
        "last_bar": None,
        "last_bar_age_hours": None,
        "gaps_found": 0,
        "bars_added": 0,
        "tail_missing_hours": 0.0,
        "tail_bars_added": 0,
        "completeness_pct": 100.0,
        "status": "OK",
    }

    rows = read_csv(csv_path)
    if not rows:
        stats["status"] = "EMPTY"
        return stats

    stats["total_bars"] = len(rows)
    stats["first_bar"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(rows[0]["timestamp"]))
    stats["last_bar"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(rows[-1]["timestamp"]))
    age_h = round((now_ts - rows[-1]["timestamp"]) / 3600, 2)
    stats["last_bar_age_hours"] = age_h

    if len(rows) < 2:
        stats["status"] = "TOO_FEW_BARS"
        return stats

    span_bars = ((rows[-1]["timestamp"] - rows[0]["timestamp"]) // BAR_SECONDS) + 1
    gaps = find_gaps(rows)
    stats["gaps_found"] = len(gaps)

    # Tail extension: bars missing AFTER the last stored bar, up to now.
    now_grid = (now_ts // BAR_SECONDS) * BAR_SECONDS  # align down to the 15m grid
    tail_start = rows[-1]["timestamp"] + BAR_SECONDS
    tail_end = now_grid
    tail_hours = max(0.0, (tail_end - rows[-1]["timestamp"]) / 3600.0)
    stats["tail_missing_hours"] = round(tail_hours, 2)
    tail_needed = tail_hours > 0.0 and tail_start <= tail_end

    new_rows = []

    # 1) Gap repair (inside the data)
    if gaps:
        print(f"[cron_d] {symbol}: {len(gaps)} gap(s) found — backfilling...")
        for gap_start, gap_end in gaps:
            from_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(gap_start))
            to_str   = time.strftime("%Y-%m-%d %H:%M", time.gmtime(gap_end))
            duration_hours = (gap_end - gap_start) / 3600
            print(f"  gap: {from_str} → {to_str} ({duration_hours:.1f}h)")
            fetched = fetch_bars(symbol, gap_start, gap_end)
            new_rows.extend(fetched)
            print(f"  fetched {len(fetched)} bars")

    # 2) Tail extension (after last bar → now)
    if tail_needed:
        from_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(tail_start))
        to_str   = time.strftime("%Y-%m-%d %H:%M", time.gmtime(tail_end))
        print(f"[cron_d] {symbol}: tail {from_str} → {to_str} ({tail_hours:.1f}h)")
        fetched = fetch_bars(symbol, tail_start, tail_end)
        stats["tail_bars_added"] = len(fetched)
        new_rows.extend(fetched)
        print(f"  fetched {len(fetched)} bars")

    if new_rows:
        merged = {r["timestamp"]: r for r in rows}
        for r in new_rows:
            merged[r["timestamp"]] = r
        write_csv(csv_path, list(merged.values()))
        stats["bars_added"] = len(new_rows)
        stats["total_bars"] = len(merged)
        if gaps and tail_needed:
            stats["status"] = "PATCHED+TAIL"
        elif tail_needed:
            stats["status"] = "TAIL_EXTENDED"
        else:
            stats["status"] = "PATCHED"

    rem_gaps = find_gaps(read_csv(csv_path))
    new_span = ((rows[-1]["timestamp"] - rows[0]["timestamp"]) // BAR_SECONDS) + 1
    pct = round((stats["total_bars"] / new_span) * 100, 2) if new_span else 100.0
    stats["completeness_pct"] = min(pct, 100.0)
    if rem_gaps:
        stats["status"] = "PARTIAL"

    return stats


def generate_reports(report_data: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    json_path = os.path.join(REPORT_DIR, "cron_d_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    md_path = os.path.join(REPORT_DIR, "cron_d_latest.md")
    lines = [
        "# Cron D: Data Completeness Report",
        f"Generated at: {report_data['generated_at']}",
        f"Total pairs: {report_data['summary']['total_pairs']} | Gaps: {report_data['summary']['total_gaps']} | Tail extensions: {report_data['summary']['total_tail_extensions']} | Bars added: {report_data['summary']['total_bars_added']}",
        "",
        "| Symbol | Status | Total Bars | First Bar | Last Bar | Age (h) | Gaps | Gap Bars | Tail (h) | Tail Bars | Completeness |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in report_data["pairs"]:
        lines.append(
            f"| {p['symbol']} | {p['status']} | {p['total_bars']} | {p['first_bar']} | {p['last_bar']} | {p['last_bar_age_hours']} | {p['gaps_found']} | {p['bars_added'] - p['tail_bars_added']} | {p['tail_missing_hours']} | {p['tail_bars_added']} | {p['completeness_pct']}% |"
        )
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[cron_d] reports saved: {json_path} and {md_path}")


def main():
    csv_files = sorted(glob.glob(os.path.join(OHLCV_DIR, "*.csv")))
    if not csv_files:
        print("[cron_d] no CSV files found in data/ohlcv/ — nothing to do")
        return

    now_ts = int(time.time())
    print(f"[cron_d] scanning {len(csv_files)} CSV files (gap repair + tail extension)...")

    pair_stats = []
    total_gaps = 0
    total_tail_ext = 0
    total_added = 0

    for csv_path in csv_files:
        stat = process_file(csv_path, now_ts)
        pair_stats.append(stat)
        total_gaps += stat["gaps_found"]
        if stat["tail_bars_added"] > 0:
            total_tail_ext += 1
        total_added += stat["bars_added"]

    report_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_ts)),
        "summary": {
            "total_pairs": len(csv_files),
            "total_gaps": total_gaps,
            "total_tail_extensions": total_tail_ext,
            "total_bars_added": total_added,
        },
        "pairs": pair_stats,
    }

    generate_reports(report_data)
    print("[cron_d] done")


if __name__ == "__main__":
    main()
