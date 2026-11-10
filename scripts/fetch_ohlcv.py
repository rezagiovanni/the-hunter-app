#!/usr/bin/env python3
"""
cron_b: fetch_ohlcv.py
Fetches 15m OHLCV candles for top 20 pairs and appends to per-pair CSV files.
Schedule: 1,16,31,46 * * * * (1 min after each 15m bar closes)

Logic per pair:
  - CSV missing → first run: fetch from bot start date (now - 1 bar) to now
  - CSV exists  → incremental: fetch only bars newer than last saved timestamp

Output: data/ohlcv/<SYMBOL>.csv
  columns: timestamp,open,high,low,close,volume
  timestamp: Unix seconds (as returned by history_v2)
  append-only, deduplicated, sorted ascending
"""

import csv
import json
import os
import time
import requests

BASE_URL = "https://indodax.com"
# Paths from paths.py - see that module for why this is centralised.
import sys as _sys

# CSV reading/writing lives in ohlcv_io so there is exactly one implementation.
# It was copied into five modules and four of the copies wrote straight over the
# live file, which would silently truncate a pair's entire history on a kill.
from ohlcv_io import CSV_COLUMNS, read_csv, write_csv  # noqa: F401
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PAIRS_FILE, OHLCV_DIR

TIMEFRAME = "15"           # 15-minute bars
BAR_SECONDS = 15 * 60      # 900 seconds per bar
CHUNK_DAYS = 240           # max chunk to avoid 504 on history_v2
CHUNK_SECONDS = CHUNK_DAYS * 86400
SLEEP_BETWEEN_PAIRS = 0.35  # rate limit safety


def fetch_bars(symbol: str, from_ts: int, to_ts: int) -> list[dict]:
    """Fetch OHLCV bars in <=240-day chunks. Returns list of bar dicts."""
    all_bars = []
    cursor = from_ts
    while cursor < to_ts:
        chunk_end = min(cursor + CHUNK_SECONDS, to_ts)
        url = (
            f"{BASE_URL}/tradingview/history_v2"
            f"?symbol={symbol}&tf={TIMEFRAME}&from={cursor}&to={chunk_end}"
        )
        try:
            resp = requests.get(url, timeout=15)
            # history_v2 returns plain-text "error code: 504" on timeout — not JSON
            if not resp.ok or resp.text.strip().startswith("error"):
                print(f"[WARN] {symbol} chunk {cursor}-{chunk_end}: {resp.status_code} {resp.text[:60]}")
                cursor = chunk_end
                time.sleep(1)
                continue
            bars = resp.json()
            if isinstance(bars, list):
                all_bars.extend(bars)
        except Exception as e:
            print(f"[WARN] {symbol} chunk {cursor}-{chunk_end}: {e}")
        cursor = chunk_end
        time.sleep(SLEEP_BETWEEN_PAIRS)
    return all_bars


def bars_to_rows(bars: list[dict]) -> list[dict]:
    """Convert API bar format to CSV row format."""
    rows = []
    for b in bars:
        rows.append({
            "timestamp": int(b["Time"]),
            "open":      float(b["Open"]),
            "high":      float(b["High"]),
            "low":       float(b["Low"]),
            "close":     float(b["Close"]),
            "volume":    float(b["Volume"]),
        })
    return rows




def process_pair(pair: dict, now_ts: int) -> None:
    symbol = pair["symbol"]
    csv_path = os.path.join(OHLCV_DIR, f"{symbol}.csv")

    if not os.path.exists(csv_path):
        # First run — fetch from one bar ago to now (bot start date)
        from_ts = now_ts - BAR_SECONDS
        print(f"[cron_b] {symbol}: first run, fetching {from_ts} → {now_ts}")
        bars = fetch_bars(symbol, from_ts, now_ts)
        rows = bars_to_rows(bars)
        if rows:
            write_csv(csv_path, rows)
            print(f"[cron_b] {symbol}: wrote {len(rows)} bars")
        else:
            print(f"[cron_b] {symbol}: no bars returned")
        return

    # Incremental — fetch only bars newer than last saved timestamp
    existing = read_csv(csv_path)
    if not existing:
        # CSV exists but empty — treat as first run
        from_ts = now_ts - BAR_SECONDS
    else:
        last_ts = existing[-1]["timestamp"]  # already sorted ascending
        from_ts = last_ts + BAR_SECONDS      # start from next expected bar

    if from_ts >= now_ts:
        print(f"[cron_b] {symbol}: up to date, skipping")
        return

    print(f"[cron_b] {symbol}: incremental {from_ts} → {now_ts}")
    bars = fetch_bars(symbol, from_ts, now_ts)
    new_rows = bars_to_rows(bars)

    if not new_rows:
        print(f"[cron_b] {symbol}: no new bars")
        return

    # Merge, deduplicate by timestamp, sort
    all_rows = {r["timestamp"]: r for r in existing}
    for r in new_rows:
        all_rows[r["timestamp"]] = r
    write_csv(csv_path, list(all_rows.values()))
    print(f"[cron_b] {symbol}: +{len(new_rows)} bars (total {len(all_rows)})")


def main():
    os.makedirs(OHLCV_DIR, exist_ok=True)
    now_ts = int(time.time())

    if not os.path.exists(PAIRS_FILE):
        print(f"[cron_b] ERROR: {PAIRS_FILE} not found — run cron_c first")
        return

    with open(PAIRS_FILE) as f:
        pairs = json.load(f)

    print(f"[cron_b] processing {len(pairs)} pairs at ts={now_ts}")
    for pair in pairs:
        process_pair(pair, now_ts)

    print("[cron_b] done")


if __name__ == "__main__":
    main()
