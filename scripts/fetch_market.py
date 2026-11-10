#!/usr/bin/env python3
"""
cron_c: fetch_market.py
Fetches top 20 IDR pairs by 24h volume + order book depth for each pair.
Schedule: every minute (* * * * *)
Output:
  data/pairs.json          — top 20 pairs list
  data/depth/<pair_id>.json — order book snapshot per pair
"""

import json
import os
import time
import requests

BASE_URL = "https://indodax.com"
# Paths from paths.py - see that module for why this is centralised.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PAIRS_FILE, DEPTH_DIR
TOP_N = 20
SLEEP_BETWEEN_DEPTH = 0.1  # seconds — stays well under 180 req/min

STABLECOIN_KEYWORDS = {"usdt", "usdc", "bidr", "dai", "busd", "idrt", "usd"}
GOLD_KEYWORDS = {"xau", "xaut", "gold"}


def is_excluded(traded_currency: str) -> bool:
    tc = traded_currency.lower()
    return tc in STABLECOIN_KEYWORDS or tc in GOLD_KEYWORDS


def fetch_pairs() -> list[dict]:
    """Fetch top 20 pairs by 24h IDR volume, excluding stablecoins and gold."""
    pairs_resp = requests.get(f"{BASE_URL}/api/pairs", timeout=10)
    pairs_resp.raise_for_status()
    pairs_data = pairs_resp.json()

    summaries_resp = requests.get(f"{BASE_URL}/api/summaries", timeout=10)
    summaries_resp.raise_for_status()
    summaries = summaries_resp.json().get("tickers", {})

    # Price increments (tick) per pair — used by the Hunter for tick-snapping
    # trigger levels (SL, rungs) and buy limit price (best bid + 1 tick)
    increments = {}
    try:
        incr_resp = requests.get(f"{BASE_URL}/api/price_increments", timeout=10)
        incr_resp.raise_for_status()
        increments = incr_resp.json().get("increments", {})
    except Exception as e:
        print(f"[WARN] price_increments failed: {e} — tick defaults to 1")

    result = []
    for p in pairs_data:
        tc = p.get("traded_currency", "")
        bc = p.get("base_currency", "")

        if bc != "idr":
            continue
        if is_excluded(tc):
            continue

        ticker_id = p.get("ticker_id", "")
        summary = summaries.get(ticker_id, {})
        vol_idr_str = summary.get("vol_idr", "0")
        try:
            vol_idr = float(vol_idr_str)
        except (ValueError, TypeError):
            vol_idr = 0.0

        # Tick: minimum price step for this pair. Keyed by ticker_id in the
        # increments map (e.g. "btc_idr": "1000"). Fallback 1 if missing.
        tick_str = increments.get(ticker_id, "1")
        try:
            tick = float(tick_str)
        except (ValueError, TypeError):
            tick = 1.0

        # Build symbol correctly — never trust the 'symbol' field from API
        # Known bug: some pairs return malformed symbol (e.g. GLIDRIDR instead of GLIDIDR)
        symbol = tc.upper() + bc.upper()

        result.append({
            "id": p.get("id", ""),
            "ticker_id": ticker_id,
            "symbol": symbol,
            "name": p.get("traded_currency_unit", tc.upper()),
            "traded_currency": tc,
            "base_currency": bc,
            "vol_idr": vol_idr,
            "tick": tick,
            # Buy-sizing fields (Step 0) — from /api/pairs, zero extra calls
            "quantity_increment": p.get("quantity_increment", 1e-8),
            "trade_min_base_currency": p.get("trade_min_base_currency", 10000),
            "trade_min_traded_currency": p.get("trade_min_traded_currency", 0.0),
            "trade_fee_percent_maker": p.get("trade_fee_percent_maker", 0.1),
            "trade_fee_percent_taker": p.get("trade_fee_percent_taker", 0.2),
        })

    result.sort(key=lambda x: x["vol_idr"], reverse=True)
    return result[:TOP_N]


def fetch_depth(pairs: list[dict]) -> None:
    """Fetch and save order book depth snapshot for each pair."""
    os.makedirs(DEPTH_DIR, exist_ok=True)
    for pair in pairs:
        pair_id = pair["id"]
        try:
            resp = requests.get(f"{BASE_URL}/api/depth/{pair_id}", timeout=10)
            resp.raise_for_status()
            depth = resp.json()
            out_path = os.path.join(DEPTH_DIR, f"{pair_id}.json")
            with open(out_path, "w") as f:
                json.dump(depth, f)
        except Exception as e:
            print(f"[WARN] depth {pair_id}: {e}")
        time.sleep(SLEEP_BETWEEN_DEPTH)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    print("[cron_c] fetching top pairs...")
    pairs = fetch_pairs()
    with open(PAIRS_FILE, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[cron_c] saved {len(pairs)} pairs → {PAIRS_FILE}")

    print("[cron_c] fetching depth...")
    fetch_depth(pairs)
    print(f"[cron_c] depth done for {len(pairs)} pairs")


if __name__ == "__main__":
    main()
