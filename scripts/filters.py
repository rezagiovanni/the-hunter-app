#!/usr/bin/env python3
"""
filters.py — The Hunter's strategy judgment: TREND SCANNER + MOMENTUM QUALITY.

Step 3 of development_plan.txt.

Provides:
  trend_scanner(candles)    — Filter A: direction (UP/DOWN/SIDEWAYS) + score_a (0-100)
  momentum_quality(candles) — Filter B: flags (OVERBOUGHT/OVERSOLD/NEUTRAL) + score_b (0-100)
  combined_score(score_a, score_b) — 0.6 * score_a + 0.4 * score_b
  evaluate_filters(candles) — all-in-one evaluation returning full breakdown dict

Dependencies:
  scripts/indicators.py (Step 2)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure scripts/ directory is on path for indicators import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import (
    ema,
    ema_slope,
    higher_highs_fraction,
    range_position,
    rsi as calc_rsi,
    volume_ratio,
)


def trend_scanner(candles: List[dict]) -> dict:
    """
    TREND SCANNER (Filter A) — 60% weight in final score.

    Evaluates:
      1. EMA(9) vs EMA(21) separation (% diff): max 40 points
      2. EMA(21) slope over 9 bars (%/hour):     max 30 points
      3. Higher-highs fraction over last 24 bars: max 30 points

    Direction:
      UP:       EMA(9) > EMA(21) AND slope > 0
      DOWN:     EMA(9) < EMA(21) AND slope < 0
      SIDEWAYS: anything else

    Score A (0-100):
      Measures upward trend strength. Downward separation or slope gives 0 points.
    """
    default_res = {
        "direction": "SIDEWAYS",
        "score": 0.0,
        "sep": 0.0,
        "slope": 0.0,
        "hh_frac": 0.0,
    }
    if not candles or len(candles) < 30:
        return default_res

    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    if len(ema9) < 2 or len(ema21) < 9:
        return default_res

    latest_ema9 = ema9[-1]
    latest_ema21 = ema21[-1]
    if latest_ema21 == 0:
        return default_res

    sep = (latest_ema9 - latest_ema21) / latest_ema21 * 100.0
    slope = ema_slope(ema21, bars=9, hours=2.0)
    hh_frac = higher_highs_fraction(highs, window=24)

    if sep > 0 and slope > 0:
        direction = "UP"
    elif sep < 0 and slope < 0:
        direction = "DOWN"
    else:
        direction = "SIDEWAYS"

    # Positive trend points only — downward separation or negative slope gives 0 pts
    sep_score = min(40.0, max(0.0, sep) / 2.0 * 40.0)
    slope_score = min(30.0, max(0.0, slope) / 0.5 * 30.0)
    hh_score = hh_frac * 30.0

    score_a = max(0.0, min(100.0, sep_score + slope_score + hh_score))

    return {
        "direction": direction,
        "score": round(score_a, 2),
        "sep": round(sep, 4),
        "slope": round(slope, 4),
        "hh_frac": round(hh_frac, 4),
    }


def momentum_quality(candles: List[dict]) -> dict:
    """
    MOMENTUM QUALITY (Filter B) — 40% weight in final score.

    Evaluates:
      1. RSI(14) sweet spot near 50: max 40 points
      2. Range position (lower is better, room to run): max 30 points
      3. Volume ratio vs 20-bar average: max 30 points

    Flags:
      OVERBOUGHT: RSI > 70
      OVERSOLD:   RSI < 30
      NEUTRAL:    30 <= RSI <= 70

    Penalty:
      If OVERBOUGHT -> score_b is multiplied by 0.4 (heavy 60% penalty).
      Does not hard-reject, but heavily penalizes chasing parabolic tops.
    """
    default_res = {
        "score": 50.0,
        "rsi": None,
        "range_pos": 0.5,
        "vol_ratio": 1.0,
        "flags": ["NEUTRAL"],
        "overbought": False,
    }
    if not candles or len(candles) < 20:
        return default_res

    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]
    lows = [float(c["l"]) for c in candles]
    volumes = [float(c["v"]) for c in candles]

    rsi_val = calc_rsi(closes, 14)
    last_close = closes[-1]
    high_24h = max(highs)
    low_24h = min(lows)

    pos = range_position(last_close, high_24h, low_24h)

    avg_vol_20 = sum(volumes[-20:]) / 20.0
    v_ratio = volume_ratio(volumes[-1], avg_vol_20)

    # Component scores
    # Recalibrated RSI curve: sweet spot 50-60, steep penalty above 65 (overbought)
    if rsi_val is None:
        rsi_score = 50.0
    elif rsi_val <= 50.0:
        rsi_score = max(0.0, 80.0 - (50.0 - rsi_val) * 2.0)
    elif rsi_val <= 60.0:
        rsi_score = 100.0
    elif rsi_val <= 65.0:
        rsi_score = max(0.0, 100.0 - (rsi_val - 60.0) * 8.0)
    else:
        rsi_score = max(0.0, 40.0 - (rsi_val - 65.0) * 4.0)

    pos_score = (1.0 - pos) * 100.0
    vol_score = min(v_ratio / 1.5, 1.0) * 100.0

    score_b = 0.4 * rsi_score + 0.3 * pos_score + 0.3 * vol_score

    # Flags & penalty
    flags = []
    overbought = False
    if rsi_val is not None:
        if rsi_val > 70.0:
            flags.append("OVERBOUGHT")
            overbought = True
        elif rsi_val < 30.0:
            flags.append("OVERSOLD")
        else:
            flags.append("NEUTRAL")
    else:
        flags.append("NEUTRAL")

    if overbought:
        score_b *= 0.4  # heavy penalty

    score_b = max(0.0, min(100.0, score_b))

    return {
        "score": round(score_b, 2),
        "rsi": round(rsi_val, 2) if rsi_val is not None else None,
        "range_pos": round(pos, 4),
        "vol_ratio": round(v_ratio, 4),
        "flags": flags,
        "overbought": overbought,
    }


def combined_score(score_a: float, score_b: float) -> float:
    """Combined strategy score: 0.6 * trend + 0.4 * momentum."""
    return round(0.6 * score_a + 0.4 * score_b, 2)


def evaluate_filters(candles: List[dict]) -> dict:
    """
    Run full filter evaluation on a candle list.
    Returns complete breakdown with combined score and pass/fail verdict.
    """
    trend = trend_scanner(candles)
    mom = momentum_quality(candles)
    total = combined_score(trend["score"], mom["score"])

    # Overextended / pullback guard:
    # Reject entries that have pumped > 1.5% above EMA(9) (exhausted moves)
    closes = [float(c["c"]) for c in candles] if candles else []
    ema9_series = ema(closes, 9) if len(closes) >= 9 else []
    dist_e9 = ((closes[-1] - ema9_series[-1]) / ema9_series[-1] * 100.0) if ema9_series else 0.0
    overextended = dist_e9 > 1.5

    eligible = (trend["direction"] == "UP") and (total >= 60.0) and (not overextended)

    return {
        "direction": trend["direction"],
        "score_a": trend["score"],
        "score_b": mom["score"],
        "total": total,
        "eligible": eligible,
        "overbought": mom["overbought"],
        "overextended": overextended,
        "dist_e9": round(dist_e9, 2),
        "trend_details": trend,
        "momentum_details": mom,
    }


# ---------------------------------------------------------------------------
# Self-check with synthetic data (run directly: python3 filters.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== filters.py self-check ===\n")

    def make_candle(t, o, h, l, c, v):
        return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}

    # 1. Synthetic UP series: steady price rise, rising volume
    up_candles = []
    base = 100.0
    for i in range(50):
        # price goes from 100 to ~150 with higher highs
        c = base + i * 1.0
        h = c + 0.5
        l = c - 0.5
        o = c - 0.2
        v = 1000.0 + i * 20.0
        up_candles.append(make_candle(i * 900, o, h, l, c, v))

    res_up = evaluate_filters(up_candles)
    assert res_up["direction"] == "UP", f"Expected UP, got {res_up['direction']}"
    assert res_up["score_a"] >= 70.0, f"Expected high score_a, got {res_up['score_a']}"
    assert res_up["total"] >= 60.0, f"Expected total >= 60, got {res_up['total']}"
    print(f"[PASS] synthetic UP series: direction={res_up['direction']}, score_a={res_up['score_a']}, total={res_up['total']}")

    # 2. Synthetic FLAT series: constant price, flat volume
    flat_candles = []
    for i in range(50):
        flat_candles.append(make_candle(i * 900, 100.0, 100.0, 100.0, 100.0, 1000.0))

    res_flat = evaluate_filters(flat_candles)
    assert res_flat["direction"] == "SIDEWAYS", f"Expected SIDEWAYS, got {res_flat['direction']}"
    assert res_flat["score_a"] <= 25.0, f"Expected low score_a, got {res_flat['score_a']}"
    assert res_flat["total"] < 60.0, f"Expected total < 60, got {res_flat['total']}"
    print(f"[PASS] synthetic FLAT series: direction={res_flat['direction']}, score_a={res_flat['score_a']}, total={res_flat['total']}")

    # 3. Synthetic DOWN series: falling price, dropping highs
    down_candles = []
    base_down = 150.0
    for i in range(50):
        c = base_down - i * 1.0
        h = c + 0.5
        l = c - 0.5
        o = c + 0.2
        v = 1000.0
        down_candles.append(make_candle(i * 900, o, h, l, c, v))

    res_down = evaluate_filters(down_candles)
    assert res_down["direction"] == "DOWN", f"Expected DOWN, got {res_down['direction']}"
    assert res_down["score_a"] <= 10.0, f"Expected low score_a on down, got {res_down['score_a']}"
    assert res_down["total"] < 40.0, f"Expected low total on down, got {res_down['total']}"
    assert not res_down["eligible"], "DOWN series must not be eligible"
    print(f"[PASS] synthetic DOWN series: direction={res_down['direction']}, score_a={res_down['score_a']}, total={res_down['total']}")

    # 4. RSI 75 (overbought) vs RSI 55: momentum score visibly penalized
    # Build series with mild rise (RSI ~55) vs parabolic spike (RSI ~75+)
    parabolic_candles = []
    for i in range(35):
        c = 100.0 + (i if i < 20 else 20.0 + (i - 20) ** 2.2)
        parabolic_candles.append(make_candle(i * 900, c - 0.5, c + 0.5, c - 0.5, c, 1000.0))

    res_parabolic = momentum_quality(parabolic_candles)
    assert res_parabolic["overbought"], f"Expected overbought flag, RSI={res_parabolic['rsi']}"
    assert "OVERBOUGHT" in res_parabolic["flags"]
    # Score must be heavily penalized (< 35)
    assert res_parabolic["score"] < 35.0, f"Expected heavily penalized score_b, got {res_parabolic['score']}"
    print(f"[PASS] overbought penalty: RSI={res_parabolic['rsi']}, overbought={res_parabolic['overbought']}, penalized score_b={res_parabolic['score']}")

    # 5. Combined math verification: 0.6 * trend + 0.4 * momentum
    comb = combined_score(80.0, 50.0)
    expected_comb = 0.6 * 80.0 + 0.4 * 50.0  # 48.0 + 20.0 = 68.0
    assert abs(comb - expected_comb) < 1e-6, f"Math mismatch: {comb} vs {expected_comb}"
    print(f"[PASS] combined math: 0.6 * 80 + 0.4 * 50 = {comb}")

    # 6. Edge cases: short candle lists (< 30 bars)
    res_short = evaluate_filters(up_candles[:10])
    assert res_short["direction"] == "SIDEWAYS"
    assert not res_short["eligible"]
    print("[PASS] short candle history handled safely")

    # 7. Overextended entry guard: parabolic spike above EMA9 is not eligible
    res_spike = evaluate_filters(parabolic_candles)
    assert res_spike["overextended"], f"Expected overextended=True, dist_e9={res_spike.get('dist_e9')}"
    assert not res_spike["eligible"], "Overextended pump must not be eligible"
    print(f"[PASS] overextended guard: dist_e9={res_spike['dist_e9']}%, eligible={res_spike['eligible']}")

    print("\n=== ALL CHECKS PASSED ===")
