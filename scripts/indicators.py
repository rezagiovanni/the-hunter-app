#!/usr/bin/env python3
"""
indicators.py — Pure technical indicator math for The Hunter.

Pure functions only: inputs are numbers / lists of numbers, output is a number.
No file reads, no API calls, no state, no side effects.

Functions:
  ema(values, period)                 — exponential moving average series
  ema_slope(ema_series, bars=9, hours=2.0) — slope normalized to %/hour
  rsi(closes, period=14)              — Wilder's RSI (0-100, None if insufficient data)
  range_position(close, high_24h, low_24h) — 0.0 (at low) to 1.0 (at high)
  volume_ratio(last_vol, avg_vol_20)  — last bar vol / 20-bar avg
  higher_highs_fraction(highs, window=24) — fraction of bars with high > prev_high
"""

from typing import List, Optional


def ema(values: List[float], period: int) -> List[float]:
    """
    Exponential Moving Average series.
    First value is SMA of the first `period` bars.
    Returns empty list if len(values) < period.
    """
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


def ema_slope(ema_series: List[float], bars: int = 9, hours: float = 2.0) -> float:
    """
    Rate of change of EMA over the last `bars` points, normalized to %/hour.
    Default bars=9 corresponds to 9 x 15m = 2.25h (approx 2h window).
    Positive = rising, negative = falling.
    Returns 0.0 if not enough points.
    """
    if len(ema_series) < bars:
        return 0.0
    latest = ema_series[-1]
    older = ema_series[-bars]
    if older == 0:
        return 0.0
    pct_change = (latest - older) / older * 100.0
    return pct_change / hours


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Wilder's RSI (14-period standard).
    Uses Wilder's smoothing: prev_avg * (period-1)/period + current/period.
    Returns float in [0.0, 100.0], or None if len(closes) <= period.
    """
    if len(closes) <= period:
        return None
    ag = al = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        ag += max(d, 0.0)
        al += max(-d, 0.0)
    ag /= period
    al /= period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0.0)) / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
    if al == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def range_position(close: float, high_24h: float, low_24h: float) -> float:
    """
    Where close sits in the 24h range: 0.0 (at low) to 1.0 (at high).
    Returns 0.5 if range is zero (flat price).
    Clamped to [0.0, 1.0].
    """
    if high_24h <= low_24h:
        return 0.5
    pos = (close - low_24h) / (high_24h - low_24h)
    return max(0.0, min(1.0, pos))


def volume_ratio(last_vol: float, avg_vol_20: float) -> float:
    """
    Ratio of last bar's volume to the 20-bar rolling average.
    > 1.0 = above average, < 1.0 = below average.
    Returns 1.0 if avg_vol is 0.
    """
    if avg_vol_20 <= 0:
        return 1.0
    return last_vol / avg_vol_20


def higher_highs_fraction(highs: List[float], window: int = 24) -> float:
    """
    Fraction of the last `window` bars that made a higher high than the bar before.
    1.0 = every bar was a higher high (strong uptrend).
    0.0 = no higher highs (downtrend or flat).
    """
    if len(highs) < 2:
        return 0.0
    w = min(window, len(highs) - 1)
    start = len(highs) - w
    count = sum(1 for i in range(start, len(highs)) if highs[i] > highs[i - 1])
    return count / w


# ---------------------------------------------------------------------------
# Self-check with known-value tests (run directly: python3 indicators.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== indicators.py self-check ===\n")

    # 1. EMA on flat series stays flat
    flat = [100.0] * 20
    e9 = ema(flat, 9)
    assert len(e9) == 12  # 20 - 9 + 1
    assert abs(e9[-1] - 100.0) < 1e-9, f"EMA on flat failed: {e9[-1]}"
    print("[PASS] EMA flat series returns constant value")

    # 2. EMA on rising series tracks upwards
    rising = list(range(1, 31))  # 1..30
    e_rising = ema(rising, 9)
    assert e_rising[-1] > e_rising[0], "EMA rising series failed"
    # Hand-checked: SMA(1..9) = 5.0, subsequent values pull towards 30
    assert e_rising[0] == 5.0
    assert e_rising[-1] < 30.0  # EMA lags the current price
    print(f"[PASS] EMA rising series: starts at SMA 5.0, ends at {e_rising[-1]:.2f} (lags current 30)")

    # 3. EMA slope: positive on rising series, negative on falling
    slope_pos = ema_slope(e_rising, bars=9, hours=2.0)
    assert slope_pos > 0, f"slope should be positive: {slope_pos}"
    e_falling = list(reversed(e_rising))
    slope_neg = ema_slope(e_falling, bars=9, hours=2.0)
    assert slope_neg < 0, f"slope should be negative: {slope_neg}"
    print(f"[PASS] EMA slope sign: positive on rise ({slope_pos:+.2f}%/h), negative on fall ({slope_neg:+.2f}%/h)")

    # 4. RSI edge cases
    # Monotonically rising: gains only, losses = 0 -> RSI = 100.0
    assert rsi(list(range(1, 20)), 14) == 100.0
    print("[PASS] RSI monotonically rising = 100.0")

    # Monotonically falling: losses only, gains = 0 -> RSI near 0
    rsi_down = rsi(list(range(20, 0, -1)), 14)
    assert rsi_down is not None and rsi_down < 1.0, f"RSI falling failed: {rsi_down}"
    print(f"[PASS] RSI monotonically falling near 0: {rsi_down:.2f}")

    # Flat series: no gains, no losses -> division by zero handled cleanly
    rsi_flat = rsi([50.0] * 20, 14)
    assert rsi_flat == 100.0  # ag=0, al=0 -> al==0 branch returns 100.0
    print("[PASS] RSI flat series handled without crash")

    # Not enough bars -> returns None
    assert rsi([1.0, 2.0, 3.0], 14) is None
    print("[PASS] RSI insufficient bars returns None")

    # 5. Range position
    assert range_position(100.0, 100.0, 0.0) == 1.0   # at high
    assert range_position(0.0, 100.0, 0.0) == 0.0     # at low
    assert range_position(50.0, 100.0, 0.0) == 0.5    # midpoint
    assert range_position(150.0, 100.0, 0.0) == 1.0   # clamped above
    assert range_position(-10.0, 100.0, 0.0) == 0.0   # clamped below
    assert range_position(50.0, 50.0, 50.0) == 0.5    # zero-range fallback
    print("[PASS] range position: high=1.0, low=0.0, mid=0.5, clamping verified")

    # 6. Volume ratio
    assert volume_ratio(150.0, 100.0) == 1.5
    assert volume_ratio(50.0, 100.0) == 0.5
    assert volume_ratio(100.0, 0.0) == 1.0  # zero avg fallback
    print("[PASS] volume ratio: 1.5x, 0.5x, zero-avg fallback")

    # 7. Higher highs fraction
    h_up = list(range(1, 30))  # every bar higher
    assert higher_highs_fraction(h_up, 24) == 1.0
    h_down = list(range(30, 1, -1))  # no bar higher
    assert higher_highs_fraction(h_down, 24) == 0.0
    h_chop = [1, 2, 1, 2, 1, 2, 1, 2] * 4
    frac_chop = higher_highs_fraction(h_chop, 24)
    assert 0.4 <= frac_chop <= 0.6, f"chop fraction unexpected: {frac_chop}"
    print(f"[PASS] higher highs fraction: all-up=1.0, all-down=0.0, chop={frac_chop:.2f}")

    print("\n=== ALL CHECKS PASSED ===")
