#!/usr/bin/env python3
"""
framework_b.py — "Dip Hunter" (EXPERIMENTAL).

A deliberately DIFFERENT strategy, built to prove the framework interface
actually supports more than one shape:

                 framework A                framework B
  direction      trend UP (EMA9 > EMA21)    oversold bounce (RSI < 35 turning up)
  entry driver   momentum quality score     mean reversion
  target         +2.0%                      +1.5%
  stop           -4.0%                      -2.5%
  ratchet        yes (+1.2% -> +0.3%)       no
  score gate     >= 60                      >= 65

Why ship a second strategy nobody asked to trade? Because an interface with
exactly one implementation is untested by construction — you cannot know the
abstraction is right until something else fits through it. B is intentionally
simple and marked experimental; it is not the default and must be activated
explicitly while the engine is STOPPED.
"""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import rsi, ema, volume_ratio              # noqa: E402
from exit_strategy import (                                # noqa: E402
    EXIT_TARGET, EXIT_HARD_SL, tick_snap,
)
from frameworks.base import Framework                      # noqa: E402
import config as _config

# --- parameters -----------------------------------------------------------
TARGET_PCT = 0.015       # +1.5% limit sell
STOP_PCT = 0.025         # -2.5% market-sell trigger
RSI_OVERSOLD = 35.0      # entry requires RSI below this
RSI_FULL_DEPTH = 20.0    # RSI at/below this scores full oversold credit
SCORE_THRESHOLD = 60.0
MIN_CANDLES = 30

# Effective entry band: with full volume credit (30 pts) the RSI component
# must supply 30+ pts, which needs RSI below ~27. So in practice framework B
# buys RSI roughly 20-27 on above-average volume — a genuine dip, not a
# shallow pullback. Calibrated 2026-11-03; see the self-check below.


class FrameworkB(Framework):
    name = "B"
    label = "Dip Hunter"
    description = (
        "RSI mean-reversion: buys oversold (RSI < 35) pairs that are turning "
        "up through EMA9, on above-average volume. +1.5% limit target, "
        "-2.5% stop, no ratchet."
    )
    version = "0.1"
    experimental = True

    # -- 1. entry ---------------------------------------------------------
    def evaluate_entry(self, candles: list, depth: dict) -> dict:
        default = {"eligible": False, "total": 0.0, "direction": "SIDEWAYS",
                   "reasons": ["NO DATA"], "meta": {}}
        if not candles or len(candles) < MIN_CANDLES:
            return default

        closes = [float(c["c"]) for c in candles]
        volumes = [float(c["v"]) for c in candles]

        rsi_val = rsi(closes, 14)
        if rsi_val is None:
            return default

        ema9_series = ema(closes, 9)
        ema9_now = ema9_series[-1]
        prev_close = closes[-2]
        prev_ema9 = ema9_series[-2]
        last_close = closes[-1]

        avg_vol_20 = sum(volumes[-20:]) / 20.0
        v_ratio = volume_ratio(volumes[-1], avg_vol_20)

        # "Turning up": the gap to EMA9 narrowed this bar (first sign of a bounce)
        gap_now = last_close - ema9_now
        gap_prev = prev_close - prev_ema9
        turning_up = gap_now > gap_prev

        reasons = []
        if rsi_val >= RSI_OVERSOLD:
            reasons.append(f"RSI {rsi_val:.1f} >= {RSI_OVERSOLD:.0f}")
        if not turning_up:
            reasons.append("NOT TURNING UP")

        # Score: deeper oversold scores higher, volume confirms the turn.
        # RSI_FULL_DEPTH..RSI_OVERSOLD maps 0..70 pts; <= RSI_FULL_DEPTH gets 70.
        rsi_depth = (RSI_OVERSOLD - rsi_val) / (RSI_OVERSOLD - RSI_FULL_DEPTH)
        rsi_component = max(0.0, min(rsi_depth, 1.0)) * 70.0
        vol_component = min(v_ratio / 1.5, 1.0) * 30.0
        total = round(rsi_component + vol_component, 2)

        if total < SCORE_THRESHOLD:
            reasons.append(f"SCORE {total:.1f} < {SCORE_THRESHOLD:.0f}")

        eligible = (rsi_val < RSI_OVERSOLD) and turning_up and (total >= SCORE_THRESHOLD)

        return {
            "eligible": eligible,
            "total": total,
            "direction": "DOWN",          # the sign that triggers a buy
            "reasons": [] if eligible else reasons,
            "meta": {
                "rsi": round(rsi_val, 2),
                "vol_ratio": round(v_ratio, 4),
                "turning_up": turning_up,
                "gap_now": round(gap_now, 4),
                "gap_prev": round(gap_prev, 4),
            },
        }

    # -- 2. exit levels ---------------------------------------------------
    def initialize_exit(self, entry_price: float, tick: float) -> dict:
        # Parameters come from config.yaml (frameworks.B.*), falling back to this
        # module's constants when no config file is present. Read once per entry
        # so a live position keeps the levels it was opened with.
        p = _config.framework_params(_config.current(), self.name)
        target_pct = p["target_pct"]
        stop_pct = p["stop_pct"]

        # Decimal, not float: 1000 * 1.015 evaluates to 1014.9999999999999 in
        # binary floating point, which tick-snaps DOWN to 1014 and silently
        # shaves a tick off the target. exit_strategy.py does the same.
        d_entry = Decimal(str(entry_price))
        target_price = tick_snap(
            float(d_entry * (Decimal("1.0") + Decimal(str(target_pct)))), tick)
        stop_trigger_price = tick_snap(
            float(d_entry * (Decimal("1.0") - Decimal(str(stop_pct)))), tick)
        return {
            "strategy": "fixed_target",
            "framework": self.name,
            "framework_version": self.version,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
            "target_price": target_price,
            "stop_trigger_price": stop_trigger_price,
            # No ratchet for B — present and disabled so the engine's tick log
            # and dashboard can read the fields uniformly across frameworks.
            "ratchet_trigger_price": None,
            "ratchet_stop_price": None,
            "ratchet_armed": False,
            "limit_order_active": True,
        }

    # -- 3. exit evaluation -----------------------------------------------
    def evaluate_exit(self, live_price: float, levels: dict) -> dict:
        target = levels["target_price"]
        stop = levels["stop_trigger_price"]

        if live_price <= stop:
            return {"action": EXIT_HARD_SL,
                    "detail": f"live {live_price} <= stop {stop}"}
        if live_price >= target:
            return {"action": EXIT_TARGET,
                    "detail": f"live {live_price} >= target {target}"}
        return {"action": "HOLD",
                "detail": f"riding at {live_price} (target {target}, stop {stop})"}


# ---------------------------------------------------------------------------
# Self-check (run: python3 scripts/frameworks/framework_b.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== framework_b.py self-check ===\n")

    fw = FrameworkB()

    def candle(t, o, h, l, c, v):
        return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}

    # 1. Steady decline -> deeply oversold, but still falling: NOT eligible
    falling = []
    for i in range(40):
        c = 200.0 - i * 2.0
        falling.append(candle(i * 900, c + 0.5, c + 0.6, c - 0.5, c, 1000.0))
    res_fall = fw.evaluate_entry(falling, {})
    assert res_fall["eligible"] is False, res_fall
    assert any("TURNING UP" in r for r in res_fall["reasons"]), res_fall["reasons"]
    print(f"[PASS] falling-knife rejected: RSI={res_fall['meta'].get('rsi')}, "
          f"reasons={res_fall['reasons']}")

    # 2. Oversold then a modest bounce on volume -> eligible
    #    A realistic dip: steady slide, then TWO small up bars (not a moonshot).
    bounce = []
    for i in range(34):
        c = 200.0 - i * 1.0          # shallow but persistent decline -> low RSI
        bounce.append(candle(i * 900, c + 0.05, c + 0.1, c - 0.1, c, 1000.0))
    for j in range(2):               # two modest recovery bars
        c = bounce[-1]["c"] + (j + 1) * 0.8
        bounce.append(candle((34 + j) * 900, c - 0.05, c + 0.1, c - 0.1, c, 3000.0))
    res_bounce = fw.evaluate_entry(bounce, {})
    assert res_bounce["eligible"] is True, res_bounce
    assert res_bounce["total"] >= SCORE_THRESHOLD, res_bounce
    print(f"[PASS] oversold bounce eligible: total={res_bounce['total']}, "
          f"RSI={res_bounce['meta']['rsi']}, vol_ratio={res_bounce['meta']['vol_ratio']}")

    # 3. Strong uptrend (high RSI) -> rejected, it is not a dip
    rising = []
    for i in range(40):
        c = 100.0 + i * 2.0
        rising.append(candle(i * 900, c - 0.5, c + 0.6, c - 0.5, c, 1000.0))
    res_rise = fw.evaluate_entry(rising, {})
    assert res_rise["eligible"] is False, res_rise
    print(f"[PASS] uptrend rejected (not a dip): RSI={res_rise['meta'].get('rsi')}")

    # 4. Short history -> safe default
    res_short = fw.evaluate_entry(falling[:10], {})
    assert res_short["eligible"] is False and res_short["reasons"] == ["NO DATA"]
    print("[PASS] short history handled safely")

    # 5. Exit levels: +1.5% / -2.5%, no ratchet
    lv = fw.initialize_exit(1000.0, 1.0)
    assert lv["target_price"] == 1015.0, lv["target_price"]
    assert lv["stop_trigger_price"] == 975.0, lv["stop_trigger_price"]
    assert lv["ratchet_trigger_price"] is None
    assert lv["ratchet_armed"] is False
    assert lv["framework"] == "B"
    print("[PASS] levels: entry 1000 -> target 1015 (+1.5%), stop 975 (-2.5%), "
          "no ratchet")

    # 6. Exit decisions
    lv2 = fw.initialize_exit(1000.0, 1.0)
    assert fw.evaluate_exit(1000.0, lv2)["action"] == "HOLD"
    assert fw.evaluate_exit(1014.0, lv2)["action"] == "HOLD"
    assert fw.evaluate_exit(1015.0, lv2)["action"] == EXIT_TARGET
    assert fw.evaluate_exit(974.0, lv2)["action"] == EXIT_HARD_SL
    print("[PASS] decisions: 1000/1014 HOLD, 1015 TARGET, 974 HARD_SL")

    # 7. describe() flags it experimental
    d = fw.describe()
    assert d["experimental"] is True and d["name"] == "B"
    print("[PASS] describe() marks framework B experimental")

    print("\n=== ALL CHECKS PASSED ===")
