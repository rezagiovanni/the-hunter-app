#!/usr/bin/env python3
"""
exit_strategy.py — The Hunter's fixed-target exit rule.

Replaces the trailing-lock ladder (archived at scripts/legacy/ladder_trailing_lock.py,
kept for possible future use).

Rule (locked with Reza, 2026-10-21; levels revised same day):
  ENTRY       -> place a resting LIMIT SELL at entry x (1 + TARGET_PCT)
  TARGET HIT  -> the resting limit fills at target_price (MAKER fee)
  STOP CROSSED-> cancel the resting limit, MARKET SELL at best bid (TAKER fee)

  TARGET_PCT = +2%      (gross price move; net is lower after fees)
  STOP_PCT   = -4%      (trigger only — the fill is lower: see STOP_SLIP_NOTE)

No rungs, no ratchet. Two prices decide the whole trade.

Why the stop is a trigger and not a fill:
  A market sell has no price control. The measured behaviour of the old engine was
  that hard stops filled ~0.94% BELOW their -3% trigger. This module keeps the
  trigger at -3% (so the decision point is unchanged) and does not pretend the
  fill is at -3%.
"""

import os
import sys
from decimal import Decimal
from typing import Optional

# ---------------------------------------------------------------------------
# Locked parameters
# ---------------------------------------------------------------------------
TARGET_PCT = 0.02            # +2% limit sell, gross
STOP_PCT = 0.04              # -4% market-sell trigger
RATCHET_TRIGGER_PCT = 0.012  # +1.2% price move arms the breakeven ratchet
RATCHET_STOP_PCT = -0.003    # +0.3% stop floor (covering round-trip fees)

EXIT_TARGET = "EXIT_TARGET"          # resting limit filled at +TARGET_PCT (maker fee)
EXIT_HARD_SL = "EXIT_HARD_SL"        # -STOP_PCT crossed -> market sell    (taker fee)
EXIT_BE_RATCHET = "EXIT_BE_RATCHET"  # breakeven ratchet triggered after +1.2% (protects profit)
EXIT_USER_STOP = "EXIT_USER_STOP"    # user pressed STOP -> immediate market sell (taker fee)


# ---------------------------------------------------------------------------
# Tick snapping (integer discipline — preserved from the ladder era)
# ---------------------------------------------------------------------------
def tick_snap(price: float, tick: float) -> float:
    """
    Snap a price DOWN to the pair's minimum tick size.
    Decimal-based to avoid floating-point artifacts.
    """
    if tick <= 0:
        return price
    d_price = Decimal(str(price))
    d_tick = Decimal(str(tick))
    snapped = (d_price // d_tick) * d_tick
    if tick >= 1.0:
        return float(int(snapped))
    return float(snapped)


# ---------------------------------------------------------------------------
# Level computation
# ---------------------------------------------------------------------------
def initialize_exit_levels(entry_price: float, tick: float,
                           target_pct: float = TARGET_PCT,
                           stop_pct: float = STOP_PCT,
                           ratchet_trigger_pct: float = RATCHET_TRIGGER_PCT,
                           ratchet_stop_pct: float = RATCHET_STOP_PCT) -> dict:
    """
    Compute exit levels at entry time and return the position exit state.

    target_price          : resting LIMIT SELL level  (snapped DOWN = fills safely)
    stop_trigger_price    : initial MARKET SELL trigger (snapped DOWN = exits sooner)
    ratchet_trigger_price : price level (+1.2%) that arms the breakeven ratchet
    ratchet_stop_price    : stop floor (+0.3%) once ratchet is armed (covers fees)
    """
    d_entry = Decimal(str(entry_price))
    target_price = tick_snap(float(d_entry * (Decimal("1.0") + Decimal(str(target_pct)))), tick)
    stop_trigger_price = tick_snap(float(d_entry * (Decimal("1.0") - Decimal(str(stop_pct)))), tick)
    ratchet_trigger_price = tick_snap(float(d_entry * (Decimal("1.0") + Decimal(str(ratchet_trigger_pct)))), tick)
    ratchet_stop_price = tick_snap(float(d_entry * (Decimal("1.0") - Decimal(str(ratchet_stop_pct)))), tick)

    return {
        "strategy": "fixed_target_ratchet",
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "target_price": target_price,
        "stop_trigger_price": stop_trigger_price,
        "ratchet_trigger_price": ratchet_trigger_price,
        "ratchet_stop_price": ratchet_stop_price,
        "ratchet_armed": False,
        "limit_order_active": True,           # the resting limit sell is live
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_exit(live_price: float, levels: dict) -> dict:
    """
    Decide whether the position exits on this tick.

    Priority:
      1. Arm ratchet: if live_price >= ratchet_trigger_price, raise stop trigger to ratchet_stop_price.
      2. STOP  — live_price <= stop_trigger_price  -> EXIT_BE_RATCHET or EXIT_HARD_SL
      3. TARGET— live_price >= target_price        -> EXIT_TARGET   (limit fills)

    Returns {"action": "HOLD" | EXIT_TARGET | EXIT_HARD_SL | EXIT_BE_RATCHET, "detail": str}
    """
    # 1. Arm ratchet if price touched +1.2%
    if not levels.get("ratchet_armed", False):
        r_trig = levels.get("ratchet_trigger_price")
        if r_trig is not None and live_price >= r_trig:
            levels["ratchet_armed"] = True
            r_stop = levels.get("ratchet_stop_price", levels["stop_trigger_price"])
            levels["stop_trigger_price"] = max(levels["stop_trigger_price"], r_stop)

    target = levels["target_price"]
    stop = levels["stop_trigger_price"]

    if live_price <= stop:
        action = EXIT_BE_RATCHET if levels.get("ratchet_armed", False) else EXIT_HARD_SL
        return {
            "action": action,
            "detail": f"live {live_price} <= stop {stop} (ratchet_armed={levels.get('ratchet_armed', False)})",
        }

    if live_price >= target:
        return {
            "action": EXIT_TARGET,
            "detail": f"live {live_price} >= target {target} (+{levels['target_pct']*100:.0f}%)",
        }

    return {
        "action": "HOLD",
        "detail": f"riding at {live_price} (target {target}, stop {stop}, ratchet={levels.get('ratchet_armed', False)})",
    }


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 exit_strategy.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== exit_strategy.py self-check ===\n")

    # 1. Levels on a clean integer-tick pair
    lv = initialize_exit_levels(1000.0, 1.0)
    assert lv["target_price"] == 1020.0, lv["target_price"]
    assert lv["stop_trigger_price"] == 960.0, lv["stop_trigger_price"]
    print(f"[PASS] entry 1000 tick 1 -> target {lv['target_price']} (+2%), stop {lv['stop_trigger_price']} (-4%)")

    # 2. Snapping down: non-integer results floor toward safety
    lv2 = initialize_exit_levels(641.0, 1.0)          # 641*1.02=653.82, 641*0.96=615.36
    assert lv2["target_price"] == 653.0, lv2["target_price"]
    assert lv2["stop_trigger_price"] == 615.0, lv2["stop_trigger_price"]
    print(f"[PASS] entry 641 tick 1 -> target {lv2['target_price']} (653.82 floored), stop {lv2['stop_trigger_price']} (615.36 floored)")

    # 3. BTC 1000-tick pair stays on tick
    lv3 = initialize_exit_levels(1_585_250.0, 1000.0)
    assert lv3["target_price"] % 1000 == 0
    assert lv3["stop_trigger_price"] % 1000 == 0
    print(f"[PASS] BTC tick 1000 -> target {lv3['target_price']}, stop {lv3['stop_trigger_price']} (both on tick)")

    # 4. Decision logic (levels from entry 1000: target 1020, stop 960)
    lv4_target = initialize_exit_levels(1000.0, 1.0)
    assert evaluate_exit(1000.0, lv4_target)["action"] == "HOLD"
    assert evaluate_exit(1011.0, lv4_target)["action"] == "HOLD"
    assert evaluate_exit(1020.0, lv4_target)["action"] == EXIT_TARGET
    assert evaluate_exit(1100.0, lv4_target)["action"] == EXIT_TARGET

    lv4_sl = initialize_exit_levels(1000.0, 1.0)
    assert evaluate_exit(959.0, lv4_sl)["action"] == EXIT_HARD_SL
    assert evaluate_exit(900.0, lv4_sl)["action"] == EXIT_HARD_SL
    print("[PASS] decisions: 1000/1011 HOLD, 1020/1100 TARGET, 959/900 STOP")

    # 5. Stop wins a same-tick double breach (conservative)
    res = evaluate_exit(1.0, lv4_sl)
    assert res["action"] == EXIT_HARD_SL
    print("[PASS] pathological tick below stop -> STOP (conservative priority)")

    # 6. Round trip: entry -> +2% -> target hit exactly once
    lv6 = initialize_exit_levels(2000.0, 1.0)
    path = [2000, 2010, 2035, 2039, 2040, 2040]
    actions = [evaluate_exit(p, lv6)["action"] for p in path]
    assert actions == ["HOLD", "HOLD", "HOLD", "HOLD", EXIT_TARGET, EXIT_TARGET]
    print("[PASS] price ramp to +2% -> fires EXIT_TARGET at exactly target")

    # 7. The locked live levels (regression guard on the configured numbers)
    assert TARGET_PCT == 0.02 and STOP_PCT == 0.04
    assert RATCHET_TRIGGER_PCT == 0.012 and RATCHET_STOP_PCT == -0.003
    print(f"[PASS] configured levels: TARGET_PCT {TARGET_PCT} (+2%), STOP_PCT {STOP_PCT} (-4%), RATCHET {RATCHET_TRIGGER_PCT} -> +0.3%")

    # 8. Breakeven Ratchet sequence: entry 1000 -> climbs to 1012 (+1.2%) -> drops to 1002 -> EXIT_BE_RATCHET
    lv8 = initialize_exit_levels(1000.0, 1.0)
    assert not lv8["ratchet_armed"]
    assert lv8["stop_trigger_price"] == 960.0
    assert evaluate_exit(1010.0, lv8)["action"] == "HOLD"
    assert not lv8["ratchet_armed"]
    # Touch +1.2%
    assert evaluate_exit(1012.0, lv8)["action"] == "HOLD"
    assert lv8["ratchet_armed"], "Ratchet must be armed at +1.2%"
    assert lv8["stop_trigger_price"] == 1003.0, f"Stop must be raised to 1003, got {lv8['stop_trigger_price']}"
    # Price falls back to 1005 (still above 1003) -> HOLD
    assert evaluate_exit(1005.0, lv8)["action"] == "HOLD"
    # Price touches 1003 -> EXIT_BE_RATCHET
    res8 = evaluate_exit(1003.0, lv8)
    assert res8["action"] == EXIT_BE_RATCHET, f"Expected EXIT_BE_RATCHET, got {res8['action']}"
    print(f"[PASS] ratchet sequence: 1000 -> 1012 (armed) -> 1003 fires {res8['action']}")

    print("\n=== ALL CHECKS PASSED ===")
