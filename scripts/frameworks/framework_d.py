#!/usr/bin/env python3
"""
framework_d.py — "Benchmark" (EXPERIMENTAL).

Trades exactly ONE pair: the #1 pair by 24h IDR volume at the moment the tick
runs. Nothing else is even looked at.

WHY THIS EXISTS

It is a LIVE-MODE TEST RIG, not a strategy. The idea is to put the smallest
possible surface in front of the live order path: one pair, one order, one exit.
If live trading misbehaves, there is one symbol in the logs and one position to
reconcile, instead of a rotating cast of twenty.

It deliberately removes almost every decision:

    pair        #1 by 24h IDR volume, whatever that is right now
    entry       NONE - it buys the pair regardless of trend, score or momentum
    target      +2.0%
    stop        -4.0%
    ratchet     none

WHAT THAT MEANS, SAID PLAINLY

There is no signal here. It will buy into a downtrend, into an overbought
spike, and into consolidation, because it does not look. That is the point -
it is testing the PLUMBING (does a real order get placed, does the fill get
read from the exchange, does the stop actually sell) and not the edge. Any
P&L it produces is meaningless as evidence about the strategy.

The gates still apply: spread, depth, freshness, candle count and cooldown all
run as normal. Buyers need a tradeable book even when the signal is ignored,
and the sizing guard still refuses a trade that would be below the exchange
minimum.

RANK IS RECOMPUTED EVERY TICK

"Top 1" is not pinned at activation. If BTC drops out of first place while a
position is open, the open position keeps running under the rules it was
entered with (positions are pinned), while new entries follow the new #1. This
is the same pinning rule every framework follows.
"""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exit_strategy import (                                # noqa: E402
    EXIT_HARD_SL, EXIT_TARGET, tick_snap, evaluate_exit as _evaluate_exit,
)
from frameworks.base import Framework                      # noqa: E402
import config as _config                                    # noqa: E402

# --- parameters -----------------------------------------------------------
TOP_N = 1                # how many of the ranked pairs to allow
TARGET_PCT = 0.02        # +2.0% limit sell
STOP_PCT = 0.04          # -4.0% market-sell trigger


class FrameworkD(Framework):
    name = "D"
    label = "Benchmark (top pair)"
    description = (
        "Trades ONLY the #1 pair by 24h IDR volume. No entry signal at all - "
        "it buys whatever is on top. +2%/-4%, no ratchet. A live-mode test rig "
        "for the order path, not a strategy: its P&L says nothing about edge."
    )
    version = "0.1"
    experimental = True

    # -- 1a. pair selection ------------------------------------------------
    def allowed_symbols(self, ranked_symbols: list) -> list:
        """Only the first N ranked pairs are ever fetched or evaluated."""
        ranked = list(ranked_symbols or [])
        return ranked[:TOP_N]

    # -- 1. entry: no signal ------------------------------------------------
    def evaluate_entry(self, candles: list, depth: dict) -> dict:
        """Always eligible.

        There is no scoring here on purpose. Adding a filter would make this
        "the top pair when it looks good", which is a strategy with an untested
        edge - and then a live-plumbing test would silently become an untested
        live strategy.
        """
        return {
            "eligible": True,
            "total": 100.0,
            "direction": "N/A",
            "reasons": [],
            "meta": {"note": "no entry signal - benchmark rig"},
        }

    # -- 2. exit levels ----------------------------------------------------
    def initialize_exit(self, entry_price: float, tick: float) -> dict:
        p = _config.framework_params(_config.current(), self.name)
        target_pct = p.get("target_pct", TARGET_PCT)
        stop_pct = p.get("stop_pct", STOP_PCT)

        # Decimal for the same reason as A and C: 1000 * 1.04 is not exact in
        # binary, and tick_snap DOWN would shave a tick off the stop.
        d_entry = Decimal(str(entry_price))
        target_price = tick_snap(
            float(d_entry * (Decimal("1.0") + Decimal(str(target_pct)))), tick)
        stop_trigger_price = tick_snap(
            float(d_entry * (Decimal("1.0") - Decimal(str(stop_pct)))), tick)

        return {
            "strategy": "fixed_target",
            "framework": self.name,
            "framework_version": self.version,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_trigger_price": stop_trigger_price,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
            "ratchet_trigger_pct": 0.0,
            "ratchet_stop_pct": 0.0,
            "ratchet_armed": False,
        }

    # -- 3. exit evaluation ------------------------------------------------
    def evaluate_exit(self, live_price: float, levels: dict) -> dict:
        return _evaluate_exit(live_price, levels)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== framework_d.py self-check ===\n")

    fw = FrameworkD()
    ranked = ["BTCIDR", "PEPEIDR", "XRPIDR", "ETHIDR", "UAIIDR"]

    # 1. only the top pair is allowed
    allowed = fw.allowed_symbols(ranked)
    assert allowed == ["BTCIDR"], allowed
    print(f"[PASS] only the #1 pair is allowed: {allowed}")

    # 2. an empty or single-pair universe does not crash
    assert fw.allowed_symbols([]) == []
    assert fw.allowed_symbols(["SOLIDR"]) == ["SOLIDR"]
    assert fw.allowed_symbols(None) == []
    print("[PASS] empty/None/short universes are handled")

    # 3. rank order is preserved, not re-sorted
    assert fw.allowed_symbols(["ZZZIDR", "AAAIDR"]) == ["ZZZIDR"]
    print("[PASS] the caller's rank order is respected (not re-sorted)")

    # 4. entry never rejects - that is the point of the rig
    r = fw.evaluate_entry([], {})
    assert r["eligible"] is True and r["reasons"] == [], r
    assert r["direction"] == "N/A"
    print("[PASS] entry is unconditional by design (no signal)")

    # 5. exit levels
    lv = fw.initialize_exit(1000.0, 1.0)
    assert lv["target_price"] == 1020.0, lv["target_price"]
    assert lv["stop_trigger_price"] == 960.0, lv["stop_trigger_price"]
    assert lv["framework"] == "D"
    assert lv["ratchet_armed"] is False
    print(f"[PASS] levels: target {lv['target_price']:.0f} (+2%), "
          f"stop {lv['stop_trigger_price']:.0f} (-4%)")

    # 6. tick snapping still applies
    lv2 = fw.initialize_exit(641.0, 1.0)
    assert lv2["target_price"] == 653.0 and lv2["stop_trigger_price"] == 615.0, lv2
    print("[PASS] levels snap to the pair's tick")

    # 7. exits fire at the levels, hold between
    assert fw.evaluate_exit(1020.0, dict(lv))["action"] == EXIT_TARGET
    assert fw.evaluate_exit(960.0, dict(lv))["action"] == EXIT_HARD_SL
    assert fw.evaluate_exit(1000.0, dict(lv))["action"] == "HOLD"
    print("[PASS] exits fire at target/stop and hold between them")

    # 8. no ratchet ever arms
    lvl = dict(lv)
    for price in (1000, 1012, 1019, 1030):
        fw.evaluate_exit(float(price), lvl)
    assert lvl.get("ratchet_armed") is False, lvl
    print("[PASS] the ratchet never arms")

    print("\n=== ALL SELF-CHECKS PASSED ===")
