#!/usr/bin/env python3
"""
framework_c.py — "Wide Stop" (EXPERIMENTAL).

A third exit geometry on the SAME entry signal as framework A. It exists to
answer one question with data instead of opinion: does a wider stop and a
larger target do better than A's +2%/-4%?

                A (Trend Rider)      C (Wide Stop)
  entry signal  trend UP + score     identical to A
  target        +2.0%                +5.0%
  stop          -4.0%                -10.0%
  ratchet       yes (+1.2%->+0.3%)   no
  payoff        0.5 : 1              0.5 : 1

WHY THE PAYOFF IS UNCHANGED

5/10 is the same ratio as 2/4, so C is not "more aggressive" in risk-reward
terms - it is the SAME bet with the levels stretched. What changes is the noise
tolerance: a -10% stop is far outside the 1-hour range of most pairs, so it
should survive shakeouts that A's -4% stop gets caught by.

WHY NO RATCHET

The point of a 10% stop is to give the trade room. Ratcheting the stop up to
breakeven at +1.2% would convert C back into A with worse targets. C removes the
ratchet deliberately.

EXIT-MEASUREMENT CONTEXT

On the live log (41 trades) only 2 reached +5% while holding, while 17 reached
it only AFTER the bot had exited. That is the hypothesis C tests: a wider stop
holds long enough for the +5% target to actually be reachable. It is a
hypothesis, not a claim - switch to C only while the engine is STOPPED, and
compare cohorts after enough trades.
"""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exit_strategy import (                                # noqa: E402
    EXIT_HARD_SL, EXIT_TARGET, tick_snap, evaluate_exit as _evaluate_exit,
)
from frameworks.base import Framework                      # noqa: E402
from filters import evaluate_filters                        # noqa: E402
import config as _config                                    # noqa: E402

# --- parameters -----------------------------------------------------------
TARGET_PCT = 0.05        # +5.0% limit sell
STOP_PCT = 0.10          # -10.0% market-sell trigger
SCORE_THRESHOLD = 60.0   # the same entry bar as A, so the comparison isolates exits


class FrameworkC(Framework):
    name = "C"
    label = "Wide Stop"
    description = (
        "A's entry signal with the exits stretched: +5% target, -10% stop, no "
        "ratchet. Same 0.5:1 payoff as A but far more noise tolerance, to test "
        "whether a wider stop lets the +5% target actually be reached."
    )
    version = "0.1"
    experimental = True

    # -- 1. entry: identical to A, on purpose -----------------------------
    def evaluate_entry(self, candles: list, depth: dict) -> dict:
        """Same entry as framework A.

        Deliberately NOT a new signal: if C also changed the entry, a difference
        in results could not be attributed to the exits, which are the thing
        being tested.
        """
        r = evaluate_filters(candles)

        reasons = []
        if r.get("direction") != "UP":
            reasons.append("NOT UP")
        if r.get("overextended"):
            reasons.append("EXTENDED")
        if (r.get("total") or 0.0) < SCORE_THRESHOLD:
            reasons.append(f"SCORE < {SCORE_THRESHOLD:.0f}")

        return {
            "eligible": bool(r.get("eligible")),
            "total": r.get("total"),
            "direction": r.get("direction"),
            "reasons": reasons,
            "meta": r,
        }

    # -- 2. exit levels ----------------------------------------------------
    def initialize_exit(self, entry_price: float, tick: float) -> dict:
        p = _config.framework_params(_config.current(), self.name)
        target_pct = p["target_pct"]
        stop_pct = p["stop_pct"]

        # Decimal, not float: 1000 * 1.05 is fine but 1000 * 1.1 is not exact in
        # binary, and tick_snap DOWN would shave a tick off the stop. The other
        # frameworks do the same for the same reason.
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
            # no ratchet: recorded as 0 so the engine's ratchet check is inert
            # and a trade record shows plainly that C never arms one
            "ratchet_trigger_pct": 0.0,
            "ratchet_stop_pct": 0.0,
            "ratchet_armed": False,
        }

    # -- 3. exit evaluation ------------------------------------------------
    def evaluate_exit(self, live_price: float, levels: dict) -> dict:
        return _evaluate_exit(live_price, levels)


# ---------------------------------------------------------------------------
# Self-check (run: python3 scripts/frameworks/framework_c.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fw = FrameworkC()
    print("=== framework_c self-check ===\n")

    # 1. levels are exactly +5% / -10%
    lv = fw.initialize_exit(1000.0, 1.0)
    assert lv["target_price"] == 1050.0, lv["target_price"]
    assert lv["stop_trigger_price"] == 900.0, lv["stop_trigger_price"]
    assert lv["target_pct"] == 0.05 and lv["stop_pct"] == 0.10
    print(f"[PASS] 1000 -> target {lv['target_price']:.0f} (+5%), "
          f"stop {lv['stop_trigger_price']:.0f} (-10%)")

    # 2. no ratchet
    assert lv["ratchet_trigger_pct"] == 0.0 and lv["ratchet_stop_pct"] == 0.0
    assert lv["ratchet_armed"] is False
    print("[PASS] ratchet disabled, and recorded as disabled")

    # 3. stamped so a trade record can be attributed to C
    assert lv["framework"] == "C" and lv["framework_version"] == "0.1"
    print("[PASS] levels stamped framework C v0.1")

    # 4. same payoff ratio as A
    ratio = lv["target_pct"] / lv["stop_pct"]
    assert abs(ratio - 0.5) < 1e-9, ratio
    print(f"[PASS] payoff {lv['target_pct']:.2f}/{lv['stop_pct']:.2f} = {ratio:.2f} "
          f"(same ratio as A)")

    # 5. tick snapping keeps the stop from rounding UP past its level
    lv2 = fw.initialize_exit(1000.0, 5.0)
    assert lv2["stop_trigger_price"] <= 900.0, lv2["stop_trigger_price"]
    assert lv2["target_price"] <= 1050.0
    print(f"[PASS] tick 5 snaps DOWN: stop {lv2['stop_trigger_price']:.0f} "
          f"<= 900, target {lv2['target_price']:.0f} <= 1050")

    # 6. the exit rules fire at the right prices
    # evaluate_exit returns an action for EVERY tick; "hold" is the action HOLD,
    # not None. Asserting None here was my mistake, not the code's.
    assert fw.evaluate_exit(1050.0, lv)["action"] == EXIT_TARGET
    assert fw.evaluate_exit(899.0, lv)["action"] == EXIT_HARD_SL
    assert fw.evaluate_exit(1000.0, lv)["action"] == "HOLD"
    assert fw.evaluate_exit(1049.0, lv)["action"] == "HOLD", "a hair below target holds"
    assert fw.evaluate_exit(901.0, lv)["action"] == "HOLD", "a hair above stop holds"
    print("[PASS] exits fire at 1050 / 900 and hold at 1000, 1049, 901")

    # 7. no candles -> not eligible, never raises
    r = fw.evaluate_entry([], {})
    assert r["eligible"] is False and r["reasons"]
    print("[PASS] empty candles -> ineligible with a reason")

    print("\n=== ALL CHECKS PASSED ===")
