#!/usr/bin/env python3
"""
framework_a.py — "Trend Rider": the strategy The Hunter has been running.

This is a THIN ADAPTER over the existing, already-measured modules:
    filters.py        — entry scoring + eligibility
    exit_strategy.py  — levels, ratchet, exit evaluation

It deliberately adds no logic of its own. The point is that framework A
behaves EXACTLY as the code did before the framework layer existed; the
golden test at the bottom of this file fails loudly if that stops being true.

If you want to change how A trades, change filters.py / exit_strategy.py.
Do not add strategy maths here — that is how two sources of truth appear.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import evaluate_filters                      # noqa: E402
from exit_strategy import (                               # noqa: E402
    EXIT_TARGET, EXIT_HARD_SL, EXIT_BE_RATCHET,
    evaluate_exit as _evaluate_exit,
    initialize_exit_levels as _initialize_exit_levels,
)
from frameworks.base import Framework                     # noqa: E402
import config as _config                                  # noqa: E402

SCORE_THRESHOLD = 60.0


class FrameworkA(Framework):
    name = "A"
    label = "Trend Rider"
    description = (
        "EMA9/21 trend + momentum quality. RSI 50-60 sweet spot, "
        "pullback guard (rejects >1.5% above EMA9), +2% limit target "
        "with -4% stop and a breakeven ratchet at +1.2%."
    )
    version = "3.0"
    experimental = False

    # -- 1. entry ---------------------------------------------------------
    def evaluate_entry(self, candles: list, depth: dict) -> dict:
        r = evaluate_filters(candles)

        reasons = []
        if r.get("direction") != "UP":
            reasons.append("NOT UP")
        if r.get("overextended"):
            reasons.append("EXTENDED")
        if (r.get("total") or 0.0) < SCORE_THRESHOLD:
            reasons.append("SCORE < 60")

        return {
            "eligible": bool(r.get("eligible")),
            "total": r.get("total"),
            "direction": r.get("direction"),
            "reasons": reasons,
            "meta": r,
        }

    # -- 2. exit levels ---------------------------------------------------
    def initialize_exit(self, entry_price: float, tick: float) -> dict:
        # Parameters come from config.yaml (frameworks.A.*), falling back to the
        # module constants when no config file is present. Read once per entry,
        # so a position keeps the levels it was opened with even if the file
        # changes while the position is live.
        p = _config.framework_params(_config.current(), self.name)
        levels = _initialize_exit_levels(
            entry_price, tick,
            target_pct=p["target_pct"],
            stop_pct=p["stop_pct"],
            ratchet_trigger_pct=p["ratchet_trigger_pct"],
            ratchet_stop_pct=p["ratchet_stop_pct"],
        )
        levels["framework"] = self.name
        levels["framework_version"] = self.version
        return levels

    # -- 3. exit evaluation -----------------------------------------------
    def evaluate_exit(self, live_price: float, levels: dict) -> dict:
        return _evaluate_exit(live_price, levels)


# ---------------------------------------------------------------------------
# Self-check / golden test (run: python3 scripts/frameworks/framework_a.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== framework_a.py golden test ===\n")

    fw = FrameworkA()

    # --- exit levels must match the pre-framework engine exactly ----------
    # These integers are copied from the live Hunt H-0001 (UAIIDR) record:
    #   entry 3083, tick 1 -> target 3144, stop 2959, ratchet trigger 3119, floor 3092
    lv = fw.initialize_exit(3083.0, 1.0)
    assert lv["target_price"] == 3144.0, lv["target_price"]
    assert lv["stop_trigger_price"] == 2959.0, lv["stop_trigger_price"]
    assert lv["ratchet_trigger_price"] == 3119.0, lv["ratchet_trigger_price"]
    assert lv["ratchet_stop_price"] == 3092.0, lv["ratchet_stop_price"]
    assert lv["target_pct"] == 0.02 and lv["stop_pct"] == 0.04
    assert lv["framework"] == "A", lv["framework"]
    print("[PASS] exit levels match live H-0001: target 3144, stop 2959, "
          "ratchet 3119 -> floor 3092")

    # --- the 1000/1.0 reference case (used throughout the engine tests) ---
    lv2 = fw.initialize_exit(1000.0, 1.0)
    assert lv2["target_price"] == 1020.0
    assert lv2["stop_trigger_price"] == 960.0
    assert lv2["ratchet_trigger_price"] == 1012.0
    assert lv2["ratchet_stop_price"] == 1003.0
    print("[PASS] reference case entry 1000: target 1020, stop 960, "
          "ratchet 1012 -> floor 1003")

    # --- exit decisions unchanged ----------------------------------------
    lv3 = fw.initialize_exit(1000.0, 1.0)
    assert fw.evaluate_exit(1000.0, lv3)["action"] == "HOLD"
    assert fw.evaluate_exit(1020.0, lv3)["action"] == EXIT_TARGET

    lv4 = fw.initialize_exit(1000.0, 1.0)
    assert fw.evaluate_exit(959.0, lv4)["action"] == EXIT_HARD_SL

    lv5 = fw.initialize_exit(1000.0, 1.0)
    assert fw.evaluate_exit(1012.0, lv5)["action"] == "HOLD"
    assert lv5["ratchet_armed"] is True
    assert lv5["stop_trigger_price"] == 1003.0
    assert fw.evaluate_exit(1003.0, lv5)["action"] == EXIT_BE_RATCHET
    print("[PASS] exit decisions: HOLD / TARGET / HARD_SL / BE_RATCHET all intact")

    # --- entry path: adapter returns the same verdict as filters.py -------
    def candle(t, o, h, l, c, v):
        return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}

    up = []
    for i in range(60):
        c = 100.0 + i * 0.5
        up.append(candle(i * 900, c - 0.2, c + 0.3, c - 0.4, c, 1000.0 + i * 10))

    direct = evaluate_filters(up)
    via_fw = fw.evaluate_entry(up, {})
    assert via_fw["eligible"] == direct["eligible"], (via_fw, direct)
    assert via_fw["total"] == direct["total"], (via_fw, direct)
    assert via_fw["direction"] == direct["direction"], (via_fw, direct)
    print(f"[PASS] entry adapter == filters.evaluate_filters "
          f"(eligible={via_fw['eligible']}, total={via_fw['total']})")

    # --- an overextended pump must still be rejected ----------------------
    pump = []
    for i in range(35):
        c = 100.0 + (i if i < 20 else 20.0 + (i - 20) ** 2.2)
        pump.append(candle(i * 900, c - 0.5, c + 0.5, c - 0.5, c, 1000.0))
    res_pump = fw.evaluate_entry(pump, {})
    assert res_pump["eligible"] is False, res_pump
    assert "EXTENDED" in res_pump["reasons"], res_pump["reasons"]
    print(f"[PASS] overextended pump rejected: reasons={res_pump['reasons']}")

    # --- describe() shape -------------------------------------------------
    d = fw.describe()
    for k in ("name", "label", "description", "version", "experimental"):
        assert k in d, k
    assert d["name"] == "A" and d["experimental"] is False
    print("[PASS] describe() exposes name/label/description/version/experimental")

    print("\n=== ALL CHECKS PASSED ===")
