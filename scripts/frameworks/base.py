#!/usr/bin/env python3
"""
base.py — the Framework interface.

A Framework bundles three things that used to be hardcoded in the engine:
  1. entry selection   (indicator scoring + eligibility)
  2. exit level setup  (initial target / stop / any ratchet levels)
  3. exit evaluation   (has the position hit target, stop, or a ratchet?)

The engine calls ONLY these three methods. It never imports filters.py or
exit_strategy.py directly, so a new strategy is a new file in this package,
not a refactor of run_hunter.py.

Contract notes
--------------
* initialize_exit() MUST include "framework": self.name in the returned dict.
  That field is what pins a position to the rules it was opened under, and
  what stamps trade records (so PnL from different frameworks is never mixed
  into one unlabelable pool).
* evaluate_exit() may mutate `levels` (the breakeven ratchet does exactly
  that). Callers must pass the live, persisted levels dict.
* Methods must never touch the network or the filesystem. Pure computation
  only — that is what makes them cheap to test and safe to run on every tick.
"""


class Framework:
    """Base class. Subclass and override the three methods."""

    #: short id used in the control file and trade records ("A", "B", ...)
    name = "?"

    #: human label for the dashboard
    label = "?"

    #: one-line description of what the framework does
    description = ""

    #: bump when the scoring or exit maths changes — stamped on trade records
    version = "0"

    #: True = shown with an EXPERIMENTAL badge; never the default
    experimental = False

    # -- 1. entry ---------------------------------------------------------
    def evaluate_entry(self, candles: list, depth: dict) -> dict:
        """
        Decide whether a pair is eligible right now.

        Returns::
            {
              "eligible": bool,
              "total":    float,   # 0-100 score
              "direction": "UP" | "DOWN" | "SIDEWAYS",
              "reasons":  list[str],   # why it was rejected (empty if eligible)
              "meta":     dict,        # framework-specific detail for the UI
            }
        """
        raise NotImplementedError

    # -- 2. exit levels ---------------------------------------------------
    def initialize_exit(self, entry_price: float, tick: float) -> dict:
        """
        Compute exit levels at fill time.

        Must include at minimum: target_price, stop_trigger_price,
        target_pct, stop_pct, and "framework": self.name.
        """
        raise NotImplementedError

    # -- 3. exit evaluation -----------------------------------------------
    def evaluate_exit(self, live_price: float, levels: dict) -> dict:
        """
        Decide whether the position exits on this tick.

        Returns {"action": "HOLD" | EXIT_TARGET | EXIT_HARD_SL | ...,
                 "detail": str}
        May mutate `levels` (e.g. arming a ratchet).
        """
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def describe(self) -> dict:
        """Metadata for GET /api/frameworks."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "version": self.version,
            "experimental": self.experimental,
        }

    def __repr__(self) -> str:
        return f"<Framework {self.name} v{self.version} '{self.label}'>"
