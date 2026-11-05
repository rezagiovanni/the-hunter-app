#!/usr/bin/env python3
"""
hunt_log.py — append-only recorder for The Hunter.

Two jobs:

1. HUNT TICKS (minute-by-minute while HOLDING)
   One JSONL row per tick per open hunt, capturing the best bid and best ask so
   a hunt's full price path can be replayed later. This is the data that answers
   "was the +5% target actually reachable before the -3% stop?"

   File: data/hunt_ticks/<HUNT_ID>.jsonl

2. HUNT EVENTS (one row per state transition)
   Entry filled, exit executed, order cancelled. Makes the tick log readable
   without joining state files.

   File: data/hunt_events.jsonl

Hunt IDs (locked with Reza, 2026-10-21):
   H-0001, H-0002, ... H-9999, H-10000, H-10001  (zero-padded to 4, then natural)
"""

import json
import os
import time
from typing import Optional

# Paths from paths.py - see that module for why this is centralised.
try:
    from paths import DATA_DIR, HUNT_TICKS_DIR as TICKS_DIR, HUNT_EVENTS_FILE as EVENTS_FILE
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import DATA_DIR, HUNT_TICKS_DIR as TICKS_DIR, HUNT_EVENTS_FILE as EVENTS_FILE


# ---------------------------------------------------------------------------
# Hunt ID
# ---------------------------------------------------------------------------
def format_hunt_id(n: int) -> str:
    """
    H-0001 ... H-9999, then H-10000, H-10001 (natural growth past 4 digits).
    """
    if n < 10000:
        return f"H-{n:04d}"
    return f"H-{n}"


def next_hunt_id(state: dict) -> str:
    """Allocate the next hunt id and advance the counter in state."""
    nxt = int(state.get("hunt_counter", 0)) + 1
    state["hunt_counter"] = nxt
    return format_hunt_id(nxt)


# ---------------------------------------------------------------------------
# Tick recording (minute-by-minute while holding)
# ---------------------------------------------------------------------------
def record_hunt_tick(hunt_id: str, payload: dict, now: Optional[float] = None) -> None:
    """
    Append one tick row to data/hunt_ticks/<hunt_id>.jsonl.

    Never raises — a logging failure must not break a tick.
    """
    if now is None:
        now = time.time()
    try:
        os.makedirs(TICKS_DIR, exist_ok=True)
        row = {"t": round(float(now), 3)}
        row.update(payload)
        with open(os.path.join(TICKS_DIR, f"{hunt_id}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Event recording (one row per transition)
# ---------------------------------------------------------------------------
def record_hunt_event(hunt_id: Optional[str], event: str, payload: dict,
                      now: Optional[float] = None) -> None:
    """
    Append a transition row to data/hunt_events.jsonl.
    Events: ENTRY_FILLED | EXIT_TARGET | EXIT_HARD_SL | EXIT_BE_RATCHET
            | EXIT_USER_STOP | ORDER_TIMEOUT | ORDER_CANCELLED
    Never raises.
    """
    if now is None:
        now = time.time()
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        row = {"t": round(float(now), 3), "hunt_id": hunt_id, "event": event}
        row.update(payload)
        with open(EVENTS_FILE, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 hunt_log.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== hunt_log.py self-check ===\n")

    # 1. Hunt ID formatting
    assert format_hunt_id(1) == "H-0001"
    assert format_hunt_id(42) == "H-0042"
    assert format_hunt_id(9999) == "H-9999"
    assert format_hunt_id(10000) == "H-10000"
    assert format_hunt_id(10001) == "H-10001"
    assert format_hunt_id(123456) == "H-123456"
    print("[PASS] hunt id: H-0001 / H-9999 / H-10000 / H-10001 / H-123456")

    # 2. Counter advances and is monotonic
    st = {}
    ids = [next_hunt_id(st) for _ in range(3)]
    assert ids == ["H-0001", "H-0002", "H-0003"], ids
    assert st["hunt_counter"] == 3
    print(f"[PASS] counter: {ids}, state counter = {st['hunt_counter']}")

    # 3. Tick + event writes (sandboxed so the live data dir is untouched)
    _live_ticks, _live_events = TICKS_DIR, EVENTS_FILE
    sandbox = tempfile.mkdtemp(prefix="hunter_log_test_")
    TICKS_DIR = os.path.join(sandbox, "hunt_ticks")
    EVENTS_FILE = os.path.join(sandbox, "events.jsonl")

    record_hunt_tick("H-0001", {"bid": 100.0, "ask": 101.0, "last": 100.5}, now=1000.0)
    record_hunt_tick("H-0001", {"bid": 100.5, "ask": 101.5, "last": 101.0}, now=1060.0)
    p = os.path.join(TICKS_DIR, "H-0001.jsonl")
    assert os.path.exists(p)
    rows = [json.loads(l) for l in open(p)]
    assert len(rows) == 2
    assert rows[0]["bid"] == 100.0 and rows[1]["bid"] == 100.5
    assert rows[0]["t"] == 1000.0
    print("[PASS] ticks: 2 rows appended, bid/ask/t preserved, ordered")

    record_hunt_event("H-0001", "ENTRY_FILLED", {"symbol": "XRPIDR"}, now=1000.0)
    record_hunt_event("H-0001", "EXIT_TARGET", {"net_pnl_idr": 410.2}, now=2000.0)
    ev = [json.loads(l) for l in open(EVENTS_FILE)]
    assert len(ev) == 2 and ev[0]["event"] == "ENTRY_FILLED"
    assert ev[1]["hunt_id"] == "H-0001" and ev[1]["net_pnl_idr"] == 410.2
    print("[PASS] events: ENTRY_FILLED + EXIT_TARGET appended with hunt_id")

    # 4. Logging never raises on a bad path
    TICKS_DIR = "/proc/definitely/not/writable"
    record_hunt_tick("H-0001", {"bid": 1.0})          # must not raise
    EVENTS_FILE = "/proc/definitely/not/writable.jsonl"
    record_hunt_event("H-0001", "X", {})              # must not raise
    print("[PASS] unwritable path: logging failure swallowed, no exception")

    TICKS_DIR, EVENTS_FILE = _live_ticks, _live_events
    assert TICKS_DIR != "/proc/definitely/not/writable"
    shutil.rmtree(sandbox, ignore_errors=True)
    print(f"[PASS] live data dir untouched ({TICKS_DIR})")

    print("\n=== ALL CHECKS PASSED ===")
