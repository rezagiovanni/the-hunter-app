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

   LIVE MODE USES ITS OWN PREFIX: L-00001, L-00002, ...

   Two reasons. A live hunt and a paper hunt are not the same kind of event -
   one is simulated money and the other is not - so they must never share a
   numbering space or a label. And the live ledger is the one an operator
   actually audits, so it gets its own sequence starting at 1 rather than
   inheriting a paper counter that has been running for weeks.
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
def format_hunt_id(n: int, live: bool = False) -> str:
    """
    H-0001 ... H-9999, then H-10000, H-10001 (natural growth past 4 digits).

    live=True gives L-00001 ... L-99999, the live ledger's own sequence.
    """
    if live:
        if n < 100000:
            return f"L-{n:05d}"
        return f"L-{n}"
    if n < 10000:
        return f"H-{n:04d}"
    return f"H-{n}"


def peek_hunt_id(state: dict, live: bool = False) -> str:
    """The id the next hunt WOULD get, without consuming it.

    An id must only be spent on a hunt that actually exists. Allocating one
    before the order is attempted burned ids on every rejected order, every
    failed live placement and every non-filling resting order, so the operator
    saw L-00003, L-00005, L-00007, L-00010 for four real hunts. Peeking lets a
    failure path name the attempt in its log without touching the sequence.
    """
    key = "live_hunt_counter" if live else "hunt_counter"
    return format_hunt_id(int(state.get(key, 0)) + 1, live=live)


def next_hunt_id(state: dict, live: bool = False) -> str:
    """Allocate the next hunt id and advance the counter in state.

    Live and paper keep SEPARATE counters (live_hunt_counter vs hunt_counter).
    Sharing one counter would make the live sequence jump by however many paper
    hunts happened to run, which is neither useful nor auditable.
    """
    key = "live_hunt_counter" if live else "hunt_counter"
    nxt = int(state.get(key, 0)) + 1
    state[key] = nxt
    return format_hunt_id(nxt, live=live)


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
def renumber_ledger(state: dict, live: bool = True) -> list:
    """Close the gaps left by ids that were allocated but never used.

    Old builds consumed an id before attempting the order, so a rejected order
    or a resting order that never filled burned a number. The trades themselves
    are complete and correctly priced - only their labels have holes.

    Renumbers the ledger in chronological order to 1..N and rewrites the
    counter to match, returning the (old, new) pairs it changed. Nothing else
    is touched: prices, quantities, fees and times stay exactly as recorded.

    Call this ONLY while no hunt is open. A hunt id appears in tick files and
    event logs by name, so renumbering mid-hunt would orphan them.
    """
    key = "live_trade_log" if live else "trade_log"
    log = state.get(key) or []
    if not live and state.get("position"):
        raise RuntimeError("refusing to renumber an open paper position")

    idx = sorted(range(len(log)),
                 key=lambda i: float((log[i] or {}).get("entry_time") or 0))
    changes = []
    for seq, i in enumerate(idx, 1):
        rec = log[i]
        old_id = rec.get("hunt_id")
        new_id = format_hunt_id(seq, live=live)
        if old_id != new_id:
            rec["hunt_id"] = new_id
            rec["hunt_id_prev"] = old_id
            changes.append((old_id, new_id))

    if live:
        state["live_hunt_counter"] = len(idx)
    else:
        state["hunt_counter"] = max(int(state.get("hunt_counter") or 0), len(idx))
    return changes


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

    # 1b. live ids carry their own prefix and their own counter
    assert format_hunt_id(1, live=True) == "L-00001"
    assert format_hunt_id(42, live=True) == "L-00042"
    assert format_hunt_id(99999, live=True) == "L-99999"
    assert format_hunt_id(100000, live=True) == "L-100000"
    print("[PASS] live hunt id: L-00001 / L-99999 / L-100000")

    _mix = {"hunt_counter": 27}
    _l = [next_hunt_id(_mix, live=True) for _ in range(3)]
    assert _l == ["L-00001", "L-00002", "L-00003"], _l
    # the paper counter must not have moved
    assert _mix["hunt_counter"] == 27, _mix
    assert _mix["live_hunt_counter"] == 3
    assert next_hunt_id(_mix) == "H-0028"
    print("[PASS] live and paper counters are independent")

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

    # 5. peek must not consume the id
    _p = {"live_hunt_counter": 4}
    assert peek_hunt_id(_p, live=True) == "L-00005", peek_hunt_id(_p, live=True)
    assert _p["live_hunt_counter"] == 4, "peek consumed the counter"
    assert next_hunt_id(_p, live=True) == "L-00005"
    assert _p["live_hunt_counter"] == 5
    print("[PASS] peek_hunt_id shows the next id without consuming it")

    # 6. renumber closes the gaps left by ids burned on failed orders
    _r = {"live_trade_log": [
        {"hunt_id": "L-00003", "entry_time": 100.0},
        {"hunt_id": "L-00005", "entry_time": 200.0},
        {"hunt_id": "L-00007", "entry_time": 300.0},
        {"hunt_id": "L-00010", "entry_time": 400.0},
    ], "live_hunt_counter": 10}
    _ch = renumber_ledger(_r, live=True)
    assert [t["hunt_id"] for t in _r["live_trade_log"]] == [
        "L-00001", "L-00002", "L-00003", "L-00004"], _r["live_trade_log"]
    assert _r["live_hunt_counter"] == 4, _r["live_hunt_counter"]
    assert _r["live_trade_log"][0]["hunt_id_prev"] == "L-00003"
    assert len(_ch) == 4, _ch
    print("[PASS] renumbering closes gaps and keeps the old id for audit")

    # 7. renumbering must preserve everything except the label
    _before = {"entry_price": 1514.0, "quantity": 13.2, "net_pnl_idr": 181.0,
               "exit_time": 400.0, "entry_time": 400.0}
    _r2 = {"live_trade_log": [dict(_before, hunt_id="L-00010")],
           "live_hunt_counter": 10}
    renumber_ledger(_r2, live=True)
    _after = dict(_r2["live_trade_log"][0])
    _after.pop("hunt_id"); _after.pop("hunt_id_prev")
    assert _after == _before, (_after, _before)
    print("[PASS] renumbering changes only the hunt id, never the money")

    print("\n=== ALL CHECKS PASSED ===")


