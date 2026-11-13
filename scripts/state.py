#!/usr/bin/env python3
"""
state.py — The Hunter's persistent memory.

Provides:
  load_state()        — read hunter_state.json; returns clean default if missing/corrupt
  save_state(state)   — atomic write via tmp + rename (no partial states on crash)
  acquire_lock()      — non-blocking flock; returns fd or None if another tick is running
  release_lock(fd)    — release the flock
  check_kill_switch() — raises HaltSignal if data/hunter_stop exists

No trading logic. No indicators. No API calls.
"""

import fcntl
import json
import os
import tempfile
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Paths come from paths.py so the data directory is resolved once, in one place,
# instead of every module inventing its own idea of where data lives.
try:
    from paths import DATA_DIR, STATE_FILE, LOCK_FILE, KILL_FILE
except ImportError:  # direct execution from another cwd
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import DATA_DIR, STATE_FILE, LOCK_FILE, KILL_FILE


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class HaltSignal(Exception):
    """Raised when the kill switch file is present."""


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------
def _default_state() -> dict:
    return {
        "phase": "HUNTING",          # HUNTING | ORDER_OPEN | HOLDING
        "balance_idr": 100_000,      # paper balance in IDR
        "order": None,               # None or active resting order dict
        "position": None,            # None or position dict (see below)
        "cooldowns": {},             # symbol -> unix timestamp when cooldown expires
        "gate_stats": {},            # gate name -> rejection count (lifetime)
        "trade_log": [],             # completed PAPER trades in current month
        "live_trade_log": [],        # completed LIVE trades (empty until live executor)
        "hunt_counter": 0,           # monotonic hunt id counter (H-0001, ...)
        "live_hunt_counter": 0,      # live hunt counter (L-00001, ...)
        "accounts": {},              # account_id -> {live_trade_log, live_hunt_counter, position, order}
    }


def active_account_id() -> Optional[str]:
    """Derive a stable 8-char account fingerprint from the active API key.

    Returns None if no API key is configured (signed out).
    Purely local: reads config.yaml, 0 network calls.
    """
    try:
        import config as _cfg
        cfg = _cfg.load()
        key = str(_cfg.get(cfg, "api.key") or "").strip()
        if not key:
            explicit = os.environ.get("HUNTER_API_KEY_FILE")
            files = [explicit] if explicit else ["indodax_api_v2.txt", "indodax_api.txt"]
            for fn in files:
                if not fn:
                    continue
                fp = fn if os.path.isabs(fn) else os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", fn)
                if os.path.exists(fp):
                    for ln in open(fp):
                        ln = ln.strip()
                        if "=" in ln and "key" in ln.split("=")[0].lower():
                            key = ln.split("=")[1].strip()
                            break
                        elif ln and not ln.startswith("#") and len(ln) > 20:
                            key = ln
                            break
                if key:
                    break
        if key:
            import hashlib
            return "acc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------
def load_state() -> dict:
    """
    Load hunter_state.json.
    Returns a clean default if the file is missing, empty, or corrupt.
    Crash rule: never raise on a bad file — always return a usable state.

    ACCOUNT ISOLATION:
    Live trade logs and live positions are scoped per active account ID.
    Switching from key1 to key2 loads key2's own trade log. Switching back
    to key1 restores key1's trades and position.
    Paper trades, candle data, and framework configs remain shared and global.
    """
    if not os.path.exists(STATE_FILE):
        state = _default_state()
    else:
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            state = _default_state()
            state.update(data)
        except (json.JSONDecodeError, OSError):
            state = _default_state()

    acc_id = active_account_id()
    state["active_account_id"] = acc_id
    if acc_id:
        accounts = state.setdefault("accounts", {})
        if acc_id not in accounts:
            existing_log = state.get("live_trade_log") or []
            existing_cnt = state.get("live_hunt_counter") or 0
            accounts[acc_id] = {
                "account_id": acc_id,
                "live_trade_log": existing_log if len(accounts) == 0 else [],
                "live_hunt_counter": existing_cnt if len(accounts) == 0 else 0,
                "position": state.get("position") if (state.get("position") or {}).get("live") else None,
                "order": state.get("order") if (state.get("order") or {}).get("live") else None,
            }
        acc_data = accounts[acc_id]
        state["live_trade_log"] = list(acc_data.get("live_trade_log") or [])
        state["live_hunt_counter"] = int(acc_data.get("live_hunt_counter") or 0)
        if acc_data.get("position"):
            state["position"] = acc_data["position"]
        elif (state.get("position") or {}).get("live"):
            state["position"] = None
        if acc_data.get("order"):
            state["order"] = acc_data["order"]
        elif (state.get("order") or {}).get("live"):
            state["order"] = None
    elif state.get("accounts"):
        state["live_trade_log"] = []
        state["live_hunt_counter"] = 0
        if (state.get("position") or {}).get("live"):
            state["position"] = None
        if (state.get("order") or {}).get("live"):
            state["order"] = None

    return state


def save_state(state: dict) -> None:
    """
    Atomic write: write to a tmp file in the same directory, then os.replace.
    If the process is killed mid-write, the original file survives intact.
    """
    acc_id = state.get("active_account_id") or active_account_id()
    if acc_id:
        acc_data = state.setdefault("accounts", {}).setdefault(acc_id, {})
        acc_data["account_id"] = acc_id
        acc_data["live_trade_log"] = list(state.get("live_trade_log") or [])
        acc_data["live_hunt_counter"] = int(state.get("live_hunt_counter") or 0)
        if (state.get("position") or {}).get("live"):
            acc_data["position"] = state.get("position")
        else:
            acc_data["position"] = None
        if (state.get("order") or {}).get("live"):
            acc_data["order"] = state.get("order")
        else:
            acc_data["order"] = None
        acc_data["updated_at"] = time.time()

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    dir_ = os.path.dirname(STATE_FILE)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        # Clean up tmp on any failure; re-raise so the caller knows
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# File lock — prevents two tick processes writing simultaneously
# ---------------------------------------------------------------------------
def acquire_lock():
    """
    Non-blocking exclusive flock on LOCK_FILE.
    Returns an open file descriptor if acquired, None if another tick holds it.
    Caller must pass the fd to release_lock() when done.
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        fd.close()
        return None


def release_lock(fd) -> None:
    """Release and close the flock fd returned by acquire_lock()."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------
def check_kill_switch() -> None:
    """
    Raise HaltSignal if data/hunter_stop exists.
    Call this at the top of every tick before any trading logic.
    To resume: delete the file (rm data/hunter_stop).
    """
    if os.path.exists(KILL_FILE):
        raise HaltSignal(f"Kill switch active — remove {KILL_FILE} to resume.")


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------
def is_on_cooldown(state: dict, symbol: str, now: float = None) -> bool:
    """True if symbol is still within its 4h cooldown window."""
    if now is None:
        now = time.time()
    expires = state["cooldowns"].get(symbol, 0)
    return now < expires


def set_cooldown(state: dict, symbol: str, hours: float = 4.0, now: float = None) -> None:
    """Record a cooldown for symbol starting now."""
    if now is None:
        now = time.time()
    state["cooldowns"][symbol] = now + hours * 3600


def clear_expired_cooldowns(state: dict, now: float = None) -> None:
    """Remove expired cooldown entries to keep the dict tidy."""
    if now is None:
        now = time.time()
    state["cooldowns"] = {
        sym: exp for sym, exp in state["cooldowns"].items() if now < exp
    }


# ---------------------------------------------------------------------------
# Gate stats helper
# ---------------------------------------------------------------------------
def record_gate_rejection(state: dict, gate_name: str) -> None:
    """Increment the lifetime rejection counter for a gate."""
    state["gate_stats"][gate_name] = state["gate_stats"].get(gate_name, 0) + 1


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 state.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-checks run against a TEMP file, never the live state.
    # The tests historically wrote straight to STATE_FILE, which wiped a
    # real paper balance + trade log when the module was run for a check.
    # ------------------------------------------------------------------
    import tempfile
    import shutil

    _LIVE_STATE = STATE_FILE
    _sandbox = tempfile.mkdtemp(prefix="hunter_state_test_")
    STATE_FILE = os.path.join(_sandbox, "hunter_state.json")
    KILL_FILE = os.path.join(_sandbox, "hunter_stop")

    print("=== state.py self-check ===")
    print(f"(sandbox: {_sandbox})\n")

    # 1. Round-trip
    s = _default_state()
    s["balance_idr"] = 99_000
    s["cooldowns"]["NEARIDR"] = time.time() + 3600
    save_state(s)
    s2 = load_state()
    assert s2["balance_idr"] == 99_000, "round-trip balance fail"
    assert "NEARIDR" in s2["cooldowns"], "round-trip cooldown fail"
    print("[PASS] round-trip: save → load returns identical state")

    # 2. Corrupt file → clean default
    with open(STATE_FILE, "w") as f:
        f.write("{broken json")
    s3 = load_state()
    assert s3["phase"] == "HUNTING", "corrupt recovery fail"
    print("[PASS] corrupt file → clean default, no crash")

    # 3. Missing file → clean default
    os.remove(STATE_FILE)
    s4 = load_state()
    assert s4["phase"] == "HUNTING", "missing file fail"
    print("[PASS] missing file → clean default")

    # 4. Atomicity: tmp + rename, no partial state
    save_state(_default_state())  # ensure STATE_FILE exists
    save_state({"phase": "HUNTING", "balance_idr": 77_000,
                "position": None, "cooldowns": {}, "gate_stats": {}})
    s5 = load_state()
    assert s5["balance_idr"] == 77_000, "atomic write fail"
    print("[PASS] atomic write: tmp + rename, no partial state")

    # 5. Flock — acquire twice, second returns None
    fd1 = acquire_lock()
    assert fd1 is not None, "first acquire fail"
    fd2 = acquire_lock()
    assert fd2 is None, "second acquire should return None (busy)"
    release_lock(fd1)
    fd3 = acquire_lock()
    assert fd3 is not None, "re-acquire after release fail"
    release_lock(fd3)
    print("[PASS] flock: two concurrent acquires → second skipped")

    # 6. Kill switch
    open(KILL_FILE, "w").close()
    try:
        check_kill_switch()
        assert False, "HaltSignal not raised"
    except HaltSignal:
        pass
    os.remove(KILL_FILE)
    check_kill_switch()  # should not raise
    print("[PASS] kill switch: raises HaltSignal when file present")

    # 7. Cooldown helpers
    now = time.time()
    s6 = _default_state()
    set_cooldown(s6, "PEPEIDR", hours=4, now=now)
    assert is_on_cooldown(s6, "PEPEIDR", now=now + 1), "cooldown active fail"
    assert not is_on_cooldown(s6, "PEPEIDR", now=now + 4 * 3600 + 1), "cooldown expired fail"
    assert not is_on_cooldown(s6, "BTCIDR", now=now), "missing cooldown fail"
    print("[PASS] cooldown helpers: set, active, expired, missing")

    # 8. Live state untouched by the run above
    assert STATE_FILE != _LIVE_STATE, "sandbox escaped into the live state path"
    print(f"[PASS] live state untouched ({_LIVE_STATE})")

    # 9. Account profile isolation
    _orig_aid = globals()["active_account_id"]
    try:
        # a) Account 1 logs a live trade
        globals()["active_account_id"] = lambda: "acc_test_1"
        st_a1 = load_state()
        st_a1["live_trade_log"].append({"hunt_id": "L-00001", "symbol": "HYPEIDR", "net_pnl_idr": 500})
        st_a1["live_hunt_counter"] = 1
        save_state(st_a1)

        # b) Sign out: live trade log is empty
        globals()["active_account_id"] = lambda: None
        st_out = load_state()
        assert st_out["live_trade_log"] == [], st_out["live_trade_log"]
        assert st_out["live_hunt_counter"] == 0

        # c) Account 2 signs in: fresh trade log, independent counter
        globals()["active_account_id"] = lambda: "acc_test_2"
        st_a2 = load_state()
        assert st_a2["live_trade_log"] == [], "account 2 must not see account 1 trades"
        assert st_a2["live_hunt_counter"] == 0
        st_a2["live_trade_log"].append({"hunt_id": "L-00001", "symbol": "UAIIDR", "net_pnl_idr": -200})
        st_a2["live_hunt_counter"] = 1
        save_state(st_a2)

        # d) Switch back to Account 1: full history restored!
        globals()["active_account_id"] = lambda: "acc_test_1"
        st_a1_restored = load_state()
        assert len(st_a1_restored["live_trade_log"]) == 1
        assert st_a1_restored["live_trade_log"][0]["symbol"] == "HYPEIDR"
        assert st_a1_restored["live_trade_log"][0]["net_pnl_idr"] == 500
        assert st_a1_restored["live_hunt_counter"] == 1

        # e) Switch back to Account 2: its history also intact
        globals()["active_account_id"] = lambda: "acc_test_2"
        st_a2_restored = load_state()
        assert len(st_a2_restored["live_trade_log"]) == 1
        assert st_a2_restored["live_trade_log"][0]["symbol"] == "UAIIDR"

        print("[PASS] account profiles: live history isolated per key, restored on switch")
    finally:
        globals()["active_account_id"] = _orig_aid

    shutil.rmtree(_sandbox, ignore_errors=True)
    print("\n=== ALL CHECKS PASSED ===")
