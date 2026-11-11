#!/usr/bin/env python3
"""
control.py — user-facing START/STOP control plane for The Hunter.

Why this is a SEPARATE file from hunter_state.json:
    The engine does load_state() -> work -> save_state() on every 60s tick.
    A start/stop toggle written into hunter_state.json by the dashboard would
    be silently overwritten by the next tick (up to 60s later). A separate
    file, written atomically by the dashboard and read at tick start, has no
    such race.

Authority model:
    This file is authoritative. If mode != RUNNING the engine refuses to hunt,
    regardless of whether the cron job is actually scheduled. Pausing/resuming
    cron_a is an optimisation (fewer pointless process spawns), never the
    safety mechanism — so a failure to reach the Hermes CLI can never leave
    the bot hunting against the user's wish.

Modes:
    RUNNING   — hunt normally
    STOPPING  — a stop is in flight; the engine liquidates then parks
    STOPPED   — parked; no scanning, no orders

No trading logic here. No API calls. No indicators.
"""

import json
import os
import tempfile
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Paths from paths.py - see that module for why this is centralised.
try:
    from paths import CONTROL_FILE, DATA_DIR
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import CONTROL_FILE, DATA_DIR

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
MODE_RUNNING = "RUNNING"
MODE_STOPPING = "STOPPING"
MODE_STOPPED = "STOPPED"

VALID_MODES = (MODE_RUNNING, MODE_STOPPING, MODE_STOPPED)

# ---------------------------------------------------------------------------
# Execution axis - WHICH executor places the order
# ---------------------------------------------------------------------------
# Deliberately a SECOND axis, independent of mode. "RUNNING" alone does not say
# whether the bot is spending simulated money or real money. A control file that
# cannot tell those apart is one bad default away from trading live by accident,
# so every money-touching code path reads this explicitly instead of inferring.
EXEC_PAPER = "PAPER"
EXEC_LIVE = "LIVE"
VALID_EXEC = (EXEC_PAPER, EXEC_LIVE)

DEFAULT_EXEC = EXEC_PAPER

DEFAULT_FRAMEWORK = "A"


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------
def _default_control() -> dict:
    return {
        "mode": MODE_RUNNING,
        "exec": DEFAULT_EXEC,
        "framework": DEFAULT_FRAMEWORK,
        "changed_at": None,
        "changed_by": None,
    }


def load_control() -> dict:
    """
    Load hunter_control.json.

    Crash rule: a missing or corrupt file means RUNNING — fail-open, so a
    botched write can never silently stop a working bot. (The opposite
    failure — hunting when the user asked to stop — is prevented by the file
    existing at all, and by STOPPING/STOPPED being explicit values.)

    Unknown mode values are coerced to RUNNING rather than trusted, so a
    hand-edited file cannot put the engine into an undefined state.
    """
    if not os.path.exists(CONTROL_FILE):
        return _default_control()
    try:
        with open(CONTROL_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _default_control()

    ctrl = _default_control()
    if isinstance(data, dict):
        ctrl.update(data)
    if ctrl.get("mode") not in VALID_MODES:
        ctrl["mode"] = MODE_RUNNING

    # The execution axis fails SAFE, unlike mode. A control file written by an
    # older version has no "exec" key at all; treating that as LIVE would arm
    # real trading because of a missing field. Anything unrecognised - absent,
    # misspelled, hand-edited to "REAL" - resolves to PAPER.
    if ctrl.get("exec") not in VALID_EXEC:
        ctrl["exec"] = EXEC_PAPER

    if not ctrl.get("framework"):
        ctrl["framework"] = DEFAULT_FRAMEWORK
    return ctrl


def save_control(ctrl: dict) -> None:
    """
    Atomic write: tmp file in the same directory, then os.replace.
    A crash mid-write leaves the previous control file intact.
    """
    os.makedirs(os.path.dirname(CONTROL_FILE), exist_ok=True)
    dir_ = os.path.dirname(CONTROL_FILE)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ctrl, f, indent=2)
        os.replace(tmp_path, CONTROL_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def set_mode(mode: str, by: str = "dashboard", framework: str = None,
             now: float = None, exec_mode: str = None) -> dict:
    """Set the control mode (and optionally the execution mode / framework).

    exec_mode is the SECOND axis. It is only meaningful to change it while
    STOPPED, so switching executor mid-trade is impossible by construction
    rather than by remembering to check. The engine refuses to start in a mode
    that does not match its executor, and this refuses to change executor while
    RUNNING.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {VALID_MODES}")
    if exec_mode is not None and exec_mode not in VALID_EXEC:
        raise ValueError(f"invalid exec {exec_mode!r}; expected one of {VALID_EXEC}")
    if now is None:
        now = time.time()

    ctrl = load_control()

    if exec_mode is not None and exec_mode != ctrl.get("exec"):
        if ctrl.get("mode") != MODE_STOPPED:
            raise ValueError(
                "the execution mode can only change while STOPPED - "
                "stop the engine first so no position is mid-trade")
        ctrl["exec"] = exec_mode

    ctrl["mode"] = mode
    ctrl["changed_at"] = now
    ctrl["changed_by"] = by
    if framework:
        ctrl["framework"] = framework
    save_control(ctrl)
    return ctrl


def set_blocked(reason: str, by: str = "engine", now: float = None) -> dict:
    """Record that the engine refused to tick, and why.

    Without this the dashboard can only see mode=RUNNING and paints "HUNTING",
    which is how an operator ends up asking three times why a WINNER is not
    being bought. The refusal is a fact about the machine and belongs in the
    control file, not in a log file nobody reads.

    Cleared by any successful tick via clear_blocked().
    """
    if now is None:
        now = time.time()
    ctrl = load_control()
    ctrl["blocked"] = {"reason": str(reason)[:300], "at": now, "by": by}
    save_control(ctrl)
    return ctrl


def clear_blocked(now: float = None) -> None:
    """Clear the block marker after a tick that actually ran."""
    ctrl = load_control()
    if ctrl.get("blocked"):
        ctrl["blocked"] = None
        save_control(ctrl)


def is_live(ctrl: dict = None) -> bool:
    """True only when the control file explicitly says LIVE.

    Deliberately strict: anything else, including a corrupt or absent file,
    is paper. This is the one predicate that decides whether money moves.
    """
    c = load_control() if ctrl is None else ctrl
    return c.get("exec") == EXEC_LIVE


def set_framework(name: str, by: str = "dashboard", now: float = None) -> dict:
    """Set the active framework. Caller must enforce the STOPPED rule."""
    if now is None:
        now = time.time()
    ctrl = load_control()
    ctrl["framework"] = (name or DEFAULT_FRAMEWORK).upper()
    ctrl["changed_at"] = now
    ctrl["changed_by"] = by
    save_control(ctrl)
    return ctrl


def is_running() -> bool:
    """True only in RUNNING — STOPPING and STOPPED both mean 'do not hunt'."""
    return load_control()["mode"] == MODE_RUNNING


def active_framework() -> str:
    return load_control().get("framework", DEFAULT_FRAMEWORK)


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 control.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-checks run against a TEMP file, never the live control file.
    # Same discipline as state.py: a self-check must not be able to
    # change how the real bot behaves.
    # ------------------------------------------------------------------
    import shutil

    _LIVE_CONTROL = CONTROL_FILE
    _sandbox = tempfile.mkdtemp(prefix="hunter_control_test_")
    CONTROL_FILE = os.path.join(_sandbox, "hunter_control.json")

    print("=== control.py self-check ===")
    print(f"(sandbox: {_sandbox})\n")

    # 1. Missing file -> RUNNING (fail-open)
    c = load_control()
    assert c["mode"] == MODE_RUNNING, c
    assert c["framework"] == DEFAULT_FRAMEWORK, c
    print("[PASS] missing file -> RUNNING, framework A")

    # 2. Round-trip through every mode
    for mode in (MODE_STOPPED, MODE_STOPPING, MODE_RUNNING):
        set_mode(mode, by="test")
        c = load_control()
        assert c["mode"] == mode, f"round-trip {mode} failed: {c}"
    print("[PASS] round-trip: STOPPED -> STOPPING -> RUNNING")

    # 3. is_running() is strict: only RUNNING counts
    set_mode(MODE_STOPPED)
    assert not is_running()
    set_mode(MODE_STOPPING)
    assert not is_running(), "STOPPING must not be treated as running"
    set_mode(MODE_RUNNING)
    assert is_running()
    print("[PASS] is_running(): RUNNING only; STOPPING/STOPPED both block")

    # 4. Corrupt file -> RUNNING, no raise
    with open(CONTROL_FILE, "w") as f:
        f.write("{broken json")
    c = load_control()
    assert c["mode"] == MODE_RUNNING, c
    print("[PASS] corrupt file -> RUNNING, no crash")

    # 5. Unknown mode value is coerced, not trusted
    with open(CONTROL_FILE, "w") as f:
        json.dump({"mode": "BANANA", "framework": "A"}, f)
    c = load_control()
    assert c["mode"] == MODE_RUNNING, c
    print("[PASS] unknown mode 'BANANA' -> coerced to RUNNING")

    # 6. Framework persists and uppercases
    set_framework("b", by="test")
    assert active_framework() == "B", active_framework()
    set_framework("a", by="test")
    assert active_framework() == "A"
    print("[PASS] framework round-trip: b -> B, a -> A")

    # 7. Atomic write leaves no .tmp files behind
    set_mode(MODE_RUNNING)
    leftovers = [f for f in os.listdir(_sandbox) if f.endswith(".tmp")]
    assert not leftovers, f"tmp files left behind: {leftovers}"
    print("[PASS] atomic write: no .tmp files left behind")

    # 8. Invalid mode is rejected loudly
    try:
        set_mode("NONSENSE")
        assert False, "set_mode should reject an invalid mode"
    except ValueError:
        pass
    print("[PASS] set_mode rejects invalid mode")

    # --- the execution axis -------------------------------------------------
    # Saved and restored around the block so a test never leaves the real
    # control file on LIVE.
    _saved = dict(load_control())
    try:
        set_mode(MODE_STOPPED)
        r = set_mode(MODE_STOPPED, exec_mode=EXEC_LIVE)
        assert r["exec"] == EXEC_LIVE, r
        assert is_live() is True
        r = set_mode(MODE_STOPPED, exec_mode=EXEC_PAPER)
        assert r["exec"] == EXEC_PAPER and is_live() is False
        print("[PASS] exec survives a round-trip (PAPER <-> LIVE) while stopped")

        # changing executor while RUNNING is refused
        set_mode(MODE_RUNNING, exec_mode=EXEC_PAPER)
        try:
            set_mode(MODE_RUNNING, exec_mode=EXEC_LIVE)
            raise AssertionError("should have refused to arm LIVE while running")
        except ValueError as e:
            assert "STOPPED" in str(e), e
        assert is_live() is False, "a refused switch must not half-apply"
        print("[PASS] arming LIVE while RUNNING is refused, and leaves PAPER")

        # an unknown exec value is rejected outright
        try:
            set_mode(MODE_STOPPED, exec_mode="REAL")
            raise AssertionError("should have rejected exec='REAL'")
        except ValueError as e:
            assert "exec" in str(e)
        print("[PASS] an unknown execution mode is rejected")

        # FAIL-SAFE: a control file with no exec key is PAPER, never LIVE
        import json as _json
        with open(CONTROL_FILE, "w") as _f:
            _json.dump({"mode": MODE_STOPPED, "framework": "A"}, _f)
        c = load_control()
        assert c["exec"] == EXEC_PAPER, c
        assert is_live() is False
        print("[PASS] a control file with no exec key resolves to PAPER")

        # FAIL-SAFE: a corrupt exec value resolves to PAPER
        for _bad in ("LIVE ", "live", "LIVEX", 1, None, ["LIVE"]):
            with open(CONTROL_FILE, "w") as _f:
                _json.dump({"mode": MODE_STOPPED, "exec": _bad}, _f)
            c = load_control()
            assert c["exec"] == EXEC_PAPER, f"{_bad!r} -> {c['exec']}"
        print("[PASS] any unrecognised exec value resolves to PAPER, including 'live'")

        # a missing file is PAPER
        import os as _os
        _os.remove(CONTROL_FILE)
        assert load_control()["exec"] == EXEC_PAPER
        assert is_live() is False
        print("[PASS] a missing control file is PAPER")
    finally:
        save_control(_saved)

    # 9. Live control file untouched
    assert CONTROL_FILE != _LIVE_CONTROL, "sandbox escaped into the live path"
    print(f"[PASS] live control untouched ({_LIVE_CONTROL})")

    shutil.rmtree(_sandbox, ignore_errors=True)
    print("\n=== ALL CHECKS PASSED ===")
