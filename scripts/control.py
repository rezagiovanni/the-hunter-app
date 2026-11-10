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

DEFAULT_FRAMEWORK = "A"


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------
def _default_control() -> dict:
    return {
        "mode": MODE_RUNNING,
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
             now: float = None) -> dict:
    """Set the control mode (and optionally the framework) and persist it."""
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {VALID_MODES}")
    if now is None:
        now = time.time()

    ctrl = load_control()
    ctrl["mode"] = mode
    ctrl["changed_at"] = now
    ctrl["changed_by"] = by
    if framework:
        ctrl["framework"] = framework
    save_control(ctrl)
    return ctrl


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

    # 9. Live control file untouched
    assert CONTROL_FILE != _LIVE_CONTROL, "sandbox escaped into the live path"
    print(f"[PASS] live control untouched ({_LIVE_CONTROL})")

    shutil.rmtree(_sandbox, ignore_errors=True)
    print("\n=== ALL CHECKS PASSED ===")
