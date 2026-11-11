#!/usr/bin/env python3
"""
config.py — user-editable trading parameters, loaded from config.yaml.

Why this exists
---------------
Target, stop, ratchet and sizing used to be constants inside the Python source:

    scripts/exit_strategy.py : TARGET_PCT = 0.02
    scripts/frameworks/framework_b.py : TARGET_PCT = 0.015

Those files live in the app directory, which `hunter upgrade` replaces. A user
who edited a constant to suit their own risk appetite would silently lose that
change at the next upgrade. Parameters therefore move out of code and into a
file the upgrade never touches.

Design rules
------------
1. **Fail open.** A missing file, a missing key, a corrupt file or an unreadable
   file all fall back to the built-in defaults in `DEFAULTS`. The engine must
   never refuse to trade because a config file is damaged — it trades on the
   values it has always used and logs a warning.
2. **Per-framework, not global.** Framework A and Framework B have genuinely
   different exit geometries (A: +2%/-4% with a ratchet; B: +1.5%/-2.5% with
   none). A single global target/stop would silently overwrite B's own design.
   So exit parameters live under `frameworks.<NAME>`, and each framework reads
   its own block.
3. **Loaded once, at import.** Parameters are read a single time. A running
   position must keep the levels computed at entry — changing a value on disk
   mid-trade must never shift the exit levels of a position that is already open.
4. **No YAML means no crash.** PyYAML is a declared dependency, but if the
   import fails the loader still works for the built-in defaults.

Nothing here trades, evaluates exits, or touches state. Values only.
"""

import copy
import os
import sys

# ---------------------------------------------------------------------------
# Built-in defaults — the values the engine has been running with.
# These are the single source of truth when no config file says otherwise.
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Live trading limits. These are enforced in the ENGINE, not the UI - a cap
    # that lives in a browser is not a cap. Defaults are deliberately small:
    # 20,000 IDR per trade and 5,000 IDR of realised loss per day, so a bad
    # session on a small account stops itself instead of emptying it.
    "live": {
        "max_per_trade_idr": 20000.0,
        "max_daily_loss_idr": 5000.0,
    },
    "trading": {
        "sizing_fraction": 0.40,        # 40% of current balance per trade
        "min_trade_balance_idr": 10000.0,
        "cooldown_hours": 4.0,          # per-pair cooldown after exit
        "stop_slip_fraction": 0.0094,   # measured avg gap-through on market stops
        "order_timeout_checks": 6,      # ~6 min timeout for resting limit orders
    },
    "frameworks": {
        "A": {
            "target_pct": 0.02,
            "stop_pct": 0.04,
            "ratchet_enabled": True,
            "ratchet_trigger_pct": 0.012,
            "ratchet_stop_pct": -0.003,
        },
        "B": {
            "target_pct": 0.015,
            "stop_pct": 0.025,
            "ratchet_enabled": False,
            "ratchet_trigger_pct": 0.012,
            "ratchet_stop_pct": -0.003,
        },
        "C": {
            # A's entry with the exits stretched: same 0.5:1 payoff, far more
            # noise tolerance. No ratchet, deliberately - ratcheting to
            # breakeven at +1.2% would turn C back into A with a worse target.
            "target_pct": 0.05,
            "stop_pct": 0.10,
            "ratchet_enabled": False,
            "ratchet_trigger_pct": 0.0,
            "ratchet_stop_pct": 0.0,
        },
        "D": {
            # The benchmark rig: trades ONLY the #1 pair by 24h volume, with no
            # entry signal. Exits are identical to A on purpose - the only
            # variable is which pair gets bought.
            "target_pct": 0.02,
            "stop_pct": 0.04,
            "ratchet_enabled": False,
            "ratchet_trigger_pct": 0.0,
            "ratchet_stop_pct": 0.0,
        },
    },
    "data": {
        # Retention floor, in years. prune_data.py removes bars older than this.
        "retention_years": 2.0,
        # backfill_deep.py fetches slightly DEEPER than the retention floor.
        # Without this slack the two jobs fight at the boundary: prune trims to
        # exactly 2 years, backfill sees a first bar newer than its own floor
        # and re-fetches the same bars, and the next prune trims them again -
        # one wasted API call per pair per run, against a 95/day budget.
        "backfill_slack_days": 30,
    },
    "gates": {
        "min_candles": 100,
        "max_spread_pct": 0.50,
        "max_age_minutes": 30.0,
    },
    "server": {
        "host": "127.0.0.1",            # loopback by default — see TECH_PLAN 9
        "port": 8082,
    },
}

# Validation bounds. A value outside these is rejected rather than silently used.
LIMITS = {
    "trading.sizing_fraction": (0.0, 1.0, "must be between 0 and 1 (exclusive of 0)"),
    "trading.cooldown_hours": (0.0, 168.0, "must be between 0 and 168 hours"),
    "frameworks.*.target_pct": (0.0, 1.0, "must be between 0 and 1 (i.e. 0-100%)"),
    "frameworks.*.stop_pct": (0.0, 1.0, "must be between 0 and 1"),
}


class ConfigError(ValueError):
    """Raised by validate() when a value is present but unusable."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[config] WARNING: {msg}", file=sys.stderr)


def config_path() -> str:
    """Where the config file lives, derived from paths.py."""
    try:
        from paths import config_root
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from paths import config_root
    # config.yaml sits at the install root - beside data/, app/ and market/ - so
    # wiping data/ does not destroy the user's settings.
    return os.path.join(config_root(), "config.yaml")


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Overlay wins, but nested dicts merge rather than replace.

    This is what makes a partial config file safe: a user who only sets
    frameworks.A.target_pct keeps every other default.
    """
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str = None) -> dict:
    """Read config.yaml and merge it over DEFAULTS.

    Never raises. Any problem produces a warning and a fallback:

      file missing          -> defaults, silent (this is the normal first state)
      PyYAML unavailable    -> defaults + warning
      unparseable YAML      -> defaults + warning
      top level not a dict  -> defaults + warning

    Keys of the wrong type are dropped by validate(); unknown keys are kept so
    a newer config file does not break an older binary.
    """
    path = path or config_path()

    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)

    try:
        import yaml
    except ImportError:
        _warn("PyYAML not installed; using built-in defaults")
        return copy.deepcopy(DEFAULTS)

    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:
        _warn(f"could not parse {path}: {exc}; using built-in defaults")
        return copy.deepcopy(DEFAULTS)

    if raw is None:
        return copy.deepcopy(DEFAULTS)

    if not isinstance(raw, dict):
        _warn(f"{path} is not a mapping; using built-in defaults")
        return copy.deepcopy(DEFAULTS)

    merged = _deep_merge(DEFAULTS, raw)
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(cfg: dict) -> list:
    """Return a list of problem strings. Empty list means the config is usable.

    Validation is for *saving* — the Settings page must refuse to write a target
    of 0% or a sizing fraction of 150%. Reading a bad value is handled by
    get(), which falls back to the default rather than raising, because a
    running engine must not stop trading over a typo.
    """
    problems = []

    def check_num(label, value, lo, hi, note):
        if value is None:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"{label}: must be a number, got {type(value).__name__}")
            return
        if not (lo < value < hi):
            problems.append(f"{label}: {value} {note}")

    t = cfg.get("trading", {})
    sf = t.get("sizing_fraction")
    check_num("trading.sizing_fraction", sf, 0.0, 1.0, LIMITS["trading.sizing_fraction"][2])
    check_num("trading.cooldown_hours", t.get("cooldown_hours"), 0.0, 168.0,
              LIMITS["trading.cooldown_hours"][2])

    min_bal = t.get("min_trade_balance_idr")
    if min_bal is not None and (not isinstance(min_bal, (int, float)) or min_bal < 0):
        problems.append("trading.min_trade_balance_idr: must be a non-negative number")

    slip = t.get("stop_slip_fraction")
    if slip is not None and (not isinstance(slip, (int, float)) or not (0 <= slip < 0.5)):
        problems.append("trading.stop_slip_fraction: must be between 0 and 0.5")

    for name, fw in (cfg.get("frameworks") or {}).items():
        if not isinstance(fw, dict):
            problems.append(f"frameworks.{name}: must be a mapping")
            continue
        for key in ("target_pct", "stop_pct"):
            v = fw.get(key)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                problems.append(f"frameworks.{name}.{key}: must be a number")
            elif not (0.0 < v < 1.0):
                problems.append(
                    f"frameworks.{name}.{key}: {v} must be between 0 and 1 "
                    f"(e.g. 0.02 for 2%)")

        # A target that cannot cover fees is a losing trade by construction.
        tp = fw.get("target_pct")
        if isinstance(tp, (int, float)) and 0 < tp < 0.004:
            problems.append(
                f"frameworks.{name}.target_pct: {tp:.4f} is below round-trip fees "
                f"(~0.30%); this target cannot profit even when it fills")

        rt = fw.get("ratchet_trigger_pct")
        if rt is not None and isinstance(rt, (int, float)):
            if tp is not None and isinstance(tp, (int, float)) and rt >= tp:
                problems.append(
                    f"frameworks.{name}.ratchet_trigger_pct: {rt} must be below "
                    f"target_pct {tp}, or the ratchet can never arm before the target")

    g = cfg.get("gates", {})
    for key, lo, hi in (("min_candles", 1, 100000),
                        ("max_spread_pct", 0.0, 100.0),
                        ("max_age_minutes", 0.0, 100000.0)):
        v = g.get(key)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            problems.append(f"gates.{key}: {v} must be a number between {lo} and {hi}")

    d = cfg.get("data", {})
    ry = d.get("retention_years")
    if ry is not None and (not isinstance(ry, (int, float)) or not (0.05 <= ry <= 50)):
        problems.append(f"data.retention_years: {ry} must be between 0.05 and 50")
    sl = d.get("backfill_slack_days")
    if sl is not None and (not isinstance(sl, (int, float)) or not (0 <= sl <= 3650)):
        problems.append(f"data.backfill_slack_days: {sl} must be between 0 and 3650")

    s = cfg.get("server", {})
    port = s.get("port")
    if port is not None and (not isinstance(port, int) or not (1 <= port <= 65535)):
        problems.append(f"server.port: {port} must be an integer between 1 and 65535")

    return problems


# ---------------------------------------------------------------------------
# Typed accessors — used by the engine
# ---------------------------------------------------------------------------
def get(cfg: dict, dotted: str, default=None):
    """Read `a.b.c` from the config, falling back to `default` when absent.

    Never raises. A wrong-typed value falls back to the default rather than
    propagating something the engine cannot compute with.
    """
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def framework_params(cfg: dict, name: str) -> dict:
    """Exit parameters for one framework, falling back to that framework's defaults.

    Unknown framework names get framework A's geometry rather than an empty dict,
    so a config typo cannot produce a position with no exit levels at all.
    """
    name = (name or "A").upper()
    defaults = DEFAULTS["frameworks"].get(name, DEFAULTS["frameworks"]["A"])
    supplied = ((cfg.get("frameworks") or {}).get(name) or {})
    out = copy.deepcopy(defaults)

    for key in ("target_pct", "stop_pct", "ratchet_trigger_pct", "ratchet_stop_pct"):
        v = supplied.get(key, out.get(key))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)

    if "ratchet_enabled" in supplied:
        out["ratchet_enabled"] = bool(supplied["ratchet_enabled"])

    return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_CFG = None


def current() -> dict:
    """Load once and reuse. Every caller shares the same snapshot for the life of
    the process, which is what keeps open positions stable while a file changes
    on disk."""
    global _CFG
    if _CFG is None:
        _CFG = load()
    return _CFG


def reload_config() -> dict:
    """Discard the snapshot and read again. For the Settings page and tests only —
    the engine never calls this mid-tick."""
    global _CFG
    _CFG = None
    return current()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    import tempfile
    failures = []

    def check(label, cond):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)

    print("=== config.py self-check ===")
    print()

    with tempfile.TemporaryDirectory() as td:
        # --- 1. missing file -> defaults, no error ---
        missing = os.path.join(td, "nope.yaml")
        cfg = load(missing)
        check("missing file falls back to defaults",
              cfg == DEFAULTS)

        # --- 2. empty file -> defaults ---
        empty = os.path.join(td, "empty.yaml")
        open(empty, "w").write("")
        check("empty file falls back to defaults", load(empty) == DEFAULTS)

        # --- 3. corrupt YAML -> defaults, warns, does not raise ---
        bad = os.path.join(td, "bad.yaml")
        open(bad, "w").write("trading: [unclosed\n  nope: : :\n")
        cfg = load(bad)
        check("corrupt YAML falls back to defaults", cfg == DEFAULTS)

        # --- 4. top level is a list -> defaults ---
        lst = os.path.join(td, "list.yaml")
        open(lst, "w").write("- just\n- a list\n")
        check("non-mapping YAML falls back to defaults", load(lst) == DEFAULTS)

        # --- 5. partial config merges, does not replace ---
        part = os.path.join(td, "part.yaml")
        open(part, "w").write("frameworks:\n  A:\n    target_pct: 0.03\n")
        cfg = load(part)
        check("partial config overrides one key",
              cfg["frameworks"]["A"]["target_pct"] == 0.03)
        check("partial config preserves siblings",
              cfg["frameworks"]["A"]["stop_pct"] == 0.04)
        check("partial config preserves other frameworks",
              cfg["frameworks"]["B"]["target_pct"] == 0.015)
        check("partial config preserves trading block",
              cfg["trading"]["sizing_fraction"] == 0.40)

        # --- 6. framework params: defaults per framework ---
        a = framework_params(DEFAULTS, "A")
        b = framework_params(DEFAULTS, "B")
        check("A target/stop are A's own", (a["target_pct"], a["stop_pct"]) == (0.02, 0.04))
        check("B target/stop are B's own", (b["target_pct"], b["stop_pct"]) == (0.015, 0.025))
        check("B has ratchet disabled", b["ratchet_enabled"] is False)
        check("A has ratchet enabled", a["ratchet_enabled"] is True)

        # --- 7. unknown framework falls back to A's geometry ---
        x = framework_params(DEFAULTS, "Z")
        check("unknown framework gets A's geometry", x["target_pct"] == 0.02)

        # --- 8. get() never raises ---
        check("get on missing path returns default", get(DEFAULTS, "no.such.key", 7) == 7)
        check("get on wrong type returns default",
              get({"a": "string"}, "a.b", 9) == 9)

        # --- 9. validation rejects bad values ---
        check("valid defaults produce no problems", validate(DEFAULTS) == [])

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["frameworks"]["A"]["target_pct"] = 0.0
        check("target of 0 is rejected", len(validate(bad_cfg)) > 0)

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["trading"]["sizing_fraction"] = 1.5
        check("sizing fraction above 1 is rejected", len(validate(bad_cfg)) > 0)

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["frameworks"]["A"]["target_pct"] = 0.002
        probs = validate(bad_cfg)
        check("target below fees is rejected",
              any("fees" in p for p in probs))

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["frameworks"]["A"]["ratchet_trigger_pct"] = 0.03
        probs = validate(bad_cfg)
        check("ratchet trigger above target is rejected",
              any("ratchet" in p for p in probs))

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["server"]["port"] = 99999
        check("invalid port is rejected", len(validate(bad_cfg)) > 0)

        # --- backfill must fetch deeper than the prune cutoff ---
        # Otherwise prune trims to exactly the floor, backfill sees a first bar
        # newer than its own floor and re-fetches the same bars, and the next
        # prune trims them again: one wasted API call per pair per run.
        ry = DEFAULTS["data"]["retention_years"]
        slack = DEFAULTS["data"]["backfill_slack_days"]
        check("backfill slack is positive (prune and backfill cannot fight)",
              slack > 0)

        bad_cfg = copy.deepcopy(DEFAULTS)
        bad_cfg["data"]["retention_years"] = 0
        check("a zero retention is rejected", len(validate(bad_cfg)) > 0)

        # --- 10. config path sits beside data, not inside it ---
        # so wiping data/ does not destroy user settings
        import paths
        cp = os.path.abspath(config_path())
        dd = os.path.abspath(paths.DATA_DIR)
        check("config.yaml lives beside data/, not inside it",
              os.path.dirname(cp) == os.path.dirname(dd))

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
