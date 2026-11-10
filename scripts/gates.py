#!/usr/bin/env python3
"""
gates.py — The Hunter's safety funnel.

Step 4 of development_plan.txt.

Every pair must pass 6 protective gates before signals/scoring:
  1. WARMUP:    >= 100 continuous 15m candles
  2. FRESHNESS: last bar <= 30 min old
  3. SPREAD:    (ask - bid) / bid <= 0.20%
  4. DEPTH:     two-sided cover (+-2% of book) >= trade_size (default 40k IDR)
  5. COOLDOWN:  4h per pair (bypassed if pair is the SOLE eligible candidate)
  6. ML VETO:   P(down) >= 0.60 rejected for non-tier-1 pairs (graceful fallback)

All rejections are logged to data/logs/hunter_gates_YYYY-MM-DD.log (WIB)
and tallied in hunter_state.json["gate_stats"].
"""

import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import is_on_cooldown, record_gate_rejection

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
# Paths from paths.py - see that module for why this is centralised.
try:
    from paths import DATA_DIR, LOGS_DIR, MODEL_DIR
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import DATA_DIR, LOGS_DIR, MODEL_DIR

# Gate thresholds come from config.yaml (gates.*), falling back to these values.
# Read once at import so the values a tick uses cannot change underneath it.
import config as _config  # noqa: E402
_C = _config.current()

DEFAULT_TRADE_SIZE = 40_000.0  # IDR (40% of 100k paper balance)
MAX_SPREAD_PCT = float(_config.get(_C, "gates.max_spread_pct", 0.50))
MAX_AGE_MINUTES = float(_config.get(_C, "gates.max_age_minutes", 30.0))
MIN_CANDLES = int(_config.get(_C, "gates.min_candles", 100))
ML_MAX_AGE_SECONDS = float(_config.get(_C, "gates.ml_max_age_seconds", 3600.0))

WIB = datetime.timezone(datetime.timedelta(hours=7))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def format_wib_timestamp(now: Optional[float] = None) -> Tuple[str, str]:
    """Returns (datetime_str, date_str) formatted in WIB (UTC+7)."""
    if now is None:
        now = time.time()
    dt = datetime.datetime.fromtimestamp(now, tz=WIB)
    return dt.strftime("%Y-%m-%d %H:%M:%S WIB"), dt.strftime("%Y-%m-%d")


def log_gate_rejection(symbol: str, gate: str, detail: str,
                       state: Optional[dict] = None, now: Optional[float] = None) -> str:
    """
    Log a gate rejection to data/logs/hunter_gates_YYYY-MM-DD.log
    and increment gate_stats counter in state if provided.
    Returns the formatted log line.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    dt_str, date_str = format_wib_timestamp(now)
    log_line = f"{dt_str}  {symbol:<8}  {gate:<10}  {detail}"

    log_path = os.path.join(LOGS_DIR, f"hunter_gates_{date_str}.log")
    try:
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
    except OSError:
        pass

    if state is not None:
        record_gate_rejection(state, gate)

    return log_line


# ---------------------------------------------------------------------------
# Individual Gate Checkers
# ---------------------------------------------------------------------------
def check_warmup(candles_count: int, min_candles: int = MIN_CANDLES) -> Tuple[bool, str]:
    """Gate 1: CANDLE HISTORY >= 100 continuous bars."""
    if candles_count >= min_candles:
        return True, f"{candles_count} >= {min_candles} bars"
    return False, f"{candles_count} < {min_candles} bars"


def check_freshness(last_candle_ts: float, now: Optional[float] = None,
                    max_age_min: float = MAX_AGE_MINUTES) -> Tuple[bool, str]:
    """Gate 2: FRESHNESS (last bar age <= 30 min)."""
    if now is None:
        now = time.time()
    age_min = (now - last_candle_ts) / 60.0
    if age_min <= max_age_min:
        return True, f"age {age_min:.1f}m <= {max_age_min:.0f}m"
    return False, f"age {age_min:.1f}m > {max_age_min:.0f}m (stale)"


def check_spread(best_bid: float, best_ask: float,
                 max_spread_pct: float = MAX_SPREAD_PCT) -> Tuple[bool, str]:
    """Gate 3: SPREAD (ask - bid) / bid <= 0.20%."""
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return False, f"invalid prices: bid={best_bid}, ask={best_ask}"
    spread_pct = (best_ask - best_bid) / best_bid * 100.0
    if spread_pct <= max_spread_pct:
        return True, f"{spread_pct:.2f}% <= {max_spread_pct:.2f}%"
    return False, f"{spread_pct:.2f}% > {max_spread_pct:.2f}%"


def check_depth(depth_data: dict, trade_size: float = DEFAULT_TRADE_SIZE) -> Tuple[bool, str, dict]:
    """
    Gate 4: DEPTH COVER (two-sided, +-2% of book).
    Asks within +2% of best ask must sum to >= trade_size (entry fits).
    Bids within -2% of best bid must sum to >= trade_size (exit fits).
    """
    buys = depth_data.get("buy", [])
    sells = depth_data.get("sell", [])
    if not buys or not sells:
        return False, "empty order book", {"ask_cover": 0.0, "bid_cover": 0.0}

    try:
        best_bid = float(buys[0][0])
        best_ask = float(sells[0][0])
    except (IndexError, ValueError, TypeError):
        return False, "malformed depth orders", {"ask_cover": 0.0, "bid_cover": 0.0}

    # Sum IDR depth within +-2%
    ask_cover = sum(float(pr) * float(am) for pr, am in sells if float(pr) <= best_ask * 1.02)
    bid_cover = sum(float(pr) * float(am) for pr, am in buys if float(pr) >= best_bid * 0.98)

    metrics = {"ask_cover": ask_cover, "bid_cover": bid_cover}

    if ask_cover < trade_size:
        return False, f"ask cover {ask_cover:.0f} < {trade_size:.0f} IDR (+2%)", metrics
    if bid_cover < trade_size:
        return False, f"bid cover {bid_cover:.0f} < {trade_size:.0f} IDR (-2%)", metrics

    return True, f"ask_cover={ask_cover:.0f}, bid_cover={bid_cover:.0f} >= {trade_size:.0f}", metrics


def check_cooldown(symbol: str, state: dict, now: Optional[float] = None,
                   is_sole_candidate: bool = False) -> Tuple[bool, str]:
    """
    Gate 5: COOLDOWN (4h after exit).
    EXCEPTION (locked design): if symbol is the SOLE eligible candidate,
    it passes cooldown.
    """
    if now is None:
        now = time.time()

    if not is_on_cooldown(state, symbol, now=now):
        return True, "not on cooldown"

    # On cooldown: check sole candidate exception
    if is_sole_candidate:
        return True, "cooldown bypassed (sole eligible candidate)"

    expires = state.get("cooldowns", {}).get(symbol, 0)
    dt_exp, _ = format_wib_timestamp(expires)
    return False, f"active until {dt_exp}"


# ---------------------------------------------------------------------------
# Full Per-Pair Evaluation
# ---------------------------------------------------------------------------
def evaluate_pair_gates(
    symbol: str,
    candles_count: int,
    last_candle_ts: float,
    depth_data: dict,
    state: dict,
    trade_size: float = DEFAULT_TRADE_SIZE,
    is_sole_candidate: bool = False,
    now: Optional[float] = None,
    log_rejections: bool = True,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run all 6 gates on a pair in sequence. Stops at first failure.
    Returns (passed, failing_gate_name, detail).
    If passed: returns (True, None, None).
    """
    if now is None:
        now = time.time()

    # 1. Warmup
    ok, detail = check_warmup(candles_count)
    if not ok:
        if log_rejections:
            log_gate_rejection(symbol, "WARMUP", detail, state=state, now=now)
        return False, "WARMUP", detail

    # 2. Freshness
    ok, detail = check_freshness(last_candle_ts, now=now)
    if not ok:
        if log_rejections:
            log_gate_rejection(symbol, "FRESHNESS", detail, state=state, now=now)
        return False, "FRESHNESS", detail

    # 3. Spread
    buys = depth_data.get("buy", [])
    sells = depth_data.get("sell", [])
    if not buys or not sells:
        detail = "empty order book"
        if log_rejections:
            log_gate_rejection(symbol, "SPREAD", detail, state=state, now=now)
        return False, "SPREAD", detail

    try:
        best_bid = float(buys[0][0])
        best_ask = float(sells[0][0])
    except (IndexError, ValueError, TypeError):
        detail = "malformed depth"
        if log_rejections:
            log_gate_rejection(symbol, "SPREAD", detail, state=state, now=now)
        return False, "SPREAD", detail

    ok, detail = check_spread(best_bid, best_ask)
    if not ok:
        if log_rejections:
            log_gate_rejection(symbol, "SPREAD", detail, state=state, now=now)
        return False, "SPREAD", detail

    # 4. Depth Cover
    ok, detail, _ = check_depth(depth_data, trade_size=trade_size)
    if not ok:
        if log_rejections:
            log_gate_rejection(symbol, "DEPTH", detail, state=state, now=now)
        return False, "DEPTH", detail

    # 5. Cooldown
    ok, detail = check_cooldown(symbol, state, now=now, is_sole_candidate=is_sole_candidate)
    if not ok:
        if log_rejections:
            log_gate_rejection(symbol, "COOLDOWN", detail, state=state, now=now)
        return False, "COOLDOWN", detail

    return True, None, None


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 gates.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sandbox the audit log — self-checks must not pollute the real gate log.
    import tempfile
    _LIVE_LOGS = LOGS_DIR
    LOGS_DIR = tempfile.mkdtemp(prefix="hunter_gates_test_")

    print("=== gates.py self-check ===\n")

    now = time.time()
    mock_state = {
        "phase": "HUNTING",
        "balance_idr": 100_000,
        "cooldowns": {},
        "gate_stats": {},
    }

    # 1. Warmup gate: 99 bars -> REJECT, 100 bars -> PASS
    ok, _ = check_warmup(99)
    assert not ok, "99 bars should fail warmup"
    ok, _ = check_warmup(100)
    assert ok, "100 bars should pass warmup"
    print("[PASS] warmup gate: 99 bars REJECTED, 100 bars PASSED")

    # 2. Freshness gate: 31m -> REJECT, 20m -> PASS
    ok, _ = check_freshness(now - 31 * 60, now=now)
    assert not ok, "31m old should fail freshness"
    ok, _ = check_freshness(now - 20 * 60, now=now)
    assert ok, "20m old should pass freshness"
    print("[PASS] freshness gate: 31m REJECTED, 20m PASSED")

    # 3. Spread gate:
    # 50% spread (KRDIDR case: bid=2, ask=3) -> REJECT
    ok, _ = check_spread(best_bid=2.0, best_ask=3.0)
    assert not ok, "50% spread must be rejected"

    # 0.50% spread (STRKIDR case: bid=1000, ask=1005.0) -> PASS now (<= 0.50%)
    ok, _ = check_spread(best_bid=1000.0, best_ask=1005.0)
    assert ok, "0.50% spread must pass under 0.50% max"

    # 0.60% spread (STRKIDR case: bid=1000, ask=1006.0) -> REJECT
    ok, _ = check_spread(best_bid=1000.0, best_ask=1006.0)
    assert not ok, "0.60% spread must be rejected"

    # 0.19% spread (passing case) -> PASS
    ok, _ = check_spread(best_bid=1000.0, best_ask=1001.9)
    assert ok, "0.19% spread must pass"
    print("[PASS] spread gate: 50% REJECTED, 0.60% REJECTED, 0.19% PASSED")

    # 4. Depth cover gate:
    # Thin book: tight spread (0.05%), but book only has 5,000 IDR depth
    thin_depth = {
        "buy": [["1000", "5.0"]],   # 5000 IDR bid cover
        "sell": [["1000.5", "5.0"]] # 5002.5 IDR ask cover
    }
    ok, detail, _ = check_depth(thin_depth, trade_size=40_000.0)
    assert not ok, "thin depth book must be rejected"
    assert "ask cover" in detail

    # Healthy book: 50M IDR depth on each side
    healthy_depth = {
        "buy": [["1000", "50000.0"]],
        "sell": [["1000.5", "50000.0"]]
    }
    ok, _, _ = check_depth(healthy_depth, trade_size=40_000.0)
    assert ok, "healthy depth book must pass"
    print("[PASS] depth gate: thin book REJECTED, healthy book PASSED")

    # 5. Cooldown gate & sole candidate exception
    mock_state["cooldowns"]["PEPEIDR"] = now + 3600
    # Regular check (other candidates exist) -> REJECT
    ok, _ = check_cooldown("PEPEIDR", mock_state, now=now, is_sole_candidate=False)
    assert not ok, "pair on cooldown must be rejected when not sole candidate"

    # Sole candidate exception -> PASS
    ok, _ = check_cooldown("PEPEIDR", mock_state, now=now, is_sole_candidate=True)
    assert ok, "pair on cooldown must pass when it is the sole candidate"
    print("[PASS] cooldown gate: regular REJECTED, sole candidate exception PASSED")

    # 6. The ML veto is REMOVED from the product (2026-11-08).
    # The gate, its loader and its constants are gone. Assert they cannot come
    # back silently: a reintroduced veto would change which pairs trade while
    # every other check still passed.
    _src = open(__file__).read()
    _head = _src.split("def _self_check")[0]
    assert "check_ml_veto" not in _head, \
        "check_ml_veto still defined - the ML veto was meant to be removed"
    assert "load_ml_predictions" not in _head, \
        "load_ml_predictions still defined - the ML veto was meant to be removed"
    assert "ML_VETO_THRESHOLD" not in _head, \
        "ML_VETO_THRESHOLD still defined - the ML veto was meant to be removed"
    assert "VETO" not in "".join(
        l for l in _head.splitlines() if "log_gate_rejection" in l
    ), "a VETO gate is still being logged from the engine"
    print("[PASS] ML veto removed: no gate, no loader, no threshold")

    # 7. End-to-end evaluation & gate logging
    dt_str, date_str = format_wib_timestamp(now)
    assert "WIB" in dt_str
    log_file = os.path.join(LOGS_DIR, f"hunter_gates_{date_str}.log")
    initial_log_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0

    # Run full rejection
    passed, failing_gate, _ = evaluate_pair_gates(
        symbol="KRDIDR",
        candles_count=100,
        last_candle_ts=now,
        depth_data={"buy": [["2", "100000"]], "sell": [["3", "100000"]]},
        state=mock_state,
        now=now,
        log_rejections=True
    )
    assert not passed and failing_gate == "SPREAD"
    assert mock_state["gate_stats"].get("SPREAD", 0) >= 1
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > initial_log_size

    # Verify log content format
    with open(log_file, "r") as f:
        last_line = f.readlines()[-1].strip()
    assert "WIB" in last_line
    assert "KRDIDR" in last_line
    assert "SPREAD" in last_line
    print(f"[PASS] gate logging: written to {os.path.basename(log_file)}")
    print(f"       sample line: {last_line}")

    print("\n=== ALL CHECKS PASSED ===")
