#!/usr/bin/env python3
"""
run_hunter.py — The Hunter main tick engine (1 pass per minute).

Step 7 of development_plan.txt.

Cycle per tick:
  1. Acquire flock (exclusive non-blocking lock)
  2. Check kill switch (halts immediately if data/hunter_stop exists)
  3. Check control file (data/hunter_control.json):
     - STOPPED  -> return immediately, no scanning, no state write
     - STOPPING -> liquidate/cancel (perform_stop) then park
     - RUNNING  -> continue
  4. Load state from data/hunter_state.json
  5. Read the active framework ONCE (control file) and route entry/exit
     through it. A position uses the framework it was opened under.
  6. Phase dispatch:
     - HUNTING:    screen universe -> gates -> framework entry -> tiebreaker -> buy order -> ORDER_OPEN
     - ORDER_OPEN: check fill (price touch) -> HOLDING, or cancel on timeout (6 checks) -> HUNTING
     - HOLDING:    framework exit evaluation -> trade_log -> cooldown -> HUNTING
  7. Save state atomically (tmp + rename)
  8. Release flock

Usage:
  python3 scripts/run_hunter.py              one tick
  python3 scripts/run_hunter.py --test       self-checks
  python3 scripts/run_hunter.py --stop-now   liquidate + park (used by dashboard)
"""

import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import (
    HaltSignal,
    acquire_lock,
    check_kill_switch,
    clear_expired_cooldowns,
    is_on_cooldown,
    load_state,
    release_lock,
    save_state,
    set_cooldown,
)
from control import (
    MODE_RUNNING,
    MODE_STOPPED,
    MODE_STOPPING,
    active_framework,
    load_control,
    save_control,
)
from frameworks import get_framework
from indicators import range_position, volume_ratio
from filters import evaluate_filters
from gates import (
    DEFAULT_TRADE_SIZE,
    evaluate_pair_gates,
    format_wib_timestamp,
    log_gate_rejection,
)
from exit_strategy import (
    EXIT_BE_RATCHET,
    EXIT_HARD_SL,
    EXIT_MANUAL_MARKET,
    EXIT_TARGET,
    EXIT_USER_STOP,
    STOP_PCT,
    TARGET_PCT,
    evaluate_exit,
    initialize_exit_levels,
)
from hunt_log import next_hunt_id, record_hunt_event, record_hunt_tick
from executor_paper import (
    check_paper_buy_fill,
    compute_coin_amount,
    compute_maker_limit_price,
    create_paper_buy_order,
    execute_paper_sell,
    validate_order_sizing,
)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
# Paths from paths.py - see that module for why this is centralised.
try:
    from paths import DATA_DIR
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import DATA_DIR
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.json")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv")

BASE_URL = "https://indodax.com"

# Tunables now come from config.yaml (trading.*), falling back to these values
# when no config file is present. Read once at import: a running position must
# keep the parameters it was opened with.
import config as _config  # noqa: E402
_C = _config.current()

MIN_TRADE_BALANCE_IDR = float(_config.get(_C, "trading.min_trade_balance_idr", 10_000.0))
SIZING_FRACTION = float(_config.get(_C, "trading.sizing_fraction", 0.40))
ORDER_TIMEOUT_CHECKS = int(_config.get(_C, "trading.order_timeout_checks", 6))
STOP_SLIP_FRACTION = float(_config.get(_C, "trading.stop_slip_fraction", 0.0094))
COOLDOWN_HOURS = float(_config.get(_C, "trading.cooldown_hours", 4.0))


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def read_candles(symbol: str, n: int = 100) -> Tuple[Optional[List[dict]], int]:
    """Read last n candles and total count from data/ohlcv/<SYMBOL>.csv."""
    csv_path = os.path.join(OHLCV_DIR, f"{symbol.upper()}.csv")
    if not os.path.exists(csv_path):
        return None, 0
    try:
        with open(csv_path, "r") as fh:
            lines = fh.readlines()
        data = [ln for ln in lines if ln.strip() and not ln.startswith("timestamp")]
        rows = []
        for ln in data[-n:]:
            p = ln.strip().split(",")
            rows.append({
                "t": int(p[0]),
                "o": float(p[1]),
                "h": float(p[2]),
                "l": float(p[3]),
                "c": float(p[4]),
                "v": float(p[5]),
            })
        return rows, len(data)
    except Exception:
        return None, 0


def load_depth(pair_id: str) -> dict:
    """Load depth data from data/depth/<pair_id>.json."""
    depth_path = os.path.join(DEPTH_DIR, f"{pair_id.lower()}.json")
    if not os.path.exists(depth_path):
        return {}
    try:
        with open(depth_path, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def fetch_live_ticker(ticker_id: str, depth_fallback: Optional[dict] = None) -> dict:
    """
    Fetch live ticker from /api/ticker/<ticker_id>.
    Falls back to depth file or best estimates if offline/timeout.
    Returns {"last": float, "buy": float, "sell": float}.
    """
    try:
        resp = requests.get(f"{BASE_URL}/api/ticker/{ticker_id}", timeout=4)
        if resp.status_code == 200:
            data = resp.json().get("ticker", {})
            return {
                "last": float(data.get("last", 0)),
                "buy": float(data.get("buy", 0)),
                "sell": float(data.get("sell", 0)),
            }
    except Exception:
        pass

    # Fallback to local depth
    if depth_fallback:
        buys = depth_fallback.get("buy", [])
        sells = depth_fallback.get("sell", [])
        if buys and sells:
            b = float(buys[0][0])
            s = float(sells[0][0])
            return {"last": b, "buy": b, "sell": s}

    return {"last": 0.0, "buy": 0.0, "sell": 0.0}


# ---------------------------------------------------------------------------
# Phase 1: HUNTING
# ---------------------------------------------------------------------------
def handle_hunting(state: dict, now: float, fw=None) -> str:
    """
    Screen universe -> gates -> framework entry -> winner tiebreaker -> place buy order.
    Returns a one-line summary.
    """
    if fw is None:
        fw = get_framework(active_framework())

    balance = state.get("balance_idr", 100_000.0)
    trade_size = balance * SIZING_FRACTION

    if trade_size < MIN_TRADE_BALANCE_IDR:
        log_gate_rejection("GLOBAL", "SIZING", f"trade size {trade_size:.0f} < 10k IDR min", state=state, now=now)
        return f"[HUNTING] Balance too low: trade_size {trade_size:.0f} IDR < 10,000 IDR minimum."

    if not os.path.exists(PAIRS_FILE):
        return "[HUNTING] pairs.json not found."

    with open(PAIRS_FILE, "r") as fh:
        pairs = json.load(fh)[:20]

    if not pairs:
        return "[HUNTING] No pairs in pairs.json."

    # 1. First pass: find candidates passing Gates 1-4 to evaluate sole-candidate cooldown rule
    gate_data = []
    pre_cooldown_survivors = []

    for p in pairs:
        sym = p["symbol"]
        pair_id = p.get("id", sym.lower())
        depth = load_depth(pair_id)
        candles, total_count = read_candles(sym, n=100)
        last_candle_ts = candles[-1]["t"] if candles else 0.0

        gate_data.append({
            "pair": p,
            "depth": depth,
            "candles": candles,
            "total_count": total_count,
            "last_ts": last_candle_ts,
        })

        # Check Gates 1-4 (exclude cooldown)
        passed_pre, _, _ = evaluate_pair_gates(
            symbol=sym,
            candles_count=total_count,
            last_candle_ts=last_candle_ts,
            depth_data=depth,
            state=state,
            trade_size=trade_size,
            is_sole_candidate=True,  # ignore cooldown in pre-pass
            now=now,
            log_rejections=False,    # log only in final pass
        )
        if passed_pre:
            pre_cooldown_survivors.append(sym)

    is_sole_candidate = len(pre_cooldown_survivors) == 1

    # 2. Final pass: run all gates with logging and cooldown exception
    gate_survivors = []
    for item in gate_data:
        p = item["pair"]
        sym = p["symbol"]
        passed, failing_gate, detail = evaluate_pair_gates(
            symbol=sym,
            candles_count=item["total_count"],
            last_candle_ts=item["last_ts"],
            depth_data=item["depth"],
            state=state,
            trade_size=trade_size,
            is_sole_candidate=(is_sole_candidate and sym in pre_cooldown_survivors),
            now=now,
            log_rejections=True,
        )
        if passed:
            gate_survivors.append(item)

    if not gate_survivors:
        return f"[HUNTING] 0 of {len(pairs)} pairs passed gates. Stand by."

    # 3. Signals / Filters evaluation (Step 3)
    scored_candidates = []
    for item in gate_survivors:
        p = item["pair"]
        candles = item["candles"]
        eval_res = fw.evaluate_entry(candles, item["depth"])

        if eval_res["eligible"]:
            scored_candidates.append({
                "pair": p,
                "depth": item["depth"],
                "candles": candles,
                "score": eval_res["total"],
                "direction": eval_res["direction"],
                "vol_idr": float(p.get("vol_idr", 0.0)),
            })

    if not scored_candidates:
        return (f"[HUNTING] {len(gate_survivors)} passed gates, but 0 passed "
                f"framework {fw.name} entry.")

    # 4. Tiebreaker: Highest 24h Volume (vol_idr)
    scored_candidates.sort(key=lambda x: x["vol_idr"], reverse=True)

    # 5. Sizing check & Order Placement
    for cand in scored_candidates:
        p = cand["pair"]
        sym = p["symbol"]
        depth = cand["depth"]
        buys = depth.get("buy", [])
        sells = depth.get("sell", [])
        best_bid = float(buys[0][0])
        best_ask = float(sells[0][0])
        tick = float(p.get("tick", 1.0))
        qty_incr = float(p.get("quantity_increment", 1e-8))

        limit_price = compute_maker_limit_price(best_bid, best_ask, tick)
        coin_amount = compute_coin_amount(trade_size, limit_price, qty_incr)

        ok_sizing, sizing_msg = validate_order_sizing(trade_size, coin_amount, p)
        if not ok_sizing:
            log_gate_rejection(sym, "SIZING", sizing_msg, state=state, now=now)
            continue

        # WINNER ACQUIRED! Allocate a hunt id and place the buy order
        hunt_id = next_hunt_id(state)
        order = create_paper_buy_order(
            symbol=sym,
            limit_price=limit_price,
            coin_amount=coin_amount,
            trade_size_idr=trade_size,
            pair_info=p,
            now=now,
            hunt_id=hunt_id,
            entry_bid=best_bid,
            entry_ask=best_ask,
            qty_increment=qty_incr,
        )
        state["order"] = order
        order["framework"] = fw.name
        order["framework_version"] = getattr(fw, "version", "?")
        state["phase"] = "ORDER_OPEN"
        return (f"[ORDER_PLACED] {hunt_id} [{fw.name}] Winner: {sym} "
                f"(Vol: {cand['vol_idr']/1e9:.1f}B, Score: {cand['score']:.1f}) | "
                f"Limit Buy: {coin_amount} @ {limit_price} ({trade_size:,.0f} IDR)")

    return "[HUNTING] All filter candidates failed order sizing minimums."


# ---------------------------------------------------------------------------
# Phase 2: ORDER_OPEN
# ---------------------------------------------------------------------------
def handle_order_open(state: dict, now: float) -> str:
    """
    Check resting limit order fill.
    Fills if price touches limit_price -> HOLDING.
    Cancels if timeout after 6 checks -> HUNTING.
    """
    order = state.get("order")
    if not order:
        state["phase"] = "HUNTING"
        return "[ORDER_OPEN] No active order found in state. Reset to HUNTING."

    sym = order["symbol"]
    hunt_id = order.get("hunt_id")
    pair_id = order.get("pair_id", sym.lower())
    depth = load_depth(pair_id)
    ticker = fetch_live_ticker(order.get("ticker_id", pair_id), depth_fallback=depth)
    live_price = ticker["last"] if ticker["last"] > 0 else order["limit_price"]
    best_bid = ticker.get("buy", 0) or live_price
    best_ask = ticker.get("sell", 0) or live_price

    # Exit levels are computed by the framework that placed this order.
    order_fw = get_framework(order.get("framework") or active_framework())
    fill_levels = order_fw.initialize_exit(order["limit_price"], order["tick"])

    filled, position = check_paper_buy_fill(
        order,
        live_price=live_price,
        now=now,
        bid=best_bid,
        ask=best_ask,
        qty_increment=float(order.get("qty_increment", 0.0) or 0.0),
        exit_levels=fill_levels,
    )
    if filled and position:
        state["position"] = position
        state["order"] = None
        state["phase"] = "HOLDING"
        # Deduct cost basis from paper balance
        state["balance_idr"] = round(state.get("balance_idr", 100_000.0) - position["total_cost_idr"], 2)
        lv = position["exit_levels"]
        record_hunt_event(hunt_id, "ENTRY_FILLED", {
            "symbol": sym,
            "entry_price": position["entry_price"],
            "coins": position["coins"],
            "cost_idr": position["total_cost_idr"],
            "target_price": lv["target_price"],
            "stop_trigger_price": lv["stop_trigger_price"],
            "bid": position.get("entry_bid"),
            "ask": position.get("entry_ask"),
        }, now=now)
        return (f"[FILL_COMPLETE] {hunt_id} [{position.get('framework','A')}] {sym} "
                f"filled! {position['coins']} @ {position['entry_price']} | "
                f"Cost: {position['total_cost_idr']:,.0f} IDR (Fee: {position['entry_fee_idr']:,.0f}) | "
                f"Target: {lv['target_price']:,.0f} (+{lv['target_pct']*100:.1f}%) "
                f"Stop: {lv['stop_trigger_price']:,.0f} -> HOLDING")

    # Check timeout
    checks = order.get("checks_elapsed", 1)
    if checks >= ORDER_TIMEOUT_CHECKS:
        state["order"] = None
        state["phase"] = "HUNTING"
        record_hunt_event(hunt_id, "ORDER_TIMEOUT", {
            "symbol": sym, "checks": checks, "limit_price": order["limit_price"],
        }, now=now)
        return f"[ORDER_TIMEOUT] {hunt_id} {sym} unfilled after {checks} checks. Cancelled -> HUNTING."

    return (f"[ORDER_PENDING] {hunt_id} {sym} limit {order['limit_price']} | "
            f"Live: {live_price} | Checks: {checks}/{ORDER_TIMEOUT_CHECKS}")


# ---------------------------------------------------------------------------
# Phase 3: HOLDING
# ---------------------------------------------------------------------------
def handle_holding(state: dict, now: float) -> str:
    """
    Track an open position against the fixed-target exit rule.

    Every tick:
      1. Read live last / best bid / best ask
      2. Log a tick row (bid + ask) to data/hunt_ticks/<hunt_id>.jsonl
      3. Evaluate:  stop crossed?  -> cancel limit, MARKET SELL
                    target hit?     -> the resting LIMIT SELL fills
                    otherwise       -> HOLD
    """
    pos = state.get("position")
    if not pos:
        state["phase"] = "HUNTING"
        return "[HOLDING] No active position in state. Reset to HUNTING."

    sym = pos["symbol"]
    hunt_id = pos.get("hunt_id") or "H-UNKNOWN"
    pair_id = pos.get("pair_id", sym.lower())
    depth = load_depth(pair_id)
    ticker = fetch_live_ticker(pair_id, depth_fallback=depth)
    live_price = ticker["last"] if ticker["last"] > 0 else pos["entry_price"]
    best_bid = ticker.get("buy", 0) or live_price
    best_ask = ticker.get("sell", 0) or live_price

    levels = pos["exit_levels"]
    target = levels["target_price"]
    stop = levels["stop_trigger_price"]

    # The position is pinned to the framework it was opened under. A later
    # framework switch must never re-rule an open trade.
    pos_fw = get_framework(pos.get("framework") or "A")

    pct_from_entry = (live_price - pos["entry_price"]) / pos["entry_price"] * 100.0

    eval_res = pos_fw.evaluate_exit(live_price, levels)
    action = eval_res["action"]

    # --- tick capture: bid/ask minute by minute while holding ---------------
    record_hunt_tick(hunt_id, {
        "symbol": sym,
        "last": live_price,
        "bid": best_bid,
        "ask": best_ask,
        "target": target,
        "stop": levels["stop_trigger_price"],
        "pct": round(pct_from_entry, 4),
        "ratchet": levels.get("ratchet_armed", False),
        "entry_price": pos["entry_price"],
    }, now=now)

    if action in (EXIT_TARGET, EXIT_HARD_SL, EXIT_BE_RATCHET):
        if action == EXIT_TARGET:
            # The resting LIMIT SELL fills at its own price — that is the point
            # of a limit order. It is NOT the market price at the moment the
            # bot notices.
            exit_price = target
            fee_pct = pos.get("maker_fee_pct", 0.1)      # resting limit = maker
            reason = EXIT_TARGET
        elif action == EXIT_BE_RATCHET:
            # Breakeven ratchet: once price climbed +1.2%, stop was raised to +0.3%
            # fills at or above the ratchet floor
            r_floor = levels.get("ratchet_stop_price", pos["entry_price"] * 1.003)
            exit_price = max(best_bid, r_floor)
            fee_pct = None                                # taker fee
            reason = EXIT_BE_RATCHET
        else:
            # Market sell: the position was TRIGGERED at its own stop level and
            # the fill lands lower. Price off the stored trigger, never off the
            # current STOP_PCT constant.
            #
            # This used to read  entry * (1 - STOP_PCT - SLIP)  which silently
            # re-derived the stop from whatever framework was active NOW. A
            # position opened under A (-4%) that lived through a switch to C
            # (-10%) recorded its loss as -10%, and one opened under C and
            # stopped after a switch to A recorded -4%. The trigger level is a
            # property of the POSITION - it was written into exit_levels at
            # entry and the ratchet may have moved it since. In live mode the
            # same mistake would place the sell at the wrong price with real
            # money on it.
            trigger = float(levels.get("stop_trigger_price")
                            or pos["entry_price"] * (1.0 - float(levels.get("stop_pct", STOP_PCT))))
            exit_price = trigger * (1.0 - STOP_SLIP_FRACTION)
            fee_pct = None                                # use position taker fee
            reason = EXIT_HARD_SL

        trade_rec, delta = execute_paper_sell(
            pos,
            exit_price=exit_price,
            exit_reason=reason,
            now=now,
            fee_override_pct=fee_pct,
            exit_bid=best_bid,
            exit_ask=best_ask,
        )

        # --- config stamping ------------------------------------------------
        # A trade record must be self-describing: which strategy opened it,
        # and with what target/stop. Without this, PnL from different
        # frameworks (or later parameter changes) becomes one unlabelable pool.
        trade_rec["framework"] = pos.get("framework", "A")
        trade_rec["framework_version"] = pos.get("framework_version", "?")
        trade_rec["target_pct"] = levels.get("target_pct")
        trade_rec["stop_pct"] = levels.get("stop_pct")
        trade_rec["ratchet_armed"] = levels.get("ratchet_armed", False)

        state["balance_idr"] = round(state.get("balance_idr", 0.0) + delta["net_proceeds"], 2)
        set_cooldown(state, sym, hours=COOLDOWN_HOURS, now=now)
        if "trade_log" not in state:
            state["trade_log"] = []
        state["trade_log"].append(trade_rec)
        state["position"] = None
        state["phase"] = "HUNTING"

        record_hunt_event(pos["hunt_id"], reason, {
            "symbol": sym,
            "entry_price": trade_rec["entry_price"],
            "exit_price": trade_rec["exit_price"],
            "net_pnl_idr": trade_rec["net_pnl_idr"],
            "net_pnl_pct": trade_rec["net_pnl_pct"],
        }, now=now)

        pnl_idr = trade_rec["net_pnl_idr"]
        pnl_pct = trade_rec["net_pnl_pct"]
        sign = "+" if pnl_idr >= 0 else ""
        return (f"[EXIT_COMPLETE] {hunt_id} {sym} sold @ {exit_price:,.0f} ({reason}) | "
                f"Net PnL: {sign}{pnl_idr:,.0f} IDR ({sign}{pnl_pct:.2f}%) | "
                f"New Balance: {state['balance_idr']:,.0f} IDR | Cooldown: 4h")

    return (f"[HOLDING] {hunt_id} {sym} at {live_price:,.0f} ({pct_from_entry:+.2f}%) | "
            f"Target: {target:,.0f} | Stop: {stop:,.0f}")


# ---------------------------------------------------------------------------
# Main Tick Loop
# ---------------------------------------------------------------------------
def perform_stop(state: dict, now: float, ctrl: dict) -> str:
    """
    Liquidate whatever is open, then park.

    Implements Reza's stop rules (2026-11-03):
      HOLDING            -> market-sell the full quantity
      ORDER_OPEN, unfilled-> cancel the whole order (no position exists)
      ORDER_OPEN, partial -> market-sell the filled qty, cancel the remainder
      HUNTING            -> nothing to do

    NOTE: the paper executor fills all-or-nothing (executor_paper.py:150), so
    the partial case is unreachable in paper mode. The branch is implemented
    anyway so the live executor inherits the contract. See LIVE MODE NOTES.

    Never raises: a stop must always end with the engine parked.
    Returns a human-readable summary line.
    """
    phase = state.get("phase")
    messages = []

    if phase == "HOLDING" and state.get("position"):
        pos = state["position"]
        pair_id = pos.get("pair_id", pos["symbol"].lower())
        depth = load_depth(pair_id)
        ticker = fetch_live_ticker(pair_id, depth_fallback=depth)
        best_bid = ticker.get("buy") or ticker.get("last") or pos["entry_price"]
        best_ask = ticker.get("sell") or best_bid

        trade_rec, delta = execute_paper_sell(
            pos,
            exit_price=best_bid,
            exit_reason=EXIT_USER_STOP,
            now=now,
            fee_override_pct=pos.get("taker_fee_pct", 0.2),
            exit_bid=best_bid,
            exit_ask=best_ask,
        )
        # --- config stamping (same contract as handle_holding) ---------------
        trade_rec["framework"] = pos.get("framework", "A")
        trade_rec["framework_version"] = pos.get("framework_version", "?")
        trade_rec["target_pct"] = pos["exit_levels"].get("target_pct")
        trade_rec["stop_pct"] = pos["exit_levels"].get("stop_pct")
        trade_rec["ratchet_armed"] = pos["exit_levels"].get("ratchet_armed", False)

        state["balance_idr"] = round(
            state.get("balance_idr", 0.0) + delta["net_proceeds"], 2)
        state.setdefault("trade_log", []).append(trade_rec)
        set_cooldown(state, pos["symbol"], hours=COOLDOWN_HOURS, now=now)
        state["position"] = None
        state["phase"] = "HUNTING"

        record_hunt_event(pos.get("hunt_id"), EXIT_USER_STOP, {
            "symbol": pos["symbol"],
            "entry_price": pos["entry_price"],
            "exit_price": best_bid,
            "net_pnl_idr": trade_rec.get("net_pnl_idr"),
            "net_pnl_pct": trade_rec.get("net_pnl_pct"),
            "reason": "user pressed STOP",
        }, now=now)
        messages.append(
            f"[STOP] market-sold {pos.get('hunt_id')} {pos['symbol']} "
            f"@ {best_bid:,.4f} ({trade_rec.get('net_pnl_pct', 0):+.2f}%)")

    elif phase == "ORDER_OPEN" and state.get("order"):
        od = state["order"]
        filled = float(od.get("filled_coin_amount") or 0.0)
        total = float(od.get("coin_amount") or 0.0)

        # --- partial fill branch (unreachable in paper; live-mode contract) --
        if filled > 0 and total > 0 and filled < total:
            messages.append(
                f"[STOP] partial fill {filled}/{total} on {od.get('symbol')} "
                f"— live mode sells the filled part then cancels the rest")
            # Paper executor never produces this state; if it somehow does,
            # fail safe by cancelling rather than inventing a position.
        else:
            messages.append(
                f"[STOP] cancelled resting buy {od.get('hunt_id')} {od.get('symbol')}")

        record_hunt_event(od.get("hunt_id"), "ORDER_CANCELLED", {
            "symbol": od.get("symbol"),
            "reason": "user pressed STOP",
            "filled_coin_amount": filled,
        }, now=now)
        state["order"] = None
        state["phase"] = "HUNTING"

    else:
        messages.append("[STOP] nothing open — parked.")

    ctrl["mode"] = MODE_STOPPED
    ctrl["changed_at"] = now
    ctrl["changed_by"] = "stop"
    save_control(ctrl)
    return " ".join(messages)


def tick(now: Optional[float] = None) -> dict:
    """Execute one pass of The Hunter."""
    if now is None:
        now = time.time()

    dt_str, _ = format_wib_timestamp(now)
    lock_fd = acquire_lock()
    if lock_fd is None:
        return {"status": "SKIPPED_LOCK", "message": f"{dt_str}  [SKIP] Another tick process holds the lock."}

    try:
        check_kill_switch()

        # --- control plane -------------------------------------------------
        # Checked BEFORE load_state()/phase dispatch: a stopped engine must do
        # no scanning, place no orders, and write no state.
        ctrl = load_control()
        if ctrl["mode"] == MODE_STOPPED:
            return {"status": "STOPPED",
                    "message": f"{dt_str}  [STOPPED] Hunting stopped by user."}

        # --- executor guard -------------------------------------------------
        # The dashboard can arm LIVE, but this engine has no live execution path
        # in this build (executor_live is used by the dashboard's own actions).
        # Refusing loudly is the only honest option: silently paper-trading while
        # the UI says LIVE would mean the operator watches simulated fills and
        # believes real money is at work - or worse, the reverse.
        if ctrl.get("exec") == "LIVE":
            # Record the refusal so the dashboard can say BLOCKED instead of
            # painting a green HUNTING badge while nothing happens. The message
            # is the same one the operator needs to read.
            try:
                from control import set_blocked
                set_blocked("LIVE is armed but this engine cannot place live "
                            "orders in this build")
            except Exception:                                # noqa: BLE001
                pass
            return {"status": "EXECUTOR_UNAVAILABLE",
                    "message": (f"{dt_str}  [BLOCKED] LIVE is armed but this engine "
                                f"cannot place live orders in this build. Set the "
                                f"executor back to PAPER, or wait for the live "
                                f"engine release.")}

        # past the guards: this tick really runs, so any previous block is stale
        try:
            from control import clear_blocked
            clear_blocked()
        except Exception:                                    # noqa: BLE001
            pass

        state = load_state()

        if ctrl["mode"] == MODE_STOPPING:
            summary = perform_stop(state, now, ctrl)
            save_state(state)
            return {"status": "OK", "phase": state.get("phase"),
                    "summary": summary, "timestamp": dt_str,
                    "balance": state.get("balance_idr")}

        clear_expired_cooldowns(state, now)

        # Read the framework ONCE at tick start. A switch mid-tick cannot
        # change the rules halfway through a decision.
        fw = get_framework(ctrl.get("framework") or active_framework())

        phase = state.get("phase", "HUNTING")

        if phase == "HUNTING":
            summary = handle_hunting(state, now, fw=fw)
        elif phase == "ORDER_OPEN":
            summary = handle_order_open(state, now)
        elif phase == "HOLDING":
            summary = handle_holding(state, now)
        else:
            state["phase"] = "HUNTING"
            summary = f"[RESET] Unknown phase {phase} reset to HUNTING."

        save_state(state)
        return {
            "status": "OK",
            "phase": state.get("phase"),
            "summary": summary,
            "timestamp": dt_str,
            "balance": state.get("balance_idr"),
        }

    except HaltSignal as e:
        return {"status": "HALTED", "message": f"{dt_str}  [HALT] {e}"}
    finally:
        release_lock(lock_fd)


def run_self_checks():
    """Simulate full end-to-end lifecycle and crash rules.

    Sandboxed: hunt_log's tick/event writers are redirected to a temp dir so
    self-check rows never land in the live data/hunt_ticks or hunt_events.jsonl
    (same class of bug as the old state.py self-check that wiped live state).
    """
    import tempfile
    import shutil
    import hunt_log

    print("=== run_hunter.py self-check ===\n")
    now = time.time()

    _live_ticks, _live_events = hunt_log.TICKS_DIR, hunt_log.EVENTS_FILE
    _sandbox = tempfile.mkdtemp(prefix="hunter_engine_test_")
    hunt_log.TICKS_DIR = os.path.join(_sandbox, "hunt_ticks")
    hunt_log.EVENTS_FILE = os.path.join(_sandbox, "events.jsonl")

    # 1. Kill switch
    kill_file = os.path.join(DATA_DIR, "hunter_stop")
    open(kill_file, "w").close()
    res_halt = tick(now=now)
    assert res_halt["status"] == "HALTED"
    os.remove(kill_file)
    print("[PASS] kill switch halts immediately")

    # 2. Sizing guard: balance < 25,000 IDR (40% < 10,000 IDR min) -> rejects sizing
    mock_state = {
        "phase": "HUNTING",
        "balance_idr": 20_000.0,
        "order": None,
        "position": None,
        "cooldowns": {},
        "gate_stats": {},
        "trade_log": [],
        "hunt_counter": 0,
    }
    s_size = handle_hunting(mock_state, now=now)
    assert "Balance too low" in s_size
    print("[PASS] sizing guard: balance 20k IDR (40% = 8k < 10k min) rejected")

    # 3. Full Cycle: HUNTING -> ORDER_OPEN -> HOLDING -> EXIT_TARGET -> HUNTING
    mock_state["balance_idr"] = 100_000.0
    p_info = {
        "id": "xrpidr",
        "symbol": "XRPIDR",
        "tick": 1.0,
        "quantity_increment": 1e-4,
        "trade_min_base_currency": 10000.0,
        "trade_fee_percent_maker": 0.15,
        "trade_fee_percent_taker": 0.30,
    }

    # NOTE: pin the framework explicitly. handle_order_open() falls back to
    # the LIVE control file when the order carries no framework, so a user
    # switching the running bot to framework B used to break this self-check
    # (and would have exercised B's levels under an "A" test).
    order = create_paper_buy_order(
        "XRPIDR", 24631.0, 1.6237, 40000.0, p_info,
        now=now, hunt_id="H-0001", entry_bid=24630.0, entry_ask=24632.0,
        qty_increment=1e-4,
    )
    order["tick"] = 1.0
    order["framework"] = "A"
    order["framework_version"] = "3.0"
    mock_state["order"] = order
    mock_state["phase"] = "ORDER_OPEN"

    # Step B: Order fill
    s_fill = handle_order_open(mock_state, now=now)
    assert mock_state["phase"] == "HOLDING"
    assert mock_state["position"] is not None
    assert mock_state["order"] is None
    assert mock_state["position"]["hunt_id"] == "H-0001"
    print(f"[PASS] ORDER_OPEN -> HOLDING: {s_fill}")

    # Step C: Exit levels are the fixed target / stop
    lv = mock_state["position"]["exit_levels"]
    assert lv["target_pct"] == 0.02
    assert lv["stop_pct"] == 0.04
    assert lv["target_price"] == 25123.0, lv["target_price"]          # 24631*1.02=25123.62 floored
    assert lv["stop_trigger_price"] == 23645.0, lv["stop_trigger_price"]  # 24631*0.96=23645.76
    print(f"[PASS] fixed-target levels: target={lv['target_price']} (+2%), stop={lv['stop_trigger_price']} (-4%)")

    # Step D: HOLD well inside the corridor -> no exit
    assert evaluate_exit(24500.0, lv)["action"] == "HOLD"
    print("[PASS] HOLDING: price below target -> HOLD")

    # Step E: Target hit -> the resting limit sells at its own price
    res_t = evaluate_exit(lv["target_price"], lv)
    assert res_t["action"] == EXIT_TARGET
    trade_rec, delta = execute_paper_sell(
        mock_state["position"], exit_price=lv["target_price"],
        exit_reason=EXIT_TARGET, now=now,
        fee_override_pct=mock_state["position"]["maker_fee_pct"],
    )
    mock_state["balance_idr"] = round(mock_state["balance_idr"] + delta["net_proceeds"], 2)
    set_cooldown(mock_state, "XRPIDR", hours=4.0, now=now)
    mock_state["trade_log"].append(trade_rec)
    mock_state["position"] = None
    mock_state["phase"] = "HUNTING"

    assert mock_state["phase"] == "HUNTING"
    assert len(mock_state["trade_log"]) == 1
    assert mock_state["trade_log"][0]["net_pnl_idr"] > 0
    assert mock_state["trade_log"][0]["hunt_id"] == "H-0001"
    assert is_on_cooldown(mock_state, "XRPIDR", now=now)
    print(f"[PASS] HOLDING -> EXIT_TARGET: Net PnL = +{trade_rec['net_pnl_idr']:.2f} IDR (+{trade_rec['net_pnl_pct']:.2f}%)")
    print(f"[PASS] Post-exit state: Phase=HUNTING, Balance={mock_state['balance_idr']:,.0f} IDR, Cooldown=active")

    # Step F: The stop path — market sell, and it does NOT fill at exactly -4%
    lv2 = initialize_exit_levels(1000.0, 1.0)          # target 1020, stop 960
    assert evaluate_exit(959.0, lv2)["action"] == EXIT_HARD_SL
    expected_stop_fill = 1000.0 * (1.0 - STOP_PCT - STOP_SLIP_FRACTION)
    assert abs(expected_stop_fill - 950.6) < 0.01, expected_stop_fill
    print(f"[PASS] STOP path: trigger at {lv2['stop_trigger_price']}, "
          f"fill modelled at {expected_stop_fill:.2f} "
          f"(-{STOP_PCT*100:.0f}% less {STOP_SLIP_FRACTION*100:.2f}% measured slippage)")

    # --- the stop must price off the POSITION, not the active framework ------
    # This drives handle_holding() end to end, with the ticker stubbed, because
    # the bug lived in that function's exit branch - not in execute_paper_sell.
    # A test that calls the executor directly passes with the bug present and is
    # therefore worthless.
    #
    # Scenario: a position opened under C (-10% stop, trigger 900) while the
    # ACTIVE framework is A (-4%). Only the stored trigger gives -10%.
    _orig_ticker = globals()["fetch_live_ticker"]
    try:
        def _stub_ticker(pair_id, depth_fallback=None):
            return {"last": 899.0, "buy": 899.0, "sell": 901.0}

        globals()["fetch_live_ticker"] = _stub_ticker
        _st = {
            "phase": "HOLDING",
            "balance_idr": 100000.0,
            "order": None,
            "cooldowns": {},
            "trade_log": [],
            "position": {
                "hunt_id": "H-STOPX", "symbol": "TESTIDR", "pair_id": "testidr",
                "entry_price": 1000.0, "coins": 10.0, "entry_time": now,
                "fees_paid_idr": 30.0,
                "maker_fee_pct": 0.1, "taker_fee_pct": 0.3,
                "framework": "C", "framework_version": "1",
                "exit_levels": {
                    "target_price": 1050.0, "stop_trigger_price": 900.0,
                    "stop_pct": 0.10, "target_pct": 0.05,
                    "ratchet_stop_price": 997.0, "ratchet_armed": False,
                },
            },
        }
        _msg = handle_holding(_st, now=now)
        _rec = _st["trade_log"][-1]
        # 900 * (1 - 0.0094) = 891.54 ; a -4% reconstruction would give ~954.
        assert abs(_rec["exit_price"] - 900.0 * (1.0 - STOP_SLIP_FRACTION)) < 0.01, \
            f"exit priced off the constant, not the position: {_rec['exit_price']}"
        assert _rec["stop_pct"] == 0.10 and _rec["framework"] == "C", _rec
        assert _rec["net_pnl_pct"] < -10.0, _rec["net_pnl_pct"]
        print(f"[PASS] a C (-10%) position stops on ITS OWN trigger: exit "
              f"{_rec['exit_price']:.2f}, {_rec['net_pnl_pct']:+.2f}% - not -4%")
    finally:
        globals()["fetch_live_ticker"] = _orig_ticker


    # Step G: The Breakeven Ratchet path
    lv_ratchet = initialize_exit_levels(1000.0, 1.0)
    assert evaluate_exit(1012.0, lv_ratchet)["action"] == "HOLD"
    assert lv_ratchet["ratchet_armed"]
    assert lv_ratchet["stop_trigger_price"] == 1003.0
    res_r = evaluate_exit(1003.0, lv_ratchet)
    assert res_r["action"] == EXIT_BE_RATCHET
    print(f"[PASS] RATCHET path: climbed +1.2% -> armed at 1003 (+0.3%) -> fired {res_r['action']}")

    # 4. Timeout check: 6 checks unfilled cancels order
    mock_state["order"] = create_paper_buy_order("BTCIDR", 1000.0, 0.001, 40000.0, p_info, now=now)
    mock_state["order"]["checks_elapsed"] = 5
    mock_state["phase"] = "ORDER_OPEN"
    # 6th check (price way above limit 1000)
    s_timeout = handle_order_open(mock_state, now=now)
    assert mock_state["phase"] == "HUNTING"
    assert mock_state["order"] is None
    assert "ORDER_TIMEOUT" in s_timeout
    print(f"[PASS] ORDER_OPEN timeout: {s_timeout}")

    # 5. perform_stop(): HOLDING -> market sell, then STOPPED
    import control as _control
    _live_control = _control.CONTROL_FILE
    _ctrl_sandbox = tempfile.mkdtemp(prefix="hunter_ctrl_test_")
    _control.CONTROL_FILE = os.path.join(_ctrl_sandbox, "hunter_control.json")

    stop_state = {
        "phase": "HOLDING",
        "balance_idr": 100_000.0,
        "order": None,
        "cooldowns": {},
        "gate_stats": {},
        "trade_log": [],
        "hunt_counter": 1,
        "position": {
            "hunt_id": "H-0001", "symbol": "XRPIDR", "pair_id": "xrpidr",
            "entry_price": 1000.0, "coins": 40.0, "cost_basis_idr": 40000.0,
            "entry_fee_idr": 40.0, "total_cost_idr": 40040.0,
            "maker_fee_pct": 0.1, "taker_fee_pct": 0.2,
            "entry_time": now,
            "exit_levels": initialize_exit_levels(1000.0, 1.0),
            "tick": 1.0, "qty_increment": 1e-4,
            "entry_bid": 1000.0, "entry_ask": 1002.0,
        },
    }
    # NOTE: the position above is synthetic (entry 1000) while perform_stop()
    # prices the exit off the LIVE ticker for XRPIDR, so the printed % is
    # meaningless here. The assertions below are what matter.
    stop_ctrl = {"mode": "STOPPING", "framework": "A", "changed_at": None, "changed_by": None}
    stop_msg = perform_stop(stop_state, now, stop_ctrl)

    assert stop_state["phase"] == "HUNTING", stop_state["phase"]
    assert stop_state["position"] is None
    assert len(stop_state["trade_log"]) == 1, stop_state["trade_log"]
    rec = stop_state["trade_log"][0]
    assert rec["exit_reason"] == "EXIT_USER_STOP", rec["exit_reason"]
    assert rec["hunt_id"] == "H-0001"
    assert stop_state["balance_idr"] > 100_000.0, "proceeds must return to balance"
    assert stop_ctrl["mode"] == "STOPPED", stop_ctrl["mode"]
    assert is_on_cooldown(stop_state, "XRPIDR", now=now)
    print("[PASS] perform_stop HOLDING: sold, parked, cooldown set, "
          "trade logged as EXIT_USER_STOP")

    # 6. perform_stop(): ORDER_OPEN unfilled -> cancel, no trade record
    stop_state2 = {
        "phase": "ORDER_OPEN", "balance_idr": 100_000.0, "cooldowns": {},
        "gate_stats": {}, "trade_log": [], "hunt_counter": 1, "position": None,
        "order": create_paper_buy_order("XRPIDR", 1000.0, 40.0, 40000.0, p_info,
                                        now=now, hunt_id="H-0002"),
    }
    stop_ctrl2 = {"mode": "STOPPING", "framework": "A"}
    msg2 = perform_stop(stop_state2, now, stop_ctrl2)
    assert stop_state2["phase"] == "HUNTING"
    assert stop_state2["order"] is None
    assert stop_state2["trade_log"] == [], "cancelling an unfilled order must not log a trade"
    assert stop_state2["balance_idr"] == 100_000.0, "no money moves on a cancel"
    assert stop_ctrl2["mode"] == "STOPPED"
    print(f"[PASS] perform_stop ORDER_OPEN: {msg2}")

    # 7. perform_stop(): HUNTING -> harmless no-op
    stop_state3 = {"phase": "HUNTING", "balance_idr": 100_000.0, "cooldowns": {},
                   "gate_stats": {}, "trade_log": [], "hunt_counter": 0,
                   "position": None, "order": None}
    stop_ctrl3 = {"mode": "STOPPING", "framework": "A"}
    msg3 = perform_stop(stop_state3, now, stop_ctrl3)
    assert stop_state3["phase"] == "HUNTING"
    assert stop_state3["trade_log"] == []
    assert stop_ctrl3["mode"] == "STOPPED"
    print(f"[PASS] perform_stop HUNTING: {msg3}")

    # 8. STOPPED mode blocks hunting entirely
    _control.set_mode(_control.MODE_STOPPED, by="test")
    res_stopped = tick(now=now)
    assert res_stopped["status"] == "STOPPED", res_stopped
    print(f"[PASS] control STOPPED: tick returns {res_stopped['status']} without hunting")
    _control.set_mode(_control.MODE_RUNNING, by="test")

    # 9. Framework routing: order opened under B fills with B's exit levels
    from frameworks import get_framework
    fw_b = get_framework("B")
    lv_b = fw_b.initialize_exit(1000.0, 1.0)
    assert lv_b["target_pct"] == 0.015 and lv_b["stop_pct"] == 0.025, lv_b
    assert lv_b["framework"] == "B"

    order_b = create_paper_buy_order("XRPIDR", 1000.0, 40.0, 40000.0, p_info, now=now)
    order_b["tick"] = 1.0
    order_b["framework"] = "B"
    filled_b, pos_b = check_paper_buy_fill(
        order_b, live_price=999.0, now=now, bid=999.0, ask=1001.0,
        exit_levels=fw_b.initialize_exit(order_b["limit_price"], order_b["tick"]))
    assert filled_b and pos_b is not None
    assert pos_b["framework"] == "B", pos_b["framework"]
    assert pos_b["exit_levels"]["target_price"] == 1015.0, pos_b["exit_levels"]
    assert pos_b["exit_levels"]["stop_trigger_price"] == 975.0, pos_b["exit_levels"]
    print("[PASS] framework routing: order under B fills with B's levels "
          f"(target {pos_b['exit_levels']['target_price']}, "
          f"stop {pos_b['exit_levels']['stop_trigger_price']})")

    # 10. A position is pinned: framework A's exit maths for a B position would
    #     give different levels, so the pin matters.
    lv_a = get_framework("A").initialize_exit(1000.0, 1.0)
    assert lv_a["target_price"] != pos_b["exit_levels"]["target_price"], \
        "A and B must not share exit levels, or pinning is untestable"
    print("[PASS] position pinning is meaningful: A target "
          f"{lv_a['target_price']} != B target {pos_b['exit_levels']['target_price']}")

    _control.CONTROL_FILE = _live_control
    shutil.rmtree(_ctrl_sandbox, ignore_errors=True)
    assert _control.CONTROL_FILE != os.path.join(_ctrl_sandbox, "hunter_control.json"), "control sandbox escaped"
    print(f"[PASS] live control untouched ({_control.CONTROL_FILE})")

    # Restore live log paths
    hunt_log.TICKS_DIR, hunt_log.EVENTS_FILE = _live_ticks, _live_events
    assert hunt_log.TICKS_DIR != os.path.join(_sandbox, "hunt_ticks"), "sandbox escaped"
    shutil.rmtree(_sandbox, ignore_errors=True)
    print(f"[PASS] live tick/event logs untouched ({hunt_log.TICKS_DIR})")

    print("\n=== ALL SELF-CHECKS PASSED ===")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--test", "-t", "test"):
        run_self_checks()
        return

    if "--sell-now" in sys.argv:
        run_sell_now()
    if "--stop-now" in sys.argv:
        run_stop_now()
        return

    res = tick()
    if res.get("status") == "OK":
        print(f"{res['timestamp']}  Phase: {res['phase']:<10}  Balance: {res['balance']:,.0f} IDR")
        print(f"  {res['summary']}")
    else:
        print(res.get("message", res))


def run_sell_now():
    """Market-sell the open position and RETURN TO HUNTING.

    Deliberately different from --stop-now, which liquidates AND parks. This is
    the "get me out of this one trade" button: it sells, records the trade with
    EXIT_MANUAL_MARKET, sets a cooldown on the pair so the engine does not
    immediately buy the same coin back, and leaves the mode RUNNING.

    Takes the engine flock so the sell cannot interleave with a tick that is
    mid-decision. Exit codes: 0 = sold, 2 = a tick holds the lock (retry),
    1 = unexpected failure, 3 = nothing to sell.

    The sell itself reuses execute_paper_sell, the same call the engine uses for
    a hard stop. Duplicating the fee/PnL arithmetic here would be the worst
    thing to duplicate: the dashboard would then report different numbers from
    the engine for the same fill.
    """
    now = time.time()
    lock_fd = acquire_lock()
    if lock_fd is None:
        print(json.dumps({"ok": False, "error": "engine tick in progress, retry"}))
        sys.exit(2)

    try:
        check_kill_switch()
        ctrl = load_control()
        state = load_state()

        if state.get("phase") != "HOLDING" or not state.get("position"):
            print(json.dumps({
                "ok": False,
                "error": "no open position to sell",
                "phase": state.get("phase"),
            }))
            sys.exit(3)

        pos = state["position"]
        symbol = pos.get("symbol")
        pair_id = pos.get("pair_id", (symbol or "").lower())
        depth = load_depth(pair_id)
        ticker = fetch_live_ticker(pair_id, depth_fallback=depth)
        best_bid = ticker.get("buy") or ticker.get("last") or pos["entry_price"]
        best_ask = ticker.get("sell") or best_bid

        trade_rec, delta = execute_paper_sell(
            pos,
            exit_price=best_bid,
            exit_reason=EXIT_MANUAL_MARKET,
            now=now,
            fee_override_pct=pos.get("taker_fee_pct", 0.2),
            exit_bid=best_bid,
            exit_ask=best_ask,
        )
        # --- the same config stamping the engine applies on any exit ----------
        trade_rec["framework"] = pos.get("framework", "A")
        trade_rec["framework_version"] = pos.get("framework_version", "?")
        levels = pos.get("exit_levels") or {}
        trade_rec["target_pct"] = levels.get("target_pct")
        trade_rec["stop_pct"] = levels.get("stop_pct")
        trade_rec["ratchet_armed"] = bool(levels.get("ratchet_armed"))

        state.setdefault("trade_log", []).append(trade_rec)
        state["balance_idr"] = state.get("balance_idr", 0) + delta["net_proceeds"]
        state["position"] = None
        state["phase"] = "HUNTING"

        # A manual exit must apply the same cooldown as any other exit, or the
        # engine buys the identical pair on the very next tick.
        hours = float(_config.get(_C, "trading.cooldown_hours", 4))
        state.setdefault("cooldowns", {})[symbol.upper()] = now + hours * 3600

        save_state(state)
        try:
            record_hunt_event(symbol, "MANUAL_MARKET_EXIT", {
                "hunt_id": trade_rec.get("hunt_id"),
                "exit_price": best_bid,
                "net_pnl_idr": trade_rec.get("net_pnl_idr"),
                "net_pnl_pct": trade_rec.get("net_pnl_pct"),
            })
        except Exception:
            pass

        print(json.dumps({
            "ok": True,
            "message": (f"sold {symbol} at {best_bid:,.0f} "
                        f"({trade_rec.get('net_pnl_pct', 0):+.2f}%)"),
            "hunt_id": trade_rec.get("hunt_id"),
            "symbol": symbol,
            "exit_price": best_bid,
            "net_pnl_idr": trade_rec.get("net_pnl_idr"),
            "net_pnl_pct": trade_rec.get("net_pnl_pct"),
            "balance": state.get("balance_idr"),
            "phase": state.get("phase"),
            "mode": ctrl.get("mode"),
        }, indent=2))
    except HaltSignal as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
    except Exception as e:                                  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    finally:
        release_lock(lock_fd)
    sys.exit(0)


def run_stop_now():
    """
    One-shot STOP: liquidate anything open, then park.

    Takes the engine flock so the stop can never interleave with a tick that
    is mid-decision. Prints JSON so the dashboard can show the result.

    Exit codes: 0 = stopped, 2 = a tick holds the lock (caller should retry),
                1 = unexpected failure.
    """
    now = time.time()
    lock_fd = acquire_lock()
    if lock_fd is None:
        print(json.dumps({"ok": False, "error": "engine tick in progress, retry"}))
        sys.exit(2)

    try:
        check_kill_switch()
        ctrl = load_control()
        state = load_state()

        # Mark STOPPING first: if we die mid-liquidation the next tick will
        # finish the job instead of resuming hunting.
        ctrl["mode"] = MODE_STOPPING
        ctrl["changed_at"] = now
        ctrl["changed_by"] = "stop-now"
        save_control(ctrl)

        summary = perform_stop(state, now, ctrl)
        save_state(state)

        print(json.dumps({
            "ok": True,
            "message": summary,
            "mode": load_control()["mode"],
            "phase": state.get("phase"),
            "balance": state.get("balance_idr"),
        }, indent=2))
    except HaltSignal as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
    except Exception as e:                                  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    finally:
        release_lock(lock_fd)
    sys.exit(0)


if __name__ == "__main__":
    main()
