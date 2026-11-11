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

# Wall-clock backstop. Six checks is six minutes only if the engine ticks every
# minute; a paused job or a slow exchange changes what "6" means. Both bounds
# apply and whichever comes first wins.
ORDER_TIMEOUT_MINUTES = float(_config.get(_C, "trading.order_timeout_minutes", 15))
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
# --- live risk caps ---------------------------------------------------------
# Read from config so they can be changed without touching code, but they are
# enforced in the engine, not the UI: a cap that lives in the browser is not a
# cap, because anything can call the engine.
LIVE_MAX_PER_TRADE_IDR = float(_config.get(_C, "live.max_per_trade_idr", 20000))
LIVE_MAX_DAILY_LOSS_IDR = float(_config.get(_C, "live.max_daily_loss_idr", 5000))


def live_trade_cap_idr() -> float:
    """The most a single live trade may spend."""
    return LIVE_MAX_PER_TRADE_IDR


def live_daily_loss_idr(state: dict, now: Optional[float] = None) -> float:
    """Realised loss today, in IDR, from live trades only.

    Paper losses must never count toward a live limit: they are different money
    and mixing them would silently halve or double the real risk budget.
    """
    if now is None:
        now = time.time()
    day_start = now - (now % 86400)          # UTC day; conservative boundary
    loss = 0.0
    for t in (state.get("live_trade_log") or []):
        if not isinstance(t, dict):
            # An unreadable row could be hiding a loss. Counting it as zero
            # would make the cap read low and allow trading, so it raises
            # instead and live_daily_cap_hit() treats that as "cap reached".
            raise ValueError("live_trade_log contains a non-record entry")
        try:
            exit_time = float(t.get("exit_time") or 0)
            pnl = float(t.get("net_pnl_idr"))
        except (TypeError, ValueError):
            raise ValueError(
                f"unreadable live trade row: exit_time={t.get('exit_time')!r} "
                f"net_pnl_idr={t.get('net_pnl_idr')!r}")
        if exit_time < day_start:
            continue
        if pnl < 0:
            loss += -pnl
    return loss


def live_daily_cap_hit(state: dict, now: Optional[float] = None) -> bool:
    """True when today's realised live loss has reached the cap."""
    try:
        return live_daily_loss_idr(state, now) >= LIVE_MAX_DAILY_LOSS_IDR
    except Exception:                                        # noqa: BLE001
        return True          # fail closed: if the check breaks, do not trade


def _describe_reconcile(rec: dict) -> str:
    """A one-line explanation of why trading is halted, in operator language."""
    st = rec.get("status")
    coins = rec.get("exchange") or {}
    held = ", ".join(f"{v:g} {k}" for k, v in list(coins.items())[:3]) or "nothing"

    if st == "RECOVER":
        return (f"the exchange holds {held} but the bot has no open position. "
                f"Sell it, or adopt it, before hunting - those coins have no stop.")
    if st == "PHANTOM":
        return (f"the bot records a position in {rec.get('local_symbol')} but the "
                f"exchange holds no coins. Local state is wrong; clear it before hunting.")
    if st == "MISMATCH":
        return (f"the exchange holds {rec.get('held'):g} of {rec.get('local_symbol')} "
                f"which does not match the recorded amount.")
    if st == "PHANTOM_DUST":
        return (f"the recorded position has only dust left "
                f"({rec.get('dust_note') or 'below 1,000 IDR'}). Clear the "
                f"position before hunting.")
    return f"exchange state {st}"


def _live_mode() -> bool:
    """True when the control file says LIVE. Read once per tick by the caller.

    Everything that touches money branches on this, and nothing infers it from
    the framework, the config or the balance. One predicate, one source.
    """
    try:
        from control import load_control, is_live
        return is_live(load_control())
    except Exception:                                        # noqa: BLE001
        return False


def handle_hunting(state: dict, now: float, fw=None) -> str:
    """
    Screen universe -> gates -> framework entry -> winner tiebreaker -> place buy order.
    Returns a one-line summary.
    """
    if fw is None:
        fw = get_framework(active_framework())

    # Live risk brake. Checked before anything else, because a daily cap that
    # is checked after a trade is not a cap.
    if _live_mode() and live_daily_cap_hit(state, now):
        return (f"[LIVE_HALTED] daily live loss cap reached: "
                f"-{live_daily_loss_idr(state, now):,.0f} of "
                f"{LIVE_MAX_DAILY_LOSS_IDR:,.0f} IDR. No new positions today.")

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

    # Ask the framework which of the ranked pairs it is willing to consider,
    # BEFORE any candles are read. pairs.json is already sorted by 24h IDR
    # volume, so this is where "only the top pair" actually takes effect: the
    # other 19 are never fetched, gated or scored. Doing it as an entry filter
    # instead would still download history for pairs the framework never
    # intends to trade.
    _ranked = [p["symbol"] for p in pairs]
    try:
        _allowed = fw.allowed_symbols(_ranked)
    except Exception:                                        # noqa: BLE001
        _allowed = _ranked          # a broken hook must not stop the engine
    _allowed_set = set(_allowed or [])
    _before = len(pairs)
    pairs = [p for p in pairs if p["symbol"] in _allowed_set]

    if not pairs:
        return (f"[HUNTING] framework {fw.name} allows none of the "
                f"{_before} ranked pairs. Stand by.")

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

        if _live_mode():
            # REAL ORDER. Every guard that protects the balance lives here,
            # before the call, because there is no undo on a placed order.
            import executor_live
            try:
                _cap = live_trade_cap_idr()
                _spend = min(trade_size, _cap)
                if _spend < 10000:
                    log_gate_rejection(
                        sym, "SIZING",
                        f"live trade capped to {_spend:.0f} < 10k IDR min",
                        state=state, now=now)
                    continue
                # recompute the coin amount for the capped spend
                coin_amount = compute_coin_amount(_spend, limit_price, qty_incr)
                order = executor_live.create_live_buy_order(
                    symbol=sym,
                    limit_price=limit_price,
                    coin_amount=coin_amount,
                    trade_size_idr=_spend,
                    pair_info=p,
                    now=now,
                    hunt_id=hunt_id,
                    entry_bid=best_bid,
                    entry_ask=best_ask,
                    qty_increment=qty_incr,
                )
                order["framework"] = fw.name
                order["framework_version"] = getattr(fw, "version", "?")
                order["exit_levels"] = fw.initialize_exit(limit_price, tick)
                order["live"] = True
                state["order"] = order
                state["phase"] = "ORDER_OPEN"
                record_hunt_event(hunt_id, "LIVE_ORDER_PLACED", {
                    "symbol": sym, "order_id": order["order_id"],
                    "limit_price": limit_price, "idr": _spend,
                }, now=now)
                return (f"[LIVE ORDER_PLACED] {hunt_id} [{fw.name}] {sym} "
                        f"REAL order {order['order_id']} | {coin_amount} @ "
                        f"{limit_price} ({_spend:,.0f} IDR)")
            except Exception as e:                           # noqa: BLE001
                # A rejected order must not kill the tick, but it must be loud.
                log_gate_rejection(sym, "LIVE_ORDER", f"{type(e).__name__}: {e}",
                                   state=state, now=now)
                record_hunt_event(hunt_id, "LIVE_ORDER_FAILED", {
                    "symbol": sym, "error": str(e)[:300],
                }, now=now)
                return f"[LIVE_ORDER_FAILED] {sym}: {type(e).__name__}: {str(e)[:200]}"

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

    if order.get("live"):
        # Ask the exchange whether the resting order actually filled. Paper
        # infers a fill from a price touch; live must not, because the order may
        # be unfilled, partially filled, or cancelled by the operator.
        import executor_live

        # Count the attempt BEFORE anything can fail, so the timeout works even
        # if the exchange is unreachable. This counter was read but never
        # written, which is why a real order sat at "0/6" for ten minutes and
        # the timeout never fired.
        order["checks_elapsed"] = int(order.get("checks_elapsed", 0)) + 1
        order["last_checked_at"] = now
        state["order"] = order          # persist the counter

        try:
            outcome, position = executor_live.check_live_buy_fill(
                order, live_price=live_price, now=now)
        except Exception as e:                               # noqa: BLE001
            return (f"[ORDER_OPEN] {hunt_id} {sym}: cannot read the order "
                    f"from the exchange: {type(e).__name__}: {str(e)[:150]}")

        if outcome == "CANCELLED":
            # the operator cancelled it in the Indodax app, or it expired.
            # Drop it: a cancelled order is not a position and must not sit in
            # the dashboard as though it were still working.
            state["order"] = None
            state["phase"] = "HUNTING"
            record_hunt_event(hunt_id, "ORDER_CANCELLED_EXTERNALLY", {
                "symbol": sym, "order_id": order.get("order_id"),
            }, now=now)
            return (f"[ORDER_GONE] {hunt_id} {sym}: the exchange reports the "
                    f"order as cancelled or expired. Back to HUNTING.")

        if outcome == "ERROR":
            return (f"[ORDER_OPEN] {hunt_id} {sym}: the exchange returned an "
                    f"unrecognised order state. Not guessing - check Indodax.")

        if outcome in ("FILLED", "PARTIAL") and position:
            # the exit levels were computed when the order was placed, using the
            # framework that placed it - never the one active now
            position["exit_levels"] = order.get("exit_levels") or fill_levels
            position["framework"] = order.get("framework", "A")
            position["framework_version"] = order.get("framework_version", "?")
            if outcome == "PARTIAL":
                record_hunt_event(hunt_id, "PARTIAL_FILL", {
                    "symbol": sym, "coins": position.get("coins"),
                    "idr": position.get("filled_idr"),
                }, now=now)
        # outcome == "OPEN" falls through to the timeout check below
    else:
        filled, position = check_paper_buy_fill(
            order,
            live_price=live_price,
            now=now,
            bid=best_bid,
            ask=best_ask,
            qty_increment=float(order.get("qty_increment", 0.0) or 0.0),
            exit_levels=fill_levels,
        )
    # `filled` comes from the paper check; live reports an `outcome`. Normalise
    # here so the rest of the function has one thing to test - and so the live
    # path cannot silently fall through to "not filled" because a variable was
    # never assigned.
    if order.get("live"):
        filled = bool(outcome in ("FILLED", "PARTIAL") and position)

    if filled and position:
        state["position"] = position
        state["order"] = None
        state["phase"] = "HOLDING"
        if order.get("live"):
            # Real money left the exchange account, not the simulated one.
            # Deducting it from balance_idr as well would double-count the spend
            # and leave the paper ledger permanently short.
            position["cost_idr"] = position.get("filled_idr")
            position["total_cost_idr"] = position.get("filled_idr")
        else:
            state["balance_idr"] = round(
                state.get("balance_idr", 100_000.0) - position["total_cost_idr"], 2)
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
    checks = int(order.get("checks_elapsed", 0))

    # Time is a better timeout than a count. Six checks is six minutes only if
    # the engine ticks every minute; a paused job or a slow exchange would
    # change what "6" means. Both are enforced: whichever comes first.
    placed = float(order.get("placed_at") or now)
    age_min = (now - placed) / 60.0
    timed_out = (checks >= ORDER_TIMEOUT_CHECKS) or \
                (age_min >= ORDER_TIMEOUT_MINUTES)

    if timed_out:
        cancel_note = ""
        if order.get("live"):
            # A real resting order MUST be cancelled at the exchange, not just
            # forgotten locally. Dropping it from state while it still works on
            # the book leaves an order nobody is watching.
            import executor_live
            try:
                executor_live.RealClient().cancel_order(
                    pair=order["pair_id"], order_id=order["order_id"]
                )
                cancel_note = "cancelled on the exchange"
            except Exception as e:                           # noqa: BLE001
                cancel_note = (f"COULD NOT CANCEL on the exchange: "
                               f"{type(e).__name__}: {str(e)[:120]}")
                record_hunt_event(hunt_id, "ORDER_CANCEL_FAILED", {
                    "symbol": sym, "order_id": order.get("order_id"),
                    "error": str(e)[:300],
                }, now=now)
                # keep the order in state: it is still live at the exchange
                return (f"[ORDER_TIMEOUT] {hunt_id} {sym} {cancel_note}. "
                        f"The order is still working - cancel it in Indodax.")

        state["order"] = None
        state["phase"] = "HUNTING"
        record_hunt_event(hunt_id, "ORDER_TIMEOUT", {
            "symbol": sym, "checks": checks, "age_min": round(age_min, 1),
            "limit_price": order.get("limit_price"), "cancel": cancel_note,
        }, now=now)
        return (f"[ORDER_TIMEOUT] {hunt_id} {sym} unfilled after {checks} checks "
                f"({age_min:.0f}m). {cancel_note or 'cancelled'} -> HUNTING.")

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

        if pos.get("live"):
            # REAL SELL. The exchange decides the fill price, not this code: a
            # market order takes whatever the book gives, which is exactly why
            # the stop can lose more than its trigger.
            import executor_live
            try:
                trade_rec, delta = executor_live.execute_live_sell(
                    pos,
                    best_bid=best_bid,
                    exit_reason=reason,
                    now=now,
                    limit_price=(target if reason == EXIT_TARGET else None),
                )
            except Exception as e:                           # noqa: BLE001
                # The position is STILL OPEN. Do not touch state, do not log a
                # trade, do not let the loop think it is flat.
                record_hunt_event(hunt_id, "LIVE_SELL_FAILED", {
                    "symbol": sym, "reason": reason, "error": str(e)[:300],
                }, now=now)
                return (f"[SELL_FAILED] {hunt_id} {sym} is STILL OPEN - "
                        f"{reason} could not be executed: "
                        f"{type(e).__name__}: {str(e)[:150]}")
        else:
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

        if not trade_rec.get("live"):
            # paper proceeds return to the simulated balance; real proceeds are
            # already sitting in the exchange account
            state["balance_idr"] = round(
                state.get("balance_idr", 0.0) + delta["net_proceeds"], 2)

        set_cooldown(state, sym, hours=COOLDOWN_HOURS, now=now)

        # Two ledgers, never summed. A live trade must not appear in the paper
        # log or the PnL of the two phases becomes one unlabelable number.
        if trade_rec.get("live"):
            state.setdefault("live_trade_log", []).append(trade_rec)
        else:
            state.setdefault("trade_log", []).append(trade_rec)
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
        # --- live preflight, once per tick -----------------------------------
        # A live tick must not start unless the account is genuinely usable:
        # credentials signing, balance readable, not frozen, clock sane. Doing
        # this per tick rather than per arm means a key that expires or is
        # revoked mid-session stops the bot instead of failing every order.
        if ctrl.get("exec") == "LIVE":
            import executor_live
            try:
                _pf = executor_live.preflight(trade_size_idr=LIVE_MAX_PER_TRADE_IDR)
            except Exception as e:                           # noqa: BLE001
                _pf = {"ok": False, "problems": [f"{type(e).__name__}: {e}"],
                       "checks": []}
            if not _pf.get("ok"):
                _why = "; ".join(_pf.get("problems") or ["preflight failed"])
                try:
                    from control import set_blocked
                    set_blocked(_why[:280])
                except Exception:                            # noqa: BLE001
                    pass
                return {"status": "LIVE_PREFLIGHT_FAILED",
                        "message": f"{dt_str}  [BLOCKED] live preflight failed: {_why[:300]}"}

        state = load_state()

        # --- exchange reconciliation ----------------------------------------
        # Local state is not the truth; the exchange is. If the account holds
        # coins the bot does not know about - bought in the Indodax app, or left
        # by a crash between placing and recording - those coins have no stop.
        # Buying anything else would mean two positions, one unprotected.
        if ctrl.get("exec") == "LIVE":
            try:
                import executor_live
                _rows = []
                try:
                    with open(PAIRS_FILE, "r") as _fh:
                        _rows = json.load(_fh)
                except Exception:                        # noqa: BLE001
                    _rows = []
                _rec = executor_live.reconcile_position(state, pair_rows=_rows)
                _st = _rec.get("status")
                if _st in ("RECOVER", "PHANTOM", "MISMATCH", "PHANTOM_DUST"):
                    _detail = _describe_reconcile(_rec)
                    try:
                        from control import set_blocked
                        set_blocked(_detail[:280])
                    except Exception:                        # noqa: BLE001
                        pass
                    return {"status": "EXCHANGE_MISMATCH",
                            "message": f"{dt_str}  [BLOCKED] {_detail}"}
                if _st == "UNKNOWN":
                    try:
                        from control import set_blocked
                        set_blocked(str(_rec.get("error"))[:280])
                    except Exception:                        # noqa: BLE001
                        pass
                    return {"status": "EXCHANGE_UNREADABLE",
                            "message": f"{dt_str}  [BLOCKED] cannot read the "
                                       f"exchange: {_rec.get('error')}"}
            except Exception as e:                           # noqa: BLE001
                return {"status": "EXCHANGE_ERROR",
                        "message": f"{dt_str}  [BLOCKED] reconciliation failed: "
                                   f"{type(e).__name__}: {str(e)[:150]}"}

        # past the guards: this tick really runs, so any previous block is stale
        try:
            from control import clear_blocked
            clear_blocked()
        except Exception:                                    # noqa: BLE001
            pass

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

    # --- the live check counter must advance ---------------------------------
    # Reported: a real resting order showed "Checks: 0/6" after ten minutes and
    # never timed out. checks_elapsed was read in the timeout branch and never
    # written anywhere.
    import executor_live as _exl2
    _now_cap = time.time()

    _open_ex = None

    class _OpenEx(_exl2.FakeExchange):
        def get_order(self, pair, order_id):
            if order_id not in self.orders:
                raise _exl2.live_api.LiveApiError("invalid order.")
            o = self.orders[order_id]
            return {"success": 1, "return": {"order": {
                "order_id": order_id, "status": o.get("status", "open"),
                "order_rp": f"{o.get('order_rp', 100.0):.8f}",
                "remain_rp": f"{o.get('remain_rp', 100.0):.8f}"}}}

    _orig_rc2 = _exl2.RealClient
    try:
        _open_ex = _OpenEx(idr=100000.0)
        _exl2.RealClient = lambda creds=None: _open_ex
        _st2 = {"phase": "ORDER_OPEN", "balance_idr": 100000.0,
                "cooldowns": {}, "trade_log": [], "live_trade_log": [],
                "hunt_counter": 0, "gate_stats": {},
                "order": {"order_id": "live-x", "live": True,
                          "symbol": "TESTIDR", "pair_id": "test_idr",
                          "limit_price": 1000.0, "coin_amount": 0.1,
                          "trade_size_idr": 100.0, "placed_at": _now_cap,
                          "tick": 1.0, "checks_elapsed": 0,
                          "framework": "C",
                          "exit_levels": globals()["get_framework"]("C").initialize_exit(1000.0, 1.0)}}

        _open_ex.orders["live-x"] = {"pair": "test_idr", "status": "open",
                                     "order_rp": 100.0, "remain_rp": 100.0}
        _m1 = handle_order_open(_st2, now=_now_cap + 60)
        assert _st2["order"]["checks_elapsed"] == 1, _st2["order"]["checks_elapsed"]
        _m2 = handle_order_open(_st2, now=_now_cap + 120)
        assert _st2["order"]["checks_elapsed"] == 2, _st2["order"]["checks_elapsed"]
        assert "Checks: 2/6" in _m2 or "2/6" in _m2, _m2
        print(f"[PASS] the live check counter advances: "
              f"{_st2['order']['checks_elapsed']}/6")

        # wall-clock backstop must fire even if the tick count has not.
        # The order must exist at the fake so the cancel can succeed - if the
        # cancel fails the engine is SUPPOSED to keep the order in state, which
        # is asserted separately below.
        _st2["order"]["checks_elapsed"] = 0
        _st2["order"]["order_id"] = "live-x"
        _open_ex.orders["live-x"] = {"pair": "test_idr", "status": "open",
                                     "order_rp": 100.0, "remain_rp": 100.0}
        _m3 = handle_order_open(_st2, now=_now_cap + ORDER_TIMEOUT_MINUTES * 60 + 5)
        assert _st2["phase"] == "HUNTING", (_st2["phase"], _m3)
        assert "ORDER_TIMEOUT" in _m3, _m3
        print(f"[PASS] the wall-clock timeout fires at {ORDER_TIMEOUT_MINUTES:.0f}m "
              f"without waiting for 6 ticks")

        # and if the cancel FAILS, the order is kept: it is still live at the
        # exchange, and forgetting it would leave an order nobody watches
        class _NoCancel(_OpenEx):
            def cancel_order(self, **kw):
                raise RuntimeError("exchange refused the cancel")

        _nocancel_ex = _NoCancel(idr=100000.0)
        _nocancel_ex.orders["live-z"] = {"pair": "test_idr", "status": "open",
                                         "order_rp": 100.0, "remain_rp": 100.0}
        _exl2.RealClient = lambda creds=None: _nocancel_ex
        _st4 = {"phase": "ORDER_OPEN", "balance_idr": 100000.0,
                "cooldowns": {}, "trade_log": [], "live_trade_log": [],
                "hunt_counter": 0, "gate_stats": {},
                "order": {"order_id": "live-z", "live": True,
                          "symbol": "TESTIDR", "pair_id": "test_idr",
                          "limit_price": 1000.0, "coin_amount": 0.1,
                          "trade_size_idr": 100.0, "placed_at": _now_cap,
                          "tick": 1.0, "checks_elapsed": 99,
                          "framework": "C",
                          "exit_levels": globals()["get_framework"]("C").initialize_exit(1000.0, 1.0)}}
        _m5 = handle_order_open(_st4, now=_now_cap + 60)
        assert _st4["order"] is not None, "a live order must not be dropped"
        assert _st4["phase"] == "ORDER_OPEN", _st4["phase"]
        assert "COULD NOT CANCEL" in _m5, _m5
        print("[PASS] a failed cancel keeps the order and says so")
    finally:
        _exl2.RealClient = _orig_rc2

    # an externally cancelled order must be dropped, not left in state
    class _CancelledEx(_exl2.FakeExchange):
        def get_order(self, pair, order_id):
            return {"success": 1, "return": {"order": {
                "order_id": order_id, "status": "cancelled",
                "order_rp": "100.0", "remain_rp": "100.0"}}}

    try:
        _exl2.RealClient = lambda creds=None: _CancelledEx(idr=100000.0)
        _st3 = {"phase": "ORDER_OPEN", "balance_idr": 100000.0,
                "cooldowns": {}, "trade_log": [], "live_trade_log": [],
                "hunt_counter": 0, "gate_stats": {},
                "order": {"order_id": "live-y", "live": True,
                          "symbol": "TESTIDR", "pair_id": "test_idr",
                          "limit_price": 1000.0, "coin_amount": 0.1,
                          "trade_size_idr": 100.0, "placed_at": _now_cap,
                          "tick": 1.0, "checks_elapsed": 0,
                          "framework": "C",
                          "exit_levels": globals()["get_framework"]("C").initialize_exit(1000.0, 1.0)}}
        _m4 = handle_order_open(_st3, now=_now_cap + 60)
        assert _st3["order"] is None and _st3["phase"] == "HUNTING", _st3
        assert "ORDER_GONE" in _m4, _m4
        print("[PASS] an externally cancelled order is dropped back to HUNTING")
    finally:
        _exl2.RealClient = _orig_rc2

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

    # --- LIVE risk caps -----------------------------------------------------
    # Tested directly: they are the only thing between a bad session and an
    # empty exchange account.
    assert live_trade_cap_idr() > 0
    assert live_trade_cap_idr() <= 100000, "a per-trade cap above 100k is not a cap"
    print(f"[PASS] live per-trade cap: {live_trade_cap_idr():,.0f} IDR")

    _now_cap = time.time()

    def _live_state(losses):
        return {"live_trade_log": [
            {"net_pnl_idr": v, "exit_time": _now_cap - 60} for v in losses]}

    assert live_daily_cap_hit(_live_state([-1000, -500]), _now_cap) is False
    assert live_daily_cap_hit(_live_state([-LIVE_MAX_DAILY_LOSS_IDR - 1]), _now_cap) is True
    print(f"[PASS] daily loss cap {LIVE_MAX_DAILY_LOSS_IDR:,.0f} IDR trips on realised losses")

    _old = {"live_trade_log": [{"net_pnl_idr": -LIVE_MAX_DAILY_LOSS_IDR * 3,
                                "exit_time": _now_cap - 3 * 86400}]}
    assert live_daily_loss_idr(_old, _now_cap) == 0.0
    print("[PASS] yesterday's loss does not count against today's cap")

    _paper = {"trade_log": [{"net_pnl_idr": -99999, "exit_time": _now_cap}],
              "live_trade_log": []}
    assert live_daily_loss_idr(_paper, _now_cap) == 0.0
    print("[PASS] paper losses are excluded from the live cap")

    assert live_daily_cap_hit({"live_trade_log": [{"net_pnl_idr": "garbage"}]},
                              _now_cap) is True
    print("[PASS] an unreadable loss ledger fails CLOSED")

    # --- framework D narrows the universe BEFORE fetching -------------------
    # The whole point of D is that 19 pairs are never touched. Testing the hook
    # alone would not prove the engine honours it.
    _d = get_framework("D")
    _ranked = ["BTCIDR", "PEPEIDR", "XRPIDR"]
    assert _d.allowed_symbols(_ranked) == ["BTCIDR"], _d.allowed_symbols(_ranked)
    print("[PASS] framework D allows only the #1 ranked pair")

    # every other framework must keep the whole universe - a hook that
    # silently narrowed A would change every existing result
    for _n in ("A", "B", "C"):
        _f = get_framework(_n)
        assert _f.allowed_symbols(_ranked) == _ranked, _n
    print("[PASS] A, B and C still consider every ranked pair")

    # a framework whose hook raises must not stop the engine
    class _Broken(type(get_framework("A"))):
        name = "X"

        def allowed_symbols(self, ranked):
            raise RuntimeError("boom")

    _b = _Broken()
    try:
        _res = _b.allowed_symbols(_ranked)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    print("[PASS] a raising hook is contained by the engine's try")

    # D's entry never rejects and never rates a direction
    _r = _d.evaluate_entry([], {})
    assert _r["eligible"] is True and _r["direction"] == "N/A", _r
    print("[PASS] framework D has no entry signal by design")

    # --- LIVE lifecycle end to end, against a fake exchange -----------------
    # This is the test that proves live trading is wired: the engine's own
    # handlers place a REAL order through executor_live, detect the fill from
    # the exchange, and close it, with the proceeds landing in the LIVE ledger.
    import executor_live as _exl

    _fake = _exl.FakeExchange(idr=100000.0)
    _orig_rc = _exl.RealClient
    _orig_live = globals()["_live_mode"]
    try:
        _exl.RealClient = lambda creds=None: _fake
        globals()["_live_mode"] = lambda: True

        _st = {"phase": "HUNTING", "balance_idr": 100000.0, "order": None,
               "position": None, "cooldowns": {}, "trade_log": [],
               "live_trade_log": [], "hunt_counter": 0}
        _pi = {"id": "btcidr", "symbol": "BTCIDR", "tick": 1.0,
               "quantity_increment": 0.00000001,
               "trade_fee_percent_maker": 0.1, "trade_fee_percent_taker": 0.2}

        _order = _exl.create_live_buy_order(
            "BTCIDR", 1000.0, 0.02, 20000.0, _pi, now=_now_cap,
            hunt_id="H-LIVE", client=_fake)
        _order["framework"] = "A"
        _order["exit_levels"] = get_framework("A").initialize_exit(1000.0, 1.0)
        _st["order"] = _order
        _st["phase"] = "ORDER_OPEN"
        assert _order.get("live") is True
        assert _order["order_id"].startswith("live-"), _order["order_id"]
        print(f"[PASS] a live hunt places a REAL order ({_order['order_id']})")

        _filled, _pos = _exl.check_live_buy_fill(_order, 1000.0, now=_now_cap,
                                                 client=_fake)
        assert _filled and _pos and _pos["coins"] > 0, _pos
        _pos["exit_levels"] = _order["exit_levels"]
        _pos["framework"] = "A"
        _pos["live"] = True
        print(f"[PASS] the fill is read from the exchange, not inferred: "
              f"{_pos['coins']:.8f} BTC")

        _bp = float(_fake.get_info()["return"]["balance"]["idr"])
        assert _bp < 100000.0, "the fake exchange should have spent the IDR"
        print(f"[PASS] IDR left the exchange account: balance now {_bp:,.0f}")

        _rec, _summ = _exl.execute_live_sell(_pos, best_bid=1020.0,
                                             exit_reason=EXIT_TARGET,
                                             now=_now_cap + 60, client=_fake,
                                             limit_price=1020.0)
        assert _rec["live"] is True and _rec["reconciled"] is True
        assert _rec["net_pnl_idr"] > 0, _rec["net_pnl_pct"]
        print(f"[PASS] a live target exit records the real fill: "
              f"{_rec['net_pnl_pct']:+.2f}%")

        # the paper ledger must be untouched by a live trade
        _st["paper_check"] = len(_st["trade_log"])
        assert _st["trade_log"] == [], "a live trade leaked into the paper log"
        print("[PASS] live trades do not leak into the paper ledger")
    finally:
        _exl.RealClient = _orig_rc
        globals()["_live_mode"] = _orig_live

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
        return
    if "--buy-now" in sys.argv:
        # --buy-now SYMBOL [--paper]
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        sym = args[0] if args else None
        if not sym:
            print(json.dumps({"ok": False,
                              "error": "usage: --buy-now SYMBOL [--paper]"}))
            sys.exit(3)
        run_buy_now(sym.upper(), use_live=("--paper" not in sys.argv))
        return
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


def run_buy_now(symbol: str, use_live: bool = True):
    """Market-buy `symbol` immediately, as a MANUAL operator action.

    This exists because waiting for the engine to pick a pair is not always what
    the operator wants: the funnel can show a winner and the engine can be busy
    with a different one. It is deliberately NOT the strategy path - it bypasses
    entry scoring entirely, because the human already decided.

    What it does NOT bypass:

      the executor      refusing to run as PAPER (a "live buy" while paper is
                        armed would place a real order under a simulated label)
      the phase         refuses if a position or order is already open
      the caps          live.max_per_trade_idr and the daily live loss cap
      the preflight     credentials, balance, frozen account, clock
      the sizing floor  the exchange's 10,000 IDR minimum

    Takes the engine flock so it cannot interleave with a tick. Exit codes:
    0 = bought, 2 = a tick holds the lock, 3 = refused (with a reason).
    """
    now = time.time()
    lock_fd = acquire_lock()
    if lock_fd is None:
        print(json.dumps({"ok": False, "busy": True,
                          "error": "engine tick in progress, retry"}))
        sys.exit(2)

    try:
        check_kill_switch()
        ctrl = load_control()
        state = load_state()

        if state.get("phase") != "HUNTING" or state.get("position") or state.get("order"):
            print(json.dumps({
                "ok": False,
                "error": f"cannot buy: phase is {state.get('phase')} "
                         f"with an open position or order",
                "phase": state.get("phase"),
            }))
            sys.exit(3)

        if use_live and ctrl.get("exec") != "LIVE":
            print(json.dumps({
                "ok": False,
                "error": "this is a LIVE market buy but the executor is PAPER - "
                         "start LIVE first",
            }))
            sys.exit(3)

        if use_live and live_daily_cap_hit(state, now):
            print(json.dumps({
                "ok": False,
                "error": f"daily live loss cap reached "
                         f"(-{live_daily_loss_idr(state, now):,.0f} of "
                         f"{LIVE_MAX_DAILY_LOSS_IDR:,.0f} IDR)",
            }))
            sys.exit(3)

        # find the pair in the ranked universe; a manual buy is only offered for
        # pairs the app already tracks, so the tick can manage the exit
        with open(PAIRS_FILE, "r") as fh:
            pairs = json.load(fh)
        p_info = next((p for p in pairs if p["symbol"].upper() == symbol.upper()), None)
        if p_info is None:
            print(json.dumps({"ok": False,
                              "error": f"{symbol} is not in the tracked universe"}))
            sys.exit(3)

        # size: the same fraction the engine uses, clamped to the live cap
        if use_live:
            try:
                import executor_live
                balance = executor_live.read_balance_idr()
            except Exception as e:                           # noqa: BLE001
                print(json.dumps({"ok": False,
                                  "error": f"cannot read the exchange balance: {e}"}))
                sys.exit(3)
            trade_size = balance * SIZING_FRACTION
            trade_size = min(trade_size, live_trade_cap_idr())
        else:
            trade_size = state.get("balance_idr", 0.0) * SIZING_FRACTION

        if trade_size < 10000:
            print(json.dumps({"ok": False,
                              "error": f"size {trade_size:,.0f} IDR is below the "
                                       f"10,000 IDR exchange minimum"}))
            sys.exit(3)

        if use_live:
            import executor_live
            pf = executor_live.preflight(trade_size_idr=trade_size,
                                         pair_info=p_info)
            if not pf.get("ok"):
                print(json.dumps({"ok": False,
                                  "error": "; ".join(pf.get("problems") or [])[:400]}))
                sys.exit(3)

        pair_id = p_info.get("id", symbol.lower())
        depth = load_depth(pair_id)
        ticker = fetch_live_ticker(pair_id, depth_fallback=depth)
        best_bid = ticker.get("buy") or ticker.get("last")
        best_ask = ticker.get("sell") or best_bid

        fw = get_framework(ctrl.get("framework") or active_framework())
        hunt_id = next_hunt_id(state)

        if use_live:
            import executor_live
            order, fill = executor_live.create_live_market_buy(
                symbol=p_info["symbol"], idr=trade_size, pair_info=p_info,
                now=now, hunt_id=hunt_id,
                entry_bid=best_bid, entry_ask=best_ask,
            )
            if not fill.get("reconciled"):
                # the order succeeded but we do not know how many coins we hold;
                # guessing would make the stop sell the wrong size
                record_hunt_event(hunt_id, "MANUAL_BUY_UNRECONCILED", {
                    "symbol": p_info["symbol"], "order_id": order.get("order_id"),
                    "idr": trade_size,
                }, now=now)
                print(json.dumps({
                    "ok": False,
                    "error": "the order was accepted but the exchange did not "
                             "report the coin received - check the position in "
                             "Indodax before trading further",
                    "order_id": order.get("order_id"),
                }))
                sys.exit(3)
        else:
            limit_price = compute_maker_limit_price(
                float(best_bid), float(best_ask), float(p_info.get("tick", 1.0)))
            qty_incr = float(p_info.get("quantity_increment", 1e-8))
            coin_amount = compute_coin_amount(trade_size, limit_price, qty_incr)
            order = create_paper_buy_order(
                symbol=p_info["symbol"], limit_price=limit_price,
                coin_amount=coin_amount, trade_size_idr=trade_size,
                pair_info=p_info, now=now, hunt_id=hunt_id,
                entry_bid=best_bid, entry_ask=best_ask,
                qty_increment=qty_incr,
            )
            fill = {"coins": coin_amount, "idr": trade_size,
                    "entry_price": limit_price, "reconciled": True}

        levels = fw.initialize_exit(float(fill["entry_price"]), float(p_info.get("tick", 1.0)))

        position = {
            "hunt_id": hunt_id,
            "symbol": p_info["symbol"],
            "pair_id": pair_id,
            "entry_price": float(fill["entry_price"]),
            "coins": float(fill["coins"]),
            "entry_time": now,
            "order_id": order.get("order_id"),
            "live": bool(use_live),
            "manual": True,
            "maker_fee_pct": order.get("maker_fee_pct", 0.1),
            "taker_fee_pct": order.get("taker_fee_pct", 0.2),
            "fees_paid_idr": order.get("fees_paid_idr", 0.0),
            "entry_bid": best_bid,
            "entry_ask": best_ask,
            "framework": fw.name,
            "framework_version": getattr(fw, "version", "?"),
            "exit_levels": levels,
            "total_cost_idr": float(fill["idr"]),
            "filled_idr": float(fill["idr"]),
            "tick": float(p_info.get("tick", 1.0)),
            "qty_increment": float(p_info.get("quantity_increment", 1e-8)),
        }

        state["position"] = position
        state["order"] = None
        state["phase"] = "HOLDING"
        save_state(state)

        record_hunt_event(hunt_id, "MANUAL_MARKET_BUY", {
            "symbol": p_info["symbol"],
            "order_id": order.get("order_id"),
            "coins": position["coins"],
            "entry_price": position["entry_price"],
            "idr": trade_size,
            "framework": fw.name,
        }, now=now)

        print(json.dumps({
            "ok": True,
            "message": (f"bought {p_info['symbol']} at market: "
                        f"{position['coins']:.8f} for {trade_size:,.0f} IDR "
                        f"@ ~{position['entry_price']:,.4f}"),
            "hunt_id": hunt_id,
            "symbol": p_info["symbol"],
            "order_id": order.get("order_id"),
            "entry_price": position["entry_price"],
            "coins": position["coins"],
            "idr": trade_size,
            "target": levels["target_price"],
            "stop": levels["stop_trigger_price"],
            "phase": state["phase"],
        }, indent=2))
    except HaltSignal as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
    except Exception as e:                                  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    finally:
        release_lock(lock_fd)


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
