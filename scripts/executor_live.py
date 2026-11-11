#!/usr/bin/env python3
"""
executor_live.py — the real-money executor.

Same function names and return shapes as executor_paper, so run_hunter swaps an
executor rather than growing a second trading code path. That is the whole point:
one engine, one set of rules, two backends. A parallel live code path would drift
from the paper one and the paper results would stop predicting live behaviour.

What is genuinely different from paper, and cannot be papered over:

  FILLS ARE NOT GUESSED. Paper says "the price touched my limit, so I own it."
  Live must ask the exchange what actually happened, because a resting limit can
  fill partially, fill late, or not fill at all.

  PARTIAL FILLS EXIST. Paper is all-or-nothing. Live can own 40% of a position,
  and the bot must know that: the target and stop apply to the coins it actually
  holds, not the amount it wanted.

  ORDERS LEAK. A crashed process leaves a resting order on the book. Every
  order recorded here is also recoverable from the exchange, so reconciliation
  does not depend on local state being intact.

  A SELL CAN FAIL. If the market moves or liquidity vanishes, the stop's market
  sell is rejected. That is reported loudly and the position stays open - it is
  NOT silently marked closed.

Every function here takes an optional `client` so tests can drive the full
lifecycle against a fake exchange without network or credentials.
"""

import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_api
from exit_strategy import (
    EXIT_HARD_SL,
    EXIT_TARGET,
    EXIT_BE_RATCHET,
)


# ---------------------------------------------------------------------------
# A tiny default client wrapper (lets tests inject a fake)
# ---------------------------------------------------------------------------

class RealClient:
    """Thin adapter so callers use one object instead of module-level calls."""

    def __init__(self, creds: Optional[Tuple[str, str]] = None):
        self.creds = creds

    def creds_or_load(self):
        if self.creds is None:
            self.creds = live_api.load_credentials()
        return self.creds

    def get_info(self):
        return live_api.get_info(creds=self.creds_or_load())

    def open_orders(self, pair=None):
        return live_api.open_orders(pair=pair, creds=self.creds_or_load())

    def get_order(self, pair, order_id):
        return live_api.get_order(pair, order_id, creds=self.creds_or_load())

    def trade(self, **kw):
        return live_api.trade(creds=self.creds_or_load(), **kw)

    def cancel_order(self, **kw):
        return live_api.cancel_order(creds=self.creds_or_load(), **kw)


# ---------------------------------------------------------------------------
# Balance / preflight
# ---------------------------------------------------------------------------

def read_balance_idr(client=None) -> float:
    """Spendable IDR, as a float.

    Indodax reports balances as strings keyed by lowercase currency. 'idr' is
    the spendable amount; 'hold' is locked in open orders and is NOT spendable.
    Reading the wrong one makes the bot think it has money tied up in orders.
    """
    client = client or RealClient()
    info = client.get_info()
    ret = info.get("return") or {}
    bal = ret.get("balance") or {}
    raw = bal.get("idr")
    if raw is None:
        raise live_api.LiveApiError("getInfo returned no IDR balance")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise live_api.LiveApiError(f"IDR balance was not a number: {raw!r}")


def read_coin_balance(symbol: str, client=None) -> float:
    """Coin held (spendable only). 'symbol' is like 'btc' for pair 'btcidr'."""
    client = client or RealClient()
    info = client.get_info()
    ret = info.get("return") or {}
    bal = ret.get("balance") or {}
    coin = symbol.lower()
    if coin.endswith("idr"):
        coin = coin[:-3]
    raw = bal.get(coin)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def preflight(client=None, trade_size_idr: float = 0.0,
              key_path: Optional[str] = None) -> Dict[str, Any]:
    """Everything that must be true before a live order is allowed.

    Returns {"ok": bool, "checks": [(name, ok, detail)], "problems": [...]}.

    This is deliberately the ONLY gate between the UI and real money, so it
    checks facts rather than assumptions: the key signs, the account is not
    frozen, the balance is real, the size clears the exchange minimum, and the
    local clock is not so far off that signing will fail intermittently.
    """
    client = client or RealClient()
    checks = []
    problems = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
        if not ok:
            problems.append(f"{name}: {detail}")

    # 1. credentials load
    try:
        client.creds_or_load()
        add("credentials", True, "key and secret loaded")
    except Exception as e:                                   # noqa: BLE001
        add("credentials", False, str(e))
        return {"ok": False, "checks": checks, "problems": problems}

    # 2. the key actually signs and the account answers
    try:
        info = client.get_info()
        add("authentication", True, "getInfo accepted the signature")
    except Exception as e:                                   # noqa: BLE001
        add("authentication", False, str(e)[:400])
        return {"ok": False, "checks": checks, "problems": problems}

    ret = info.get("return") or {}

    # 3. server time vs local clock - the single most common signing failure
    try:
        server_ms = int(ret.get("server_time", 0))
        if server_ms > 0:
            drift = abs(int(time.time() * 1000) - server_ms) / 1000.0
            add("clock", drift < 5.0, f"local clock is {drift:.1f}s from the exchange")
        else:
            add("clock", True, "exchange did not report server_time; skipped")
    except Exception as e:                                   # noqa: BLE001
        add("clock", False, str(e))

    # 4. account not frozen
    frozen = str(ret.get("frozen", "0") or "0")
    add("account not frozen", frozen in ("0", "", "False", "false"), f"frozen={frozen}")

    # 5. real IDR balance
    try:
        bal = read_balance_idr(client)
        add("balance readable", True, f"IDR {bal:,.0f}")
        if trade_size_idr:
            ok = bal >= trade_size_idr
            add("balance covers one trade", ok,
                f"IDR {bal:,.0f} available, trade needs IDR {trade_size_idr:,.0f}")
    except Exception as e:                                   # noqa: BLE001
        add("balance readable", False, str(e))

    # 6. exchange minimum
    if trade_size_idr:
        add("above exchange minimum", trade_size_idr >= 10000.0,
            f"IDR {trade_size_idr:,.0f} vs the 10,000 IDR minimum")

    # 7. key file permissions - a WARNING, not a blocker.
    # A world-readable trading key on a shared machine is a real risk, but
    # refusing to trade over a file mode would block a legitimate operator for
    # a fixable nuisance. It is surfaced loudly instead, and named in the
    # START LIVE confirmation.
    warnings = []
    try:
        kp = key_path
        if kp is None:
            from paths import config_root
            kp = os.path.join(str(config_root()), "indodax_api.txt")
        warn = live_api._permissions_warning(kp)
        if warn:
            warnings.append(warn)
        add("credential file permissions readable", True,
            warn or "not readable by other users")
    except Exception:                                        # noqa: BLE001
        pass

    return {"ok": not problems, "checks": checks, "problems": problems,
            "warnings": warnings}


# ---------------------------------------------------------------------------
# Buy
# ---------------------------------------------------------------------------

def create_live_buy_order(symbol: str, limit_price: float, coin_amount: float,
                          trade_size_idr: float, pair_info: dict,
                          now: Optional[float] = None,
                          hunt_id: Optional[str] = None,
                          entry_bid: Optional[float] = None,
                          entry_ask: Optional[float] = None,
                          qty_increment: float = 0.0,
                          client=None) -> dict:
    """Place a REAL resting limit buy and return the order record.

    Keeps the same shape as create_paper_buy_order so the engine does not branch,
    plus the exchange's own order id so the order can be reconciled later.
    """
    client = client or RealClient()
    if now is None:
        now = time.time()

    pair_id = pair_info.get("id", symbol.lower())
    resp = client.trade(pair=pair_id, side="buy", idr=trade_size_idr,
                        price=limit_price)
    ret = resp.get("return") or {}
    order_id = str(ret.get("order_id") or ret.get("id") or "")
    if not order_id:
        raise live_api.LiveApiError(
            f"the exchange accepted the order but returned no order id: "
            f"{str(resp)[:200]}")

    return {
        "order_id": order_id,
        "exchange": "indodax",
        "live": True,
        "hunt_id": hunt_id,
        "symbol": symbol,
        "pair_id": pair_id,
        "side": "buy",
        "limit_price": float(limit_price),
        "coin_amount": float(coin_amount),
        "trade_size_idr": float(trade_size_idr),
        "placed_at": now,
        "checks_elapsed": 0,
        "tick": float(pair_info.get("tick", 1.0)),
        "maker_fee_pct": float(pair_info.get("trade_fee_percent_maker", 0.1)),
        "taker_fee_pct": float(pair_info.get("trade_fee_percent_taker", 0.2)),
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "qty_increment": qty_increment,
        # filled amounts are discovered from the exchange, never assumed
        "filled_coins": 0.0,
        "filled_idr": 0.0,
    }


def check_live_buy_fill(order: dict, live_price: float,
                        now: Optional[float] = None,
                        client=None) -> Tuple[bool, Optional[dict]]:
    """Ask the exchange whether the resting buy filled.

    Returns (filled, position).

    Unlike paper, a partial fill is a real state: the position is opened with
    the coins actually received. If the order is still open and nothing has
    filled, it is not a position yet.
    """
    client = client or RealClient()
    if now is None:
        now = time.time()

    detail = client.get_order(order["pair_id"], order["order_id"])
    ret = detail.get("return") or {}
    order_ret = ret.get("order") or ret

    status = str(order_ret.get("status") or "").lower()
    # Indodax reports filled amounts in several shapes depending on the endpoint
    filled_coins = _f(order_ret.get("filled_coin") or order_ret.get("coin_received"))
    filled_idr = _f(order_ret.get("filled_idr") or order_ret.get("idr_spent")
                    or order_ret.get("price"))

    if status in ("filled", "closed") and filled_coins > 0:
        return True, _make_position(order, filled_coins, filled_idr, now)

    if filled_coins > 0 and status in ("open", "partially_filled", ""):
        # A partial fill is a live position with fewer coins than planned.
        return True, _make_position(order, filled_coins, filled_idr, now,
                                    partial=True)

    return False, None


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _make_position(order: dict, coins: float, idr_spent: float,
                   now: float, partial: bool = False) -> dict:
    """Build the position dict the engine expects, with the coins actually held."""
    entry = (idr_spent / coins) if coins > 0 and idr_spent > 0 else order["limit_price"]
    return {
        "hunt_id": order.get("hunt_id"),
        "symbol": order["symbol"],
        "pair_id": order["pair_id"],
        "entry_price": entry,
        "coins": coins,
        "entry_time": now,
        "order_id": order["order_id"],
        "filled_idr": idr_spent,
        "partial_fill": bool(partial),
        "maker_fee_pct": order.get("maker_fee_pct", 0.1),
        "taker_fee_pct": order.get("taker_fee_pct", 0.2),
        "fees_paid_idr": order.get("fees_paid_idr", 0.0),
        "entry_bid": order.get("entry_bid"),
        "entry_ask": order.get("entry_ask"),
        "framework": order.get("framework"),
        "framework_version": order.get("framework_version"),
        "exit_levels": order.get("exit_levels"),
    }


# ---------------------------------------------------------------------------
# Sell
# ---------------------------------------------------------------------------

def execute_live_sell(position: dict, best_bid: float, exit_reason: str,
                      now: Optional[float] = None,
                      client=None,
                      limit_price: Optional[float] = None) -> Tuple[dict, dict]:
    """Sell the position for real and return (trade_record, summary).

    A TARGET exit rests a LIMIT sell at the target (maker fee). Every other exit
    is a MARKET sell and takes whatever the book gives - which is exactly why
    the hard stop can fill well below its trigger.

    If the sell is rejected the position is NOT reported as closed: the exception
    propagates so the engine keeps the position open and the operator sees it.
    """
    client = client or RealClient()
    if now is None:
        now = time.time()

    coins = float(position["coins"])
    pair_id = position.get("pair_id", position["symbol"].lower())

    if exit_reason == EXIT_TARGET and limit_price is not None:
        resp = client.trade(pair=pair_id, side="sell", coin=coins,
                            price=limit_price)
        fee_pct = float(position.get("maker_fee_pct", 0.1))
    else:
        resp = client.trade(pair=pair_id, side="sell", coin=coins)
        fee_pct = float(position.get("taker_fee_pct", 0.2))

    ret = resp.get("return") or {}
    order_id = str(ret.get("order_id") or ret.get("id") or "")

    # For a market sell Indodax fills immediately; the received IDR is in the
    # response when present. If it is not, we do not invent a price - the caller
    # must reconcile from getOrder. Falling back to best_bid would record a
    # number the exchange may never have given.
    received = _f(ret.get("receive") or ret.get("idr_received")
                  or ret.get("filled_idr"))
    filled_coins = _f(ret.get("filled_coin")) or coins

    if received > 0:
        exit_price = received / filled_coins if filled_coins else 0.0
        pending = False
    else:
        exit_price = 0.0
        pending = True

    gross = received if received > 0 else 0.0
    fees = gross * fee_pct / 100.0 if gross else 0.0
    net = gross - fees
    cost = float(position.get("filled_idr")
                 or position["entry_price"] * coins)
    net_pnl_idr = net - cost
    net_pnl_pct = (net_pnl_idr / cost * 100.0) if cost else 0.0

    trade_record = {
        "hunt_id": position.get("hunt_id"),
        "symbol": position["symbol"],
        "pair_id": pair_id,
        "side": "sell",
        "live": True,
        "order_id": order_id,
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "quantity": coins,
        "gross_pnl_idr": gross - cost,
        "fees_idr": fees,
        "net_pnl_idr": net_pnl_idr,
        "net_pnl_pct": net_pnl_pct,
        "entry_time": position.get("entry_time"),
        "exit_time": now,
        "exit_reason": exit_reason,
        "entry_bid": position.get("entry_bid"),
        "entry_ask": position.get("entry_ask"),
        "exit_bid": best_bid,
        "exit_ask": None,
        "framework": position.get("framework", "A"),
        "framework_version": position.get("framework_version"),
        "target_pct": (position.get("exit_levels") or {}).get("target_pct"),
        "stop_pct": (position.get("exit_levels") or {}).get("stop_pct"),
        "ratchet_armed": (position.get("exit_levels") or {}).get("ratchet_armed", False),
        "reconciled": (not pending),
    }

    summary = {
        "net_proceeds": net,
        "exit_price": exit_price,
        "coins_sold": filled_coins,
        "pending_reconciliation": pending,
    }
    return trade_record, summary


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile(client=None) -> Dict[str, Any]:
    """Compare the exchange's view of the account with what the bot believes.

    Called before arming LIVE and after any crash. Returns the facts; deciding
    what to do about a mismatch is the caller's job, but a mismatch must block
    live trading rather than be averaged away.
    """
    client = client or RealClient()
    out = {"ok": True, "idr": None, "balances": {}, "open_orders": [], "problems": []}
    try:
        info = client.get_info()
        ret = info.get("return") or {}
        bal = ret.get("balance") or {}
        out["idr"] = _f(bal.get("idr"))
        out["balances"] = {k: _f(v) for k, v in bal.items() if _f(v) > 0}
        hold = ret.get("balance_hold") or {}
        out["hold"] = {k: _f(v) for k, v in hold.items() if _f(v) > 0}
    except Exception as e:                                   # noqa: BLE001
        out["ok"] = False
        out["problems"].append(f"cannot read balances: {str(e)[:200]}")
        return out

    try:
        oo = client.open_orders()
        orders = (oo.get("return") or {}).get("orders") or []
        if isinstance(orders, dict):
            orders = [dict(v, pair=k) for k, v in orders.items()]
        for o in orders:
            out["open_orders"].append({
                "order_id": str(o.get("order_id") or o.get("id") or ""),
                "type": o.get("type"),
                "pair": o.get("pair"),
                "price": _f(o.get("price")),
                "remaining": _f(o.get("remain_coin") or o.get("orig_coin")),
            })
    except Exception as e:                                   # noqa: BLE001
        out["problems"].append(f"cannot list open orders: {str(e)[:200]}")
        out["ok"] = False

    return out


def cancel_all_open_orders(client=None) -> Dict[str, Any]:
    """Cancel every resting order. Used by STOP LIVE before halting.

    An orphaned resting order is a position you do not know you own, so STOP
    cancels first and reports anything it could not cancel.
    """
    client = client or RealClient()
    rec = reconcile(client)
    cancelled, failed = [], []
    for o in rec.get("open_orders", []):
        try:
            client.cancel_order(pair=o["pair"], order_id=o["order_id"],
                                kind=(o.get("type") or "buy"))
            cancelled.append(o)
        except Exception as e:                               # noqa: BLE001
            failed.append({"order": o, "error": str(e)[:200]})
    return {"cancelled": cancelled, "failed": failed,
            "ok": not failed}


# ---------------------------------------------------------------------------
# Self-check (a fake exchange, no network, no credentials)
# ---------------------------------------------------------------------------

class FakeExchange:
    """In-memory exchange that behaves like Indodax for the paths we exercise."""

    def __init__(self, idr=500000.0, fill_mode="full"):
        self.idr = idr
        self.coins = {}
        self.orders = {}
        self.fill_mode = fill_mode
        self.calls = []
        self._n = 0

    def creds_or_load(self):
        return ("fakekey", "fakesecret")

    def get_info(self):
        return {"success": 1, "return": {
            "balance": {"idr": f"{self.idr:.8f}", **self.coins},
            "balance_hold": {"idr": "0"},
            "server_time": int(time.time() * 1000),
            "frozen": "0",
        }}

    def open_orders(self, pair=None):
        return {"success": 1, "return": {"orders": [
            dict(o, order_id=k) for k, o in self.orders.items()
            if o.get("status") == "open" and (pair is None or o["pair"] == pair)
        ]}}

    def get_order(self, pair, order_id):
        o = self.orders[order_id]
        return {"success": 1, "return": {"order": {
            "order_id": order_id,
            "status": o["status"],
            "filled_coin": f"{o.get('filled_coin', 0):.8f}",
            "filled_idr": f"{o.get('filled_idr', 0):.8f}",
        }}}

    def trade(self, pair, side, idr=None, coin=None, price=None):
        self.calls.append({"pair": pair, "side": side, "idr": idr,
                           "coin": coin, "price": price})
        self._n += 1
        oid = f"live-{self._n}"
        if side == "buy":
            self.idr -= float(idr)
            # In "full" mode the limit price is treated as already touched, so
            # the order fills immediately. In "resting" mode it sits on the book
            # until the test moves it.
            if self.fill_mode == "full":
                got = float(idr) / float(price or 1000)
                self.coins[pair[:-3]] = self.coins.get(pair[:-3], 0) + got
                self.orders[oid] = {"pair": pair, "status": "filled",
                                    "filled_coin": got, "filled_idr": float(idr)}
                return {"success": 1, "return": {"order_id": oid,
                                                 "receive_coin": f"{got:.8f}"}}
            self.orders[oid] = {"pair": pair, "status": "open",
                                "filled_coin": 0, "filled_idr": 0}
            return {"success": 1, "return": {"order_id": oid}}
        # sell
        got_coin = float(coin)
        self.coins[pair[:-3]] = self.coins.get(pair[:-3], 0) - got_coin
        proceeds = got_coin * float(price or 1000)
        self.idr += proceeds
        self.orders[oid] = {"pair": pair, "status": "filled",
                            "filled_coin": got_coin, "filled_idr": proceeds}
        return {"success": 1, "return": {"order_id": oid,
                                         "receive": f"{proceeds:.8f}"}}

    def cancel_order(self, pair, order_id, kind="buy"):
        self.orders[order_id]["status"] = "cancelled"
        return {"success": 1, "return": {"order_id": order_id}}


def _self_check():
    print("=== executor_live.py self-check ===\n")

    pair_info = {"id": "btcidr", "tick": 1.0,
                 "trade_fee_percent_maker": 0.1, "trade_fee_percent_taker": 0.2}

    # --- 1. a full fill becomes a position with the coins actually received ---
    ex = FakeExchange(idr=100000.0)
    order = create_live_buy_order("BTCIDR", 1000.0, 0.1, 100.0, pair_info,
                                  now=1.0, hunt_id="H-L1", client=ex)
    assert order["live"] is True and order["order_id"].startswith("live-"), order
    filled, pos = check_live_buy_fill(order, 1000.0, now=2.0, client=ex)
    assert filled and pos and pos["coins"] > 0, pos
    print(f"[PASS] a real buy returns an exchange order id ({order['order_id']})")

    # --- 2. a resting order that has not filled is NOT a position -----------
    ex2 = FakeExchange(idr=100000.0, fill_mode="resting")
    o2 = create_live_buy_order("BTCIDR", 1000.0, 0.1, 100.0, pair_info,
                               now=1.0, hunt_id="H-L2", client=ex2)
    filled2, pos2 = check_live_buy_fill(o2, 1000.0, now=2.0, client=ex2)
    assert filled2 is False and pos2 is None, (filled2, pos2)
    print("[PASS] an unfilled resting order is not treated as a position")

    # --- 3. a PARTIAL fill opens a position with fewer coins ----------------
    # paper is all-or-nothing; live is not. The position must reflect what was
    # actually bought, or the stop sells coins the bot does not hold.
    ex3 = FakeExchange(idr=100000.0, fill_mode="resting")
    o3 = create_live_buy_order("BTCIDR", 1000.0, 0.1, 100.0, pair_info,
                               now=1.0, hunt_id="H-L3", client=ex3)
    ex3.orders[o3["order_id"]].update(status="partially_filled",
                                      filled_coin=0.04, filled_idr=40.0)
    filled3, pos3 = check_live_buy_fill(o3, 1000.0, now=2.0, client=ex3)
    assert filled3 and pos3["partial_fill"] is True, pos3
    assert abs(pos3["coins"] - 0.04) < 1e-9, pos3["coins"]
    assert abs(pos3["entry_price"] - 1000.0) < 1e-6, pos3["entry_price"]
    print("[PASS] a partial fill becomes a position with the actual coin amount")

    # --- 4. a market sell records the exchange's proceeds, not a guess ------
    pos4 = dict(pos3)
    pos4.update({"framework": "A", "exit_levels": {"target_pct": 0.02,
                                                   "stop_pct": 0.04},
                 "exit_bid": 990.0, "exit_ask": 1001.0})
    rec, summ = execute_live_sell(pos4, best_bid=990.0,
                                  exit_reason=EXIT_HARD_SL, now=3.0, client=ex3)
    assert rec["live"] is True and rec["reconciled"] is True, rec
    assert rec["exit_price"] > 0, rec
    assert rec["net_pnl_idr"] == summ["net_proceeds"] - pos4["filled_idr"]
    print(f"[PASS] a market sell records the real fill: exit {rec['exit_price']:.2f}, "
          f"net {rec['net_pnl_idr']:+.2f} IDR")

    # --- 5. a sell with no reported proceeds is marked unreconciled ---------
    class NoProceeds(FakeExchange):
        def trade(self, pair, side, **kw):
            self._n += 1
            return {"success": 1, "return": {"order_id": f"x{self._n}"}}

    ex5 = NoProceeds()
    rec5, s5 = execute_live_sell(dict(pos4), 990.0, EXIT_HARD_SL, now=3.0, client=ex5)
    assert s5["pending_reconciliation"] is True, s5
    assert rec5["exit_price"] == 0.0
    print("[PASS] a sell without reported proceeds is flagged, not invented")

    # --- 6. preflight blocks when the balance cannot cover the trade --------
    ex6 = FakeExchange(idr=5000.0)
    pf = preflight(client=ex6, trade_size_idr=32000.0, key_path="/tmp/none")
    assert pf["ok"] is False
    names = [c[0] for c in pf["checks"]]
    assert "balance covers one trade" in names
    assert "above exchange minimum" in names
    print(f"[PASS] preflight refuses: {pf['problems'][0][:70]}")

    # --- 7. preflight passes when everything is right -----------------------
    ex7 = FakeExchange(idr=200000.0)
    pf7 = preflight(client=ex7, trade_size_idr=32000.0, key_path="/tmp/none")
    assert pf7["ok"] is True, pf7["problems"]
    print("[PASS] preflight passes with a funded account")

    # --- 8. preflight catches a frozen account ------------------------------
    class Frozen(FakeExchange):
        def get_info(self):
            d = super().get_info()
            d["return"]["frozen"] = "1"
            return d

    pf8 = preflight(client=Frozen(idr=200000.0), trade_size_idr=32000.0, key_path="/tmp/none")
    assert pf8["ok"] is False
    assert any("frozen" in p for p in pf8["problems"])
    print("[PASS] preflight refuses a frozen account")

    # --- 9. STOP cancels resting orders first -------------------------------
    ex9 = FakeExchange(idr=100000.0, fill_mode="resting")
    create_live_buy_order("BTCIDR", 1000.0, 0.1, 100.0, pair_info, now=1.0,
                          hunt_id="H-L9", client=ex9)
    res9 = cancel_all_open_orders(client=ex9)
    assert res9["ok"] and len(res9["cancelled"]) == 1, res9
    assert all(o["status"] == "cancelled" for o in ex9.orders.values())
    print("[PASS] STOP cancels every resting order (an orphan is a position you don't know)")

    # --- 10. reconcile reports the exchange's own view ----------------------
    rec10 = reconcile(client=ex9)
    assert rec10["ok"] and rec10["idr"] is not None
    assert rec10["open_orders"] == []
    print(f"[PASS] reconcile reads the account: IDR {rec10['idr']:,.0f}, "
          f"{len(rec10['open_orders'])} open orders")

    # --- 11. a target exit uses a LIMIT sell (maker fee) --------------------
    ex11 = FakeExchange(idr=100000.0)
    pos11 = {"symbol": "BTCIDR", "pair_id": "btcidr", "coins": 0.1,
             "entry_price": 1000.0, "filled_idr": 100.0, "entry_time": 1.0,
             "maker_fee_pct": 0.1, "taker_fee_pct": 0.2,
             "exit_levels": {"target_pct": 0.02, "stop_pct": 0.04}}
    rec11, _ = execute_live_sell(pos11, 1020.0, EXIT_TARGET, now=4.0,
                                 client=ex11, limit_price=1020.0)
    assert ex11.calls[-1]["price"] == 1020.0, ex11.calls[-1]
    print("[PASS] a target exit rests a LIMIT sell at the target")

    # --- 12. a hard stop is a MARKET sell -----------------------------------
    ex12 = FakeExchange(idr=100000.0)
    execute_live_sell(dict(pos11), 960.0, EXIT_HARD_SL, now=4.0, client=ex12)
    assert ex12.calls[-1]["price"] is None, ex12.calls[-1]
    print("[PASS] a hard stop is a MARKET sell - it takes what the book gives")

    print("\n=== ALL SELF-CHECKS PASSED ===")


if __name__ == "__main__":
    _self_check()
