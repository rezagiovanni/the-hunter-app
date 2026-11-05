#!/usr/bin/env python3
"""
executor_paper.py — Simulated paper trading engine for The Hunter.

Step 6 of development_plan.txt.

Provides:
  compute_maker_limit_price() — capped maker price: min(bid + tick, ask - tick)
  compute_coin_amount()       — clean coin amount strictly quantized to increment
  validate_order_sizing()     — verifies min IDR and min coin thresholds
  create_paper_buy_order()    — creates resting limit buy order
  check_paper_buy_fill()      — simulates fill when live price touches limit
  execute_paper_sell()        — simulates market sell at best bid with taker fee
  calculate_pnl()             — honest PnL math (gross proceeds, fees, net IDR/pct)

Dependencies:
  scripts/exit_strategy.py (tick_snap, initialize_exit_levels)
"""

from decimal import Decimal, ROUND_FLOOR
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exit_strategy import (
    EXIT_HARD_SL,
    EXIT_TARGET,
    initialize_exit_levels,
    tick_snap,
)


# ---------------------------------------------------------------------------
# Order Pricing & Sizing
# ---------------------------------------------------------------------------
def compute_maker_limit_price(best_bid: float, best_ask: float, tick: float) -> float:
    """
    Entry price cap: min(best_bid + 1 tick, best_ask - 1 tick).
    Guarantees order rests as MAKER.
    On tight 0-1 tick spreads (e.g. bid == ask), ensures limit price stays
    strictly below ask, never crossing the book as taker.
    """
    raw_limit = min(best_bid + tick, best_ask - tick)
    return tick_snap(raw_limit, tick)


def compute_coin_amount(trade_size_idr: float, limit_price: float,
                        quantity_increment: float) -> float:
    """
    Compute exact coin quantity:
      coin_amount = floor(trade_size / limit_price / quantity_increment) * quantity_increment
    Uses Decimal to avoid floating point precision artifacts.
    """
    if limit_price <= 0 or trade_size_idr <= 0 or quantity_increment <= 0:
        return 0.0

    d_size = Decimal(str(trade_size_idr))
    d_price = Decimal(str(limit_price))
    d_incr = Decimal(str(quantity_increment))

    units = (d_size / d_price) // d_incr
    snapped_coin = units * d_incr
    return float(snapped_coin)


def validate_order_sizing(trade_size_idr: float, coin_amount: float,
                          pair_info: dict) -> Tuple[bool, str]:
    """
    Validate trade against exchange rules from pairs.json:
      - trade_min_base_currency   (min IDR per order, e.g. 10,000)
      - trade_min_traded_currency (min coin per order)
    """
    min_idr = float(pair_info.get("trade_min_base_currency", 10_000.0))
    min_coin = float(pair_info.get("trade_min_traded_currency", 0.0))

    if trade_size_idr < min_idr:
        return False, f"trade size {trade_size_idr:.0f} IDR < pair minimum {min_idr:.0f} IDR"

    if coin_amount < min_coin:
        return False, f"coin amount {coin_amount} < pair minimum {min_coin}"

    if coin_amount <= 0:
        return False, f"coin amount is 0 (order size too small)"

    return True, "sizing valid"


# ---------------------------------------------------------------------------
# Order Lifecycle (Paper Mode)
# ---------------------------------------------------------------------------
def create_paper_buy_order(
    symbol: str,
    limit_price: float,
    coin_amount: float,
    trade_size_idr: float,
    pair_info: dict,
    now: Optional[float] = None,
    hunt_id: Optional[str] = None,
    entry_bid: Optional[float] = None,
    entry_ask: Optional[float] = None,
    qty_increment: float = 0.0,
) -> dict:
    """Create a resting limit buy order for ORDER_OPEN phase."""
    if now is None:
        now = time.time()

    maker_fee_pct = float(pair_info.get("trade_fee_percent_maker", 0.1))
    taker_fee_pct = float(pair_info.get("trade_fee_percent_taker", 0.2))

    return {
        "order_id": f"paper-{int(now * 1000)}",
        "hunt_id": hunt_id,
        "symbol": symbol,
        "pair_id": pair_info.get("id", symbol.lower()),
        "side": "buy",
        "limit_price": limit_price,
        "coin_amount": coin_amount,
        "trade_size_idr": trade_size_idr,
        "placed_at": now,
        "checks_elapsed": 0,
        "tick": float(pair_info.get("tick", 1.0)),
        "maker_fee_pct": maker_fee_pct,
        "taker_fee_pct": taker_fee_pct,
        # Book snapshot at order placement (for later analysis)
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "qty_increment": qty_increment,
    }


def check_paper_buy_fill(order: dict, live_price: float,
                         now: Optional[float] = None,
                         bid: Optional[float] = None,
                         ask: Optional[float] = None,
                         qty_increment: float = 0.0,
                         exit_levels: Optional[dict] = None) -> Tuple[bool, Optional[dict]]:
    """
    Check if resting limit buy order fills.
    Simulated fill condition: live price touches or crosses limit_price (live_price <= limit_price).
    Returns (filled, position_dict_or_None).

    exit_levels: computed by the ACTIVE FRAMEWORK. When provided it is used
    verbatim, so the framework decides target/stop/ratchet. When omitted the
    legacy fixed-target levels are computed here (kept for older callers and
    tests).
    """
    if now is None:
        now = time.time()

    order["checks_elapsed"] = order.get("checks_elapsed", 0) + 1

    limit_price = order["limit_price"]
    if live_price <= limit_price:
        # FILLED
        coin_amount = order["coin_amount"]
        tick = order["tick"]
        cost_basis_idr = limit_price * coin_amount
        maker_fee_pct = order["maker_fee_pct"]
        entry_fee_idr = cost_basis_idr * (maker_fee_pct / 100.0)
        total_cost = cost_basis_idr + entry_fee_idr

        # Exit levels come from the framework that opened this hunt.
        if exit_levels is None:
            exit_levels = initialize_exit_levels(limit_price, tick)

        position = {
            "hunt_id": order.get("hunt_id"),
            "symbol": order["symbol"],
            "pair_id": order.get("pair_id", order["symbol"].lower()),
            "framework": exit_levels.get("framework", order.get("framework", "A")),
            "framework_version": exit_levels.get("framework_version", "?"),
            "entry_price": limit_price,
            "coins": coin_amount,
            "cost_basis_idr": round(cost_basis_idr, 2),
            "entry_fee_idr": round(entry_fee_idr, 2),
            "total_cost_idr": round(total_cost, 2),
            "maker_fee_pct": maker_fee_pct,
            "taker_fee_pct": order["taker_fee_pct"],
            "entry_time": now,
            "exit_levels": exit_levels,
            "tick": tick,
            "qty_increment": qty_increment,
            # Book snapshot at the moment of the fill
            "entry_bid": bid if bid is not None else order.get("entry_bid"),
            "entry_ask": ask if ask is not None else order.get("entry_ask"),
        }
        return True, position

    return False, None


# ---------------------------------------------------------------------------
# PnL & Sell Execution
# ---------------------------------------------------------------------------
def calculate_pnl(
    entry_price: float,
    exit_price: float,
    coin_amount: float,
    maker_fee_pct: float,
    taker_fee_pct: float
) -> dict:
    """
    Honest round-trip PnL calculation:
      cost_basis = entry_price * coins
      entry_fee  = cost_basis * (maker_fee_pct / 100)
      total_cost = cost_basis + entry_fee

      gross_proceeds = exit_price * coins
      exit_fee       = gross_proceeds * (taker_fee_pct / 100)
      net_proceeds   = gross_proceeds - exit_fee

      gross_pnl_idr  = gross_proceeds - cost_basis
      total_fees     = entry_fee + exit_fee
      net_pnl_idr    = net_proceeds - total_cost
      net_pnl_pct    = (net_pnl_idr / total_cost) * 100
    """
    cost_basis = entry_price * coin_amount
    entry_fee = cost_basis * (maker_fee_pct / 100.0)
    total_cost = cost_basis + entry_fee

    gross_proceeds = exit_price * coin_amount
    exit_fee = gross_proceeds * (taker_fee_pct / 100.0)
    net_proceeds = gross_proceeds - exit_fee

    gross_pnl_idr = gross_proceeds - cost_basis
    total_fees = entry_fee + exit_fee
    net_pnl_idr = net_proceeds - total_cost
    net_pnl_pct = (net_pnl_idr / total_cost) * 100.0

    return {
        "cost_basis": round(cost_basis, 2),
        "entry_fee": round(entry_fee, 2),
        "total_cost": round(total_cost, 2),
        "gross_proceeds": round(gross_proceeds, 2),
        "exit_fee": round(exit_fee, 2),
        "net_proceeds": round(net_proceeds, 2),
        "gross_pnl_idr": round(gross_pnl_idr, 2),
        "total_fees": round(total_fees, 2),
        "net_pnl_idr": round(net_pnl_idr, 2),
        "net_pnl_pct": round(net_pnl_pct, 4),
    }


def execute_paper_sell(
    position: dict,
    exit_price: float,
    exit_reason: str,
    now: Optional[float] = None,
    fee_override_pct: Optional[float] = None,
    exit_bid: Optional[float] = None,
    exit_ask: Optional[float] = None,
) -> Tuple[dict, dict]:
    """
    Execute a simulated sell.

    exit_reason:
      EXIT_TARGET   -> the resting LIMIT sell filled (MAKER fee)
      EXIT_HARD_SL  -> market sell at best bid (TAKER fee)

    fee_override_pct lets the caller force the fee tier (a resting limit earns
    the maker tier even though the position was exited by a sell).
    Returns (trade_record, updated_state_summary).
    """
    if now is None:
        now = time.time()

    maker_fee = position.get("maker_fee_pct", 0.1)
    taker_fee = position.get("taker_fee_pct", 0.2)

    if fee_override_pct is not None:
        # Use the override for BOTH legs so the recorded total is unambiguous
        maker_used = taker_used = fee_override_pct
    elif exit_reason == EXIT_TARGET:
        maker_used, taker_used = maker_fee, maker_fee     # resting limit = maker
    else:
        maker_used, taker_used = maker_fee, taker_fee     # market sell = taker

    pnl = calculate_pnl(
        entry_price=position["entry_price"],
        exit_price=exit_price,
        coin_amount=position["coins"],
        maker_fee_pct=maker_used,
        taker_fee_pct=taker_used,
    )

    trade_record = {
        "hunt_id": position.get("hunt_id"),
        "symbol": position["symbol"],
        "pair_id": position.get("pair_id", position["symbol"].lower()),
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "quantity": position["coins"],
        "gross_pnl_idr": pnl["gross_pnl_idr"],
        "fees_idr": pnl["total_fees"],
        "net_pnl_idr": pnl["net_pnl_idr"],
        "net_pnl_pct": pnl["net_pnl_pct"],
        "entry_time": position["entry_time"],
        "exit_time": now,
        "exit_reason": exit_reason,
        # Book snapshots — entry/exit best bid + ask, for later analysis
        "entry_bid": position.get("entry_bid"),
        "entry_ask": position.get("entry_ask"),
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
    }

    state_delta = {
        "net_proceeds": pnl["net_proceeds"],
        "net_pnl_idr": pnl["net_pnl_idr"],
    }

    return trade_record, state_delta


# ---------------------------------------------------------------------------
# Self-check (run directly: python3 executor_paper.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== executor_paper.py self-check ===\n")

    # 1. Entry price cap: guarantee MAKER order
    # Wide spread: bid=1000, ask=1005, tick=1 -> limit = min(1001, 1004) = 1001 (rests 1 tick above bid)
    p1 = compute_maker_limit_price(1000.0, 1005.0, 1.0)
    assert p1 == 1001.0, f"Expected 1001, got {p1}"

    # Tight 0-1 tick spread: bid=1000, ask=1001, tick=1 -> limit = min(1001, 1000) = 1000
    # Crucial: NEVER touches 1001 (which would fill as taker!)
    p2 = compute_maker_limit_price(1000.0, 1001.0, 1.0)
    assert p2 == 1000.0, f"Tight spread must cap at 1000, got {p2}"

    # Zero spread (SOLIDR case): bid=1000, ask=1000, tick=1 -> limit = min(1001, 999) = 999
    p3 = compute_maker_limit_price(1000.0, 1000.0, 1.0)
    assert p3 == 999.0, f"Zero spread must cap below ask at 999, got {p3}"
    print("[PASS] entry price cap: wide spread -> 1001, tight spread -> 1000, zero spread -> 999 (maker guaranteed)")

    # 2. Coin amount quantization
    # 40,000 IDR / 24,631 price / 0.0001 increment -> clean multiple
    coins = compute_coin_amount(40_000.0, 24631.0, 0.0001)
    # 40000 / 24631 = 1.623969... -> 1.6239
    assert coins == 1.6239
    assert Decimal(str(coins)) % Decimal("0.0001") == Decimal("0.0")
    print(f"[PASS] coin amount quantization: 40,000 / 24631 with 0.0001 incr = {coins}")

    # 3. Order sizing validation
    mock_pair_info = {
        "id": "btcidr",
        "trade_min_base_currency": 10000.0,
        "trade_min_traded_currency": 0.0001,
        "trade_fee_percent_maker": 0.1,
        "trade_fee_percent_taker": 0.2,
        "tick": 1000.0,
    }
    # Below minimum IDR (5,000 IDR < 10,000 IDR) -> REJECT
    ok, detail = validate_order_sizing(5_000.0, 0.001, mock_pair_info)
    assert not ok and "trade size" in detail
    # Below minimum coin (0.00005 < 0.0001) -> REJECT
    ok, detail = validate_order_sizing(40_000.0, 0.00005, mock_pair_info)
    assert not ok and "coin amount" in detail
    # Valid trade -> PASS
    ok, detail = validate_order_sizing(40_000.0, 0.001, mock_pair_info)
    assert ok
    print("[PASS] order sizing validation: below IDR rejected, below coin rejected, valid passes")

    # 4. Design Scenario PnL Match: ~+410 IDR (+1.02% to +1.04%)
    # XRP entry: entry=24631, exit=24999, coins=1.6237, maker_fee=0.15%, taker_fee=0.30%
    # (Design scenario parameters from section 11)
    scenario_pnl = calculate_pnl(
        entry_price=24631.0,
        exit_price=24999.0,
        coin_amount=1.6237,
        maker_fee_pct=0.15,
        taker_fee_pct=0.30,
    )
    # The design narrative rounded cost_basis to 40,060 (giving exactly +410 IDR / +1.02%).
    # Exact floating point math gives +415.76 IDR / +1.04%.
    assert abs(scenario_pnl["net_pnl_idr"] - 415.76) < 0.1, f"Expected ~415.76 IDR, got {scenario_pnl['net_pnl_idr']}"
    assert abs(scenario_pnl["net_pnl_pct"] - 1.04) < 0.05, f"Expected ~1.04%, got {scenario_pnl['net_pnl_pct']}"
    print(f"[PASS] design scenario PnL match: exact net_pnl_idr=+{scenario_pnl['net_pnl_idr']:.2f} IDR (+{scenario_pnl['net_pnl_pct']:.2f}%), narrative rounded ~+410 IDR (+1.02%)")

    # 5. Order fill & sell execution lifecycle (fixed-target exit)
    # XRP here uses tick=1 so the +2% / -4% levels land on clean integers.
    xrp_info = dict(mock_pair_info)
    xrp_info["tick"] = 1.0
    order = create_paper_buy_order(
        "XRPIDR", 24631.0, 1.6237, 40_000.0, xrp_info,
        hunt_id="H-0001", entry_bid=24630.0, entry_ask=24632.0, qty_increment=0.0001,
    )
    # Price still above limit -> not filled
    filled, pos = check_paper_buy_fill(order, live_price=24700.0)
    assert not filled and pos is None

    # Price dips to limit -> filled
    filled, pos = check_paper_buy_fill(order, live_price=24631.0,
                                       bid=24630.0, ask=24632.0, qty_increment=0.0001)
    assert filled and pos is not None
    assert pos["entry_price"] == 24631.0
    assert pos["hunt_id"] == "H-0001"
    assert pos["exit_levels"]["target_price"] == 25123.0      # +2% floored
    assert pos["exit_levels"]["stop_trigger_price"] == 23645.0  # -4% floored
    assert pos["entry_bid"] == 24630.0 and pos["entry_ask"] == 24632.0

    # Exit at the TARGET -> resting limit fills at its own price, MAKER fee
    trade_rec, delta = execute_paper_sell(
        pos, exit_price=pos["exit_levels"]["target_price"],
        exit_reason=EXIT_TARGET, exit_bid=25860.0, exit_ask=25864.0,
    )
    assert trade_rec["symbol"] == "XRPIDR"
    assert trade_rec["hunt_id"] == "H-0001"
    assert trade_rec["net_pnl_idr"] > 0
    assert delta["net_proceeds"] > 0
    assert trade_rec["exit_bid"] == 25860.0 and trade_rec["exit_ask"] == 25864.0
    print(f"[PASS] TARGET exit: fill {trade_rec['exit_price']}, net +{trade_rec['net_pnl_idr']:.2f} IDR "
          f"(+{trade_rec['net_pnl_pct']:.2f}%), book captured")

    # The target exit must beat the same trade exiting via a MARKET sell at the
    # same price, because a resting limit pays the maker tier, not the taker tier.
    mkt, _ = execute_paper_sell(
        pos, exit_price=pos["exit_levels"]["target_price"], exit_reason=EXIT_HARD_SL,
    )
    assert trade_rec["net_pnl_idr"] > mkt["net_pnl_idr"], (trade_rec["net_pnl_idr"], mkt["net_pnl_idr"])
    print(f"[PASS] fee tier: limit exit +{trade_rec['net_pnl_idr']:.2f} > market exit "
          f"+{mkt['net_pnl_idr']:.2f} IDR on the same price (maker vs taker)")

    # Stop exit: market sell, and the caller can override the fill price
    stop_fill = 24631.0 * (1.0 - 0.04 - 0.0094)
    stop_rec, _ = execute_paper_sell(pos, exit_price=stop_fill, exit_reason=EXIT_HARD_SL)
    assert stop_rec["exit_reason"] == EXIT_HARD_SL
    assert stop_rec["net_pnl_idr"] < 0
    print(f"[PASS] STOP exit: fill {stop_fill:,.2f} -> net {stop_rec['net_pnl_idr']:,.2f} IDR "
          f"({stop_rec['net_pnl_pct']:.2f}%)")

    print("\n=== ALL CHECKS PASSED ===")
