#!/usr/bin/env python3
"""
market_analysis.py — market-wide analytics computed from SCRAPED candles.

The Analytics page used to show only what the bot traded. This module adds
the other half: what the whole market is doing, computed from the same
ohlcv/ CSVs the scrapers already write.

Three reports, all pure computation over stored candles (no network):

  market_pulse()     - 24h movers, breadth, median spread proxy
  volatility_board() - ATR% ranking: which pairs actually move enough
                       to reach a +2% / +5% target
  signal_sweep()     - run the ACTIVE framework's entry logic across
                       every pair with candles: "if the engine ticked
                       right now, who qualifies and why"

Cost control: only the last ~250 bars of each CSV are read. Parsing all
136 files fully took 32s; the tail reader does the same sweep in under a
second, so the page can compute on load instead of needing a cron.
"""

import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths                                    # noqa: E402

BAR_SECONDS = 900          # candles are 15-minute bars
TAIL_BARS = 250            # enough for EMA21/RSI14/ATR14 + 24h lookback


# ---------------------------------------------------------------------------
# Tail reader - the whole point of this module's speed
# ---------------------------------------------------------------------------
def read_tail(path, n=TAIL_BARS):
    """Last n rows of a candle CSV without parsing the whole file.

    Seeks to a byte window near EOF and parses only that. The first line of
    the window may be cut mid-row; it is dropped unless it is the header.
    Returns [] for a missing/empty file - "no history yet", not an error.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return []

    win = 220 * n                        # rows are ~55-90 bytes; big slack
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - win))
            chunk = f.read().decode("utf-8", "ignore")
    except OSError:
        return []

    lines = [l for l in chunk.splitlines() if l.strip()]
    if size > win and lines:
        lines = lines[1:]                # possibly-truncated first row
    if lines and lines[0].startswith("timestamp"):
        lines = lines[1:]
    if not lines:
        return []

    rows = []
    reader = csv.DictReader(
        io.StringIO("timestamp,open,high,low,close,volume\n" + "\n".join(lines)))
    for r in reader:
        try:
            rows.append({
                "timestamp": int(r["timestamp"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            })
        except (KeyError, TypeError, ValueError):
            continue                     # skip a malformed row, keep the rest
    rows.sort(key=lambda r: r["timestamp"])
    return rows[-n:]


def _candles(rows):
    """ohlcv_io row dicts -> the o/h/l/c/v shape filters/indicators expect."""
    return [{"t": r["timestamp"], "o": r["open"], "h": r["high"],
             "l": r["low"], "c": r["close"], "v": r["volume"]} for r in rows]


def _ohlcv_dir():
    return os.path.join(paths.resolve_data_dir(), "ohlcv")


def _pairs(limit=100):
    """Every pair that has a candle file, newest first, capped at `limit`."""
    d = _ohlcv_dir()
    try:
        files = [f for f in os.listdir(d) if f.endswith(".csv")]
    except OSError:
        return []
    out = []
    for f in files:
        try:
            out.append((os.path.getmtime(os.path.join(d, f)), f[:-4]))
        except OSError:
            continue
    out.sort(reverse=True)               # freshest first
    return [name for _, name in out[:limit]]


def _last_price(candles):
    return float(candles[-1]["c"]) if candles else None


def _pct_change_24h(candles):
    """Close now vs ~24h ago, from 15m bars (96 bars). None if not enough."""
    if len(candles) < 96:
        return None
    then = float(candles[-96]["c"])
    now = float(candles[-1]["c"])
    if not then:
        return None
    return (now - then) / then * 100.0


def _atr_pct(candles, period=14):
    """Average True Range as % of last close - volatility in one number."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(len(candles) - period, len(candles)):
        h = float(candles[i]["h"])
        l = float(candles[i]["l"])
        pc = float(candles[i - 1]["c"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    close = float(candles[-1]["c"])
    if not close:
        return None
    return sum(trs) / len(trs) / close * 100.0


def _spread_proxy_pct(candles):
    """Intraday (h-l)/c of the last bar - a crude but uniform spread proxy."""
    if not candles:
        return None
    h = float(candles[-1]["h"])
    l = float(candles[-1]["l"])
    c = float(candles[-1]["c"])
    if not c:
        return None
    return (h - l) / c * 100.0


# ---------------------------------------------------------------------------
# Report 1: market pulse
# ---------------------------------------------------------------------------
def market_pulse(limit=100):
    """24h movers, breadth and spread picture across all scraped pairs."""
    movers = []
    spreads = []
    up = dn = flat = 0
    last_seen_ts = None

    for pair in _pairs(limit):
        rows = read_tail(os.path.join(_ohlcv_dir(), pair + ".csv"))
        candles = _candles(rows)
        if not candles:
            continue
        last_seen_ts = candles[-1]["t"]
        chg = _pct_change_24h(candles)
        sp = _spread_proxy_pct(candles)
        if sp is not None:
            spreads.append(sp)
        if chg is None:
            flat += 1
            continue
        if chg > 0.5:
            up += 1
        elif chg < -0.5:
            dn += 1
        else:
            flat += 1
        movers.append({
            "pair": pair,
            "change_24h": round(chg, 2),
            "last": _last_price(candles),
            "atr_pct": _atr_pct(candles),
        })

    movers.sort(key=lambda m: m["change_24h"], reverse=True)
    total_scored = up + dn + flat
    return {
        "gainers": movers[:8],
        "losers": list(reversed(movers[-8:])),
        "breadth_up": up,
        "breadth_down": dn,
        "breadth_flat": flat,
        "scored": total_scored,
        "median_spread_pct": (sorted(spreads)[len(spreads) // 2]
                              if spreads else None),
        "generated_at": last_seen_ts,
    }


# ---------------------------------------------------------------------------
# Report 2: volatility board
# ---------------------------------------------------------------------------
def volatility_board(limit=100, top=15):
    """ATR% ranking - which pairs move enough to reach a target at all."""
    out = []
    for pair in _pairs(limit):
        rows = read_tail(os.path.join(_ohlcv_dir(), pair + ".csv"))
        candles = _candles(rows)
        if not candles:
            continue
        atr = _atr_pct(candles)
        if atr is None:
            continue
        chg = _pct_change_24h(candles)
        out.append({
            "pair": pair,
            "atr_pct": round(atr, 3),
            "last": _last_price(candles),
            "change_24h": chg,
        })
    out.sort(key=lambda r: r["atr_pct"], reverse=True)
    return out[:top]


# ---------------------------------------------------------------------------
# Report 3: signal sweep
# ---------------------------------------------------------------------------
def signal_sweep(limit=100, top=20):
    """Run the ACTIVE framework's entry logic over every scraped pair.

    Answers: "if the engine ticked right now, who qualifies and why?"
    Uses the framework's own evaluate_entry() so the answer is the engine's
    answer, not an approximation.
    """
    from control import load_control
    from frameworks import get_framework

    try:
        control = load_control() or {}
    except Exception:                                   # noqa: BLE001
        control = {}
    name = control.get("framework", "A")
    try:
        fw = get_framework(name)
    except Exception:                                   # noqa: BLE001
        fw = get_framework("A")

    out = []
    for pair in _pairs(limit):
        rows = read_tail(os.path.join(_ohlcv_dir(), pair + ".csv"))
        candles = _candles(rows)
        if not candles:
            continue
        try:
            r = fw.evaluate_entry(candles, {})
        except Exception:                               # noqa: BLE001
            continue
        out.append({
            "pair": pair,
            "eligible": bool(r.get("eligible")),
            "total": r.get("total"),
            "direction": r.get("direction"),
            "reasons": r.get("reasons") or [],
            "rsi": ((r.get("meta") or {}).get("momentum_details") or {}).get("rsi"),
            "last": _last_price(candles),
        })

    eligible = [r for r in out if r["eligible"]]
    eligible.sort(key=lambda r: (r.get("total") or 0), reverse=True)
    near = [r for r in out if not r["eligible"]
            and (r.get("total") or 0) >= 40]
    near.sort(key=lambda r: (r.get("total") or 0), reverse=True)
    return {
        "framework": fw.name,
        "framework_label": fw.label,
        "eligible": eligible[:top],
        "near_misses": near[:10],
        "scanned": len(out),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    import time

    failures = []

    def check(label, cond):
        print(("[PASS] " if cond else "[FAIL] ") + label)
        if not cond:
            failures.append(label)

    print("=== market_analysis.py self-check ===\n")

    # -- tail reader against a known file --------------------------------
    d = _ohlcv_dir()
    sample = None
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            sample = os.path.join(d, f)
            break

    if sample:
        from ohlcv_io import read_csv
        full = read_csv(sample)
        tail = read_tail(sample, 50)
        check("tail reader returns the last N rows",
              len(tail) == 50 and tail == full[-50:])
        t0 = time.time()
        read_tail(sample, TAIL_BARS)
        dt = time.time() - t0
        check(f"tail read is fast ({dt*1000:.0f}ms)", dt < 1.0)

    # -- ATR on a flat synthetic series is 0 ------------------------------
    flat = [{"t": i * 900, "o": 100.0, "h": 100.0, "l": 100.0,
             "c": 100.0, "v": 1.0} for i in range(50)]
    check("ATR of a flat series is 0", _atr_pct(flat) == 0.0)
    wide = [{"t": i * 900, "o": 100.0, "h": 110.0, "l": 90.0,
             "c": 100.0, "v": 1.0} for i in range(50)]
    # every bar spans 90..110, so TR = 20 on a close of 100 = 20%
    check("ATR of a swinging series is ~20%",
          abs(_atr_pct(wide) - 20.0) < 0.01)

    # -- 24h change is None without history ------------------------------
    check("24h change needs 96 bars", _pct_change_24h(flat[:50]) is None)
    day = [{"t": i * 900, "o": 100.0, "h": 100.0, "l": 100.0,
            "c": 100.0, "v": 1.0} for i in range(100)]
    day_chg = _pct_change_24h(day)
    check("24h change of a flat series is 0",
          day_chg is not None and abs(day_chg) < 1e-9)

    # -- full sweep stays well under a page-load budget -------------------
    t0 = time.time()
    pulse = market_pulse(limit=40)
    dt = time.time() - t0
    check(f"market_pulse(40 pairs) under 2s ({dt:.2f}s)", dt < 2.0)
    check("pulse shape", all(k in pulse for k in
          ("gainers", "losers", "breadth_up", "breadth_down", "scored")))
    check("gainers sorted desc",
          all(a["change_24h"] >= b["change_24h"]
              for a, b in zip(pulse["gainers"], pulse["gainers"][1:])))
    check("losers sorted asc",
          all(a["change_24h"] <= b["change_24h"]
              for a, b in zip(pulse["losers"], pulse["losers"][1:])))

    t0 = time.time()
    vol = volatility_board(limit=40)
    dt = time.time() - t0
    check(f"volatility_board(40 pairs) under 2s ({dt:.2f}s)", dt < 2.0)
    check("volatility sorted desc",
          all(a["atr_pct"] >= b["atr_pct"] for a, b in zip(vol, vol[1:])))

    t0 = time.time()
    sweep = signal_sweep(limit=40)
    dt = time.time() - t0
    check(f"signal_sweep(40 pairs) under 3s ({dt:.2f}s)", dt < 3.0)
    check("sweep reports the framework it ran", "framework" in sweep)
    check("sweep separates eligible from misses",
          isinstance(sweep.get("eligible"), list)
          and isinstance(sweep.get("near_misses"), list))

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== ALL SELF-CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
