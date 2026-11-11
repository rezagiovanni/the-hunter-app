#!/usr/bin/env python3
"""
Dashboard for The Hunter cron progress.
Serves a simple HTML page at http://localhost:8082

Sections:
- OHLCV Data Summary (total/complete/stale cards + folder sizes)
- Candlestick Chart (pair dropdown, timeframe selector, date from/to)
- Stale Data Identification (details for stale pairs only)
- Top 20 Pairs (24h volume)

No auto-refresh — the page is static per load. Change any chart control
to re-fetch data.

Chart timeframes (aggregated from the stored 15m bars):
    15m = raw, 1h = 4 bars, 3h = 12, 6h = 24, 12h = 48, 1D = 96
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Share the engine's gate thresholds so the dashboard board never drifts
# from the live engine when constants change (e.g. spread 0.20% -> 0.50%).
try:
    from scripts.gates import MAX_SPREAD_PCT
except ImportError:
    MAX_SPREAD_PCT = 0.50  # fallback: match scripts/gates.py

# Paths come from paths.py so the app runs from wherever it is installed.
# See that module for the resolution rules; on the dev machine this resolves to
# exactly the same directories the hardcoded paths used to name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    APP_ROOT, DATA_DIR as _DATA, OHLCV_DIR as _OHLCV,
    MODEL_DIR as _MODEL, HUNT_TICKS_DIR as _TICKS, STATIC_DIR as _STATIC,
)

BASE_DIR = Path(APP_ROOT)
DATA_DIR = Path(_DATA)
OHLCV_DIR = Path(_OHLCV)
MODEL_DIR = Path(_MODEL)
HUNT_TICKS_DIR = Path(_TICKS)   # per-hunt minute ticks (last price per minute)

STALE_THRESHOLD_MINUTES = 45

# ---------------------------------------------------------------------------
# Fast CSV access
# ---------------------------------------------------------------------------
# data/ohlcv holds ~113 files / ~368 MB / ~7M bars. Reading every file in full
# to look at its LAST line took ~3s per page load, which made every refresh
# feel dead. Two cheap primitives replace that:
#   _tail_lines()        reads only the last few KB of a file
#   _count_bars_cached() counts data lines once per (size, mtime)
# ---------------------------------------------------------------------------

_TAIL_BYTES = 64 * 1024          # ample for 96 15m candles (~5 KB)
_BAR_COUNT_CACHE = {}            # path -> (size, mtime_ns, count)


def _tail_lines(path, n: int):
    """Return up to the last n complete lines of a file, without reading it all."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - _TAIL_BYTES))
        chunk = fh.read()
    text = chunk.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if len(chunk) < size and lines:
        lines = lines[1:]        # first line is probably partial
    return [ln for ln in lines if ln.strip()][-n:]


def _count_bars_cached(path) -> int:
    """
    Number of data rows in an OHLCV csv, cached per (size, mtime).

    Counting is the only genuinely expensive operation left; doing it once per
    file version rather than once per page load removes ~3s from a refresh.
    """
    try:
        st = os.stat(path)
    except OSError:
        return 0
    key = (st.st_size, st.st_mtime_ns)
    cached = _BAR_COUNT_CACHE.get(str(path))
    if cached and cached[0] == key[0] and cached[1] == key[1]:
        return cached[2]
    try:
        with open(path, "rb") as fh:
            count = sum(1 for ln in fh if ln.strip()) - 1     # minus header
        count = max(0, count)
        _BAR_COUNT_CACHE[str(path)] = (key[0], key[1], count)
        return count
    except OSError:
        return 0


MAX_CHART_POINTS = 1500   # server-side aggregation cap so charts stay renderable
WIB_OFFSET = 7 * 3600     # UTC+7 (WIB) — all display timestamps use this

# ML veto layer (optional, self-healing):
# - predictions.json is written by model/predict.py (runs under model/venv — has sklearn)
# - the dashboard only READS the JSON; if stale it spawns a background refresh
# - everything degrades gracefully to "—" if the model folder is absent
ML_MAX_AGE_SEC = 120      # refresh in background when older than this
ML_REFRESH_COOLDOWN = 45  # min seconds between refresh spawns

# Timeframe -> number of 15m bars per candle
TIMEFRAMES = {"15m": 1, "1h": 4, "3h": 12, "6h": 24, "12h": 48, "1d": 96}
DEFAULT_PAIR = "BTCIDR"
DEFAULT_TF = "15m"
DEFAULT_FROM = "2024-01-01"

app = FastAPI(title="The Hunter Dashboard")

# short-lived cache for the real balance; the exchange rate-limits
_LIVE_BAL_CACHE = {"at": 0.0, "data": None}

# Favicon / static assets (dagger icon)
STATIC_DIR = Path(_STATIC)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _fmt_epoch(ts):
    try:
        ts_int = int(ts)
        return time.strftime("%Y-%m-%d %H:%M:%S WIB", time.gmtime(ts_int + WIB_OFFSET))
    except (ValueError, TypeError):
        return str(ts)


def _live_prices_from_ticks(hunt_id, entry_price, coins, fee_pct_in=0.1, fee_pct_out=0.2):
    """
    Read the hunt's tick file and return the most recent (last, bid, ask).

    Falls back to (None, None, None) when the file has no usable rows.
    """
    if not hunt_id:
        return (None, None, None)
    path = HUNT_TICKS_DIR / f"{hunt_id}.jsonl"
    if not path.exists():
        return (None, None, None)
    last = bid = ask = None
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("last") is not None:
                    last = row.get("last")
                    bid = row.get("bid")
                    ask = row.get("ask")
    except OSError:
        return (None, None, None)
    return (last, bid, ask)


def _floating_pnl(entry_price, coins, live_price, maker_fee_pct=0.1, taker_fee_pct=0.2):
    """
    Mark-to-market PnL if the position were closed at live_price right now.

    Mirrors executor_paper.calculate_pnl so the open row and a realised trade
    record are computed the same way (same fee model, same denominator).
    Returns (net_pnl_idr, net_pnl_pct) or (None, None).
    """
    try:
        cost_basis = float(entry_price) * float(coins)
        if cost_basis <= 0:
            return (None, None)
        entry_fee = cost_basis * (maker_fee_pct / 100.0)
        total_cost = cost_basis + entry_fee
        proceeds = float(live_price) * float(coins)
        exit_fee = proceeds * (taker_fee_pct / 100.0)
        net_proceeds = proceeds - exit_fee
        net_idr = net_proceeds - total_cost
        return (net_idr, (net_idr / total_cost) * 100.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return (None, None)


def _fmt_duration(entry_ts, exit_ts):
    try:
        secs = int(float(exit_ts) - float(entry_ts))
    except (ValueError, TypeError):
        return "—"
    if secs < 0:
        return "—"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, s = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {s}s"
    return f"{s}s"


def _trade_stats_html(log, start_balance: float = 100_000.0) -> str:
    """Summary strip shown above a trade history table.

    Cards: Total Hunts · PnL · Winrate · Max Drawdown ·
           Min Duration · Max Duration · Max Profit · Max Loss
    Max drawdown = worst peak-to-trough slide of the *realized* equity
    curve (start balance + cumulative net PnL, trade by trade).
    Returns "" when there is no history to summarize.
    """
    if not log:
        return ""

    n = len(log)
    net = sum(t.get("net_pnl_idr", 0.0) or 0.0 for t in log)
    wins = sum(1 for t in log if (t.get("net_pnl_idr", 0.0) or 0.0) > 0)
    winrate = wins / n * 100.0

    # Realized equity curve in append order (append order == chronological)
    equity = start_balance
    peak = equity
    max_dd_pct = 0.0
    for t in log:
        equity += t.get("net_pnl_idr", 0.0) or 0.0
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd_pct = max(max_dd_pct, dd)

    durations = []
    for t in log:
        try:
            d = float(t.get("exit_time", 0)) - float(t.get("entry_time", 0))
            if d >= 0:
                durations.append(d)
        except (ValueError, TypeError):
            pass
    min_hold = _fmt_duration(0, min(durations)) if durations else "—"
    max_hold = _fmt_duration(0, max(durations)) if durations else "—"

    nets = [t.get("net_pnl_idr", 0.0) or 0.0 for t in log]
    best = max(nets)
    worst = min(nets)
    best_sign = "+" if best >= 0 else "-"
    worst_sign = "+" if worst >= 0 else "-"

    pnl_sign = "+" if net >= 0 else ""
    net_cls = "pass" if net >= 0 else "fail"
    wr_cls = "pass" if winrate >= 50 else "fail"

    cards = (
        '<div class="stat-card"><div class="sv num-hunts">{}</div>'
        '<div class="sl">Total Hunts</div></div>'.format(n) +
        '<div class="stat-card"><div class="sv {}">{}IDR {:,.0f}</div>'
        '<div class="sl">PnL</div></div>'.format(net_cls, pnl_sign, net) +
        '<div class="stat-card"><div class="sv {}">{:.1f}%</div>'
        '<div class="sl">Winrate</div></div>'.format(wr_cls, winrate) +
        '<div class="stat-card"><div class="sv fail">{:.1f}%</div>'
        '<div class="sl">Max Drawdown</div></div>'.format(max_dd_pct) +
        '<div class="stat-card"><div class="sv">{}</div>'
        '<div class="sl">Min Duration</div></div>'.format(min_hold) +
        '<div class="stat-card"><div class="sv">{}</div>'
        '<div class="sl">Max Duration</div></div>'.format(max_hold) +
        '<div class="stat-card"><div class="sv {}">{}IDR {:,.0f}</div>'
        '<div class="sl">Max Profit</div></div>'.format(
            "pass" if best >= 0 else "fail", best_sign, abs(best)) +
        '<div class="stat-card"><div class="sv {}">{}IDR {:,.0f}</div>'
        '<div class="sl">Max Loss</div></div>'.format(
            "pass" if worst >= 0 else "fail", worst_sign, abs(worst))
    )
    return f'<div class="stats-strip">{cards}</div>'


def get_tick_map():
    ticks = {}
    pairs_file = DATA_DIR / "pairs.json"
    if pairs_file.exists():
        try:
            with open(pairs_file) as fh:
                pairs = json.load(fh)
            if isinstance(pairs, list):
                for pair in pairs:
                    sym = str(pair.get("symbol", "")).lower()
                    if sym:
                        ticks[sym] = pair.get("tick", "")
        except Exception:
            pass
    return ticks


def get_depth_metrics(pair_id: str):
    """
    Read data/depth/<pair_id>.json and return (bid_depth_idr, ask_depth_idr, spread_pct).
    bid/ask depth = total IDR value of all book levels. spread = (ask-bid)/bid %.
    Any missing/broken file -> ('—', '—', '—').
    """
    depth_file = DATA_DIR / "depth" / f"{pair_id}.json"
    if not depth_file.exists():
        return ("—", "—", "—")
    try:
        with open(depth_file) as fh:
            depth = json.load(fh)
        buys = depth.get("buy", [])
        sells = depth.get("sell", [])
        if not buys or not sells:
            return ("—", "—", "—")
        bid_depth = sum(float(p) * float(a) for p, a in buys)
        ask_depth = sum(float(p) * float(a) for p, a in sells)
        best_bid = float(buys[0][0])
        best_ask = float(sells[0][0])
        spread = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0.0
        return (f"{bid_depth:,.0f}", f"{ask_depth:,.0f}", f"{spread:.2f}%")
    except Exception:
        return ("—", "—", "—")


def get_pairs_info(ticks):
    pairs_file = DATA_DIR / "pairs.json"
    if not pairs_file.exists():
        return "<i>Not yet created</i>"
    try:
        with open(pairs_file) as fh:
            pairs = json.load(fh)
        if not (isinstance(pairs, list) and pairs):
            return "<p>No pairs data</p>"
        html = "<table><thead><tr><th>Symbol</th><th>Volume (IDR)</th><th>Bid Depth (IDR)</th><th>Ask Depth (IDR)</th><th>Spread</th></tr></thead><tbody>"
        for pair in pairs[:20]:
            sym = pair.get("symbol", "N/A")
            vol = pair.get("vol_idr", 0)
            try:
                vol_str = f"{int(vol):,}"
            except (ValueError, TypeError):
                vol_str = str(vol)
            bid_d, ask_d, spread = get_depth_metrics(pair.get("id", ""))
            html += f"<tr><td>{sym}</td><td>{vol_str}</td><td>{bid_d}</td><td>{ask_d}</td><td>{spread}</td></tr>"
        html += "</tbody></table>"
        return html
    except Exception as exc:
        return f"<pre>Error reading pairs: {exc}</pre>"


def get_ohlcv_summary_and_stale(ticks):
    """
    Returns (summary_dict, stale_html).
    summary_dict: {total_pairs, complete_count, stale_count}
    stale_html: HTML table of only stale pairs, sorted oldest first.
    """
    if not OHLCV_DIR.exists():
        return {"total_pairs": 0, "complete_count": 0, "stale_count": 0}, "<i>No OHLCV directory</i>"
    files = sorted(OHLCV_DIR.glob("*.csv"))
    if not files:
        return {"total_pairs": 0, "complete_count": 0, "stale_count": 0}, "<i>No CSV files</i>"

    now = time.time()
    stale_rows = []
    total_pairs = len(files)
    stale_count = 0

    for f in files:
        pair = f.stem.upper()
        tick = ticks.get(pair.lower(), "")
        try:
            # Only the last row matters for staleness; bar count comes from cache.
            tail = _tail_lines(f, 1)
            bars = _count_bars_cached(f)
            if not tail:
                stale_count += 1
                stale_rows.append((pair, 99999, "N/A", "NO DATA", tick, bars))
                continue
            last_fields = tail[-1].strip().split(",")
            ts_raw = last_fields[0]
            try:
                ts_int = int(ts_raw)
            except ValueError:
                stale_count += 1
                stale_rows.append((pair, 99999, "Parse error", "ERROR", tick, bars))
                continue
            age_minutes = (now - ts_int) / 60.0
            if age_minutes > STALE_THRESHOLD_MINUTES:
                stale_count += 1
                last_str = _fmt_epoch(ts_int)
                age_str = f"{age_minutes:.0f}m" if age_minutes < 60 else f"{age_minutes/60:.1f}h"
                stale_rows.append((pair, -age_minutes, last_str, age_str, tick, bars))
        except Exception:
            stale_count += 1
            stale_rows.append((pair, 99999, "Error", "ERROR", tick, "?"))

    stale_rows.sort(key=lambda x: x[1])
    if not stale_rows:
        stale_html = "<p style='color:#4ade80; font-weight:bold;'>No stale pairs found. All OHLCV data is fresh.</p>"
    else:
        stale_html = "<table><thead><tr>"
        stale_html += "<th>Pair</th><th>Bars</th><th>Last Timestamp</th><th>Age</th></tr></thead><tbody>"
        for pair, _neg_age, last_str, age_str, tick, bars in stale_rows:
            stale_html += f"<tr><td>{pair}</td><td>{bars}</td><td>{last_str}</td><td>{age_str}</td></tr>"
        stale_html += "</tbody></table>"

    summary = {
        "total_pairs": total_pairs,
        "complete_count": total_pairs - stale_count,
        "stale_count": stale_count,
    }
    return summary, stale_html


def get_folder_size(directory: Path) -> str:
    """Return human-readable total size of a directory (recursive)."""
    try:
        if not directory.exists():
            return "—"
        total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
        for unit in ("B", "KB", "MB", "GB"):
            if total < 1024 or unit == "GB":
                if unit == "B":
                    return f"{int(total)} B"
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} GB"
    except Exception:
        return "—"


def list_ohlcv_pairs():
    """Return sorted list of pair symbols that have CSVs."""
    if not OHLCV_DIR.exists():
        return []
    return sorted(f.stem.upper() for f in OHLCV_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# Filter board — live preview of The Hunter's prey-selection funnel.
# Signal scoring here is a PREVIEW; the authoritative implementation is
# build step 3 (filters.py). Formulas mirror the design spec.
# ---------------------------------------------------------------------------

PAPER_TRADE_SIZE = 40000   # 40% of the 100k paper starting balance (design)
TRADES_PER_PAGE = 10       # trade-log pagination page size (Live + Paper)


def _read_candles(symbol: str, n: int = 96):
    """Read last n candles + total bar count from a pair CSV."""
    csv_path = OHLCV_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        return None, 0
    try:
        # Read only the tail — the full file is ~3 MB of tick history the
        # board never looks at. Total bars come from the cached counter.
        tail = _tail_lines(csv_path, n)
        rows = []
        for ln in tail:
            if ln.startswith("timestamp") or not ln.strip():
                continue
            p = ln.split(",")
            try:
                rows.append({"t": int(p[0]), "o": float(p[1]), "h": float(p[2]),
                             "l": float(p[3]), "c": float(p[4]), "v": float(p[5])})
            except (ValueError, IndexError):
                continue
        return rows, _count_bars_cached(csv_path)
    except Exception:
        return None, 0


def _ema_series(values, period):
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _rsi(closes, period=14):
    """Wilder's RSI. Returns None if not enough data."""
    if len(closes) <= period:
        return None
    ag = al = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        ag += max(d, 0)
        al += max(-d, 0)
    ag /= period
    al /= period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0)) / period
        al = (al * (period - 1) + max(-d, 0)) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _block_reason(row):
    """First failing gate, or None if all gates pass.

    Mirrors the engine's gate order: warmup, freshness, spread, depth,
    cooldown, ML veto (gate 6), then filter eligibility.
    """
    for gate, label in (("candles", "WARMUP"), ("fresh", "STALE"), ("spread", "SPREAD"),
                        ("depth", "DEPTH"), ("cooldown", "COOLDOWN")):
        if not row[gate][0]:
            return label
    if row["total"] is None:
        return "NO DATA"
    if row.get("direction") != "UP":
        return "NOT UP"
    if not row.get("not_extended", True):
        return "EXTENDED"
    if row["total"] < 60:
        return "SCORE < 60"
    return None


def build_filter_board(trade_size: float = PAPER_TRADE_SIZE):
    """
    Live preview of the Hunter's funnel for all 20 pairs.
    Returns (rows, winner_symbol, eligible_count).
    """
    pairs_file = DATA_DIR / "pairs.json"
    if not pairs_file.exists():
        return [], None, 0
    with open(pairs_file) as fh:
        pairs = json.load(fh)

    cooldowns = {}
    state_file = DATA_DIR / "hunter_state.json"
    if state_file.exists():
        try:
            with open(state_file) as fh:
                cooldowns = json.load(fh).get("cooldowns", {})
        except Exception:
            pass

    now = time.time()
    rows = []
    for p in pairs[:20]:
        sym = p["symbol"]
        row = {"pair": sym, "vol": p.get("vol_idr", 0)}

        # gate: candle history
        candles, total = _read_candles(sym, 96)
        row["candles"] = (total >= 100, f"{total:,}")

        # gate: freshness
        if candles:
            age_min = (now - candles[-1]["t"]) / 60
            row["fresh"] = (age_min <= 30, f"{age_min:.0f}m")
        else:
            row["fresh"] = (False, "—")

        # gates: spread + depth (same depth file)
        spread_pass, spread_disp = False, "—"
        depth_pass, depth_disp = False, "—"
        try:
            with open(DATA_DIR / "depth" / f"{p['id']}.json") as fh:
                depth = json.load(fh)
            buys = [[float(x[0]), float(x[1])] for x in depth.get("buy", [])]
            sells = [[float(x[0]), float(x[1])] for x in depth.get("sell", [])]
            if buys and sells:
                bid, ask = buys[0][0], sells[0][0]
                spread = (ask - bid) / bid * 100
                spread_pass = spread <= MAX_SPREAD_PCT
                spread_disp = f"{spread:.2f}%"
                ask_cover = sum(pr * am for pr, am in sells if pr <= ask * 1.02)
                bid_cover = sum(pr * am for pr, am in buys if pr >= bid * 0.98)
                depth_pass = ask_cover >= trade_size and bid_cover >= trade_size
                depth_disp = f"{min(ask_cover, bid_cover) / 1e6:.0f}M"
        except Exception:
            pass
        row["spread"] = (spread_pass, spread_disp)
        row["depth"] = (depth_pass, depth_disp)

        # gate: cooldown (from hunter state if the bot exists; else clear)
        # Engine stores cooldowns keyed by SYMBOL (e.g. "ARCIDR"); check both
        # symbol and ticker_id so the board never misses an active cooldown.
        cd_until = max(cooldowns.get(sym, 0), cooldowns.get(p.get("ticker_id", ""), 0))
        row["cooldown"] = (now >= cd_until, "4h" if cd_until else "—")

        # signals (preview)
        row["trend"] = (False, "—")
        row["momentum_disp"] = "—"
        row["total"] = None
        row["direction"] = None
        if candles and len(candles) >= 30:
            closes = [c["c"] for c in candles]
            ema9 = _ema_series(closes, 9)
            ema21 = _ema_series(closes, 21)
            if len(ema9) >= 2 and len(ema21) >= 9:
                sep = (ema9[-1] - ema21[-1]) / ema21[-1] * 100
                slope = (ema21[-1] - ema21[-9]) / ema21[-9] * 100 / 2  # %/hour
                highs = [c["h"] for c in candles]
                hh = sum(1 for i in range(max(1, len(highs) - 24), len(highs))
                         if highs[i] > highs[i - 1])
                hh_frac = hh / 24
                direction = "UP" if sep > 0 and slope > 0 else (
                    "DOWN" if sep < 0 and slope < 0 else "SIDE")
                score_a = (min(40, abs(sep) / 2 * 40)
                           + min(30, abs(slope) / 0.5 * 30)
                           + hh_frac * 30)

                rsi = _rsi(closes, 14)
                last = candles[-1]
                lows24 = min(c["l"] for c in candles)
                highs24 = max(c["h"] for c in candles)
                pos = (last["c"] - lows24) / (highs24 - lows24) if highs24 > lows24 else 0.5
                avg_vol = sum(c["v"] for c in candles[-20:]) / 20
                vol_ratio = last["v"] / avg_vol if avg_vol > 0 else 1.0

                # Recalibrated RSI curve: sweet spot 50-60, steep penalty above 65
                if rsi is None:
                    rsi_score = 50.0
                elif rsi <= 50.0:
                    rsi_score = max(0.0, 80.0 - (50.0 - rsi) * 2.0)
                elif rsi <= 60.0:
                    rsi_score = 100.0
                elif rsi <= 65.0:
                    rsi_score = max(0.0, 100.0 - (rsi - 60.0) * 8.0)
                else:
                    rsi_score = max(0.0, 40.0 - (rsi - 65.0) * 4.0)

                pos_score = (1 - pos) * 100
                vol_score = min(vol_ratio / 1.5, 1) * 100
                score_b = 0.4 * rsi_score + 0.3 * pos_score + 0.3 * vol_score
                overbought = rsi is not None and rsi > 70
                if overbought:
                    score_b *= 0.4  # heavy penalty

                dist_e9 = (last["c"] - ema9[-1]) / ema9[-1] * 100.0
                not_extended = dist_e9 <= 1.5

                row["trend"] = (direction == "UP" and not_extended, f"{direction} {score_a:.0f}" + (" !PUMP" if not not_extended else ""))
                row["momentum_disp"] = f"{score_b:.0f}" + (" !OB" if overbought else "")
                row["total"] = 0.6 * score_a + 0.4 * score_b
                row["direction"] = direction
                row["not_extended"] = not_extended
                row["dist_e9"] = dist_e9

        gates_pass = all(row[g][0] for g in ("candles", "fresh", "spread", "depth", "cooldown"))
        row["gates_pass"] = gates_pass
        row["eligible"] = (gates_pass and row["total"] is not None
                           and row["total"] >= 60 and row.get("direction") == "UP"
                           and row.get("not_extended", True))
        rows.append(row)

    rows.sort(key=lambda r: -(r["total"] if r["total"] is not None else -1))
    eligible = [r for r in rows if r["eligible"]]
    winner = max(eligible, key=lambda r: r["vol"])["pair"] if eligible else None
    return rows, winner, len(eligible)


def _cell(pass_fail, disp):
    cls = "pass" if pass_fail else "fail"
    mark = "✓" if pass_fail else "✗"
    return f'<td class="{cls}">{mark} {disp}</td>'


# ---------------------------------------------------------------------------
def _trading_log_section(state: dict) -> str:
    """Render Trading Log panel with Live Mode / Paper Mode switcher."""
    phase = state.get("phase", "HUNTING")

    # A blocked engine is not hunting. This badge is the biggest thing on the
    # panel and it read "HUNTING" for days while every tick was refused, which
    # is why the same question kept being asked: a WINNER that is never bought.
    _blocked = None
    try:
        from control import load_control
        _blocked = (load_control() or {}).get("blocked")
    except Exception:                                        # noqa: BLE001
        _blocked = None
    balance = state.get("balance_idr", 100_000.0)
    pos = state.get("position")
    order = state.get("order")
    trade_log = state.get("trade_log", [])

    # What the EXCHANGE holds. This is checked before anything else on the
    # panel, because "the bot thinks it is flat but Indodax holds coins" is the
    # state where a real position has no stop and the operator is looking at an
    # empty table.
    _exch = None
    try:
        from control import load_control as _lc2, is_live as _il2
        if _il2(_lc2()):
            import executor_live as _exl3
            _exch = _exl3.reconcile_position(state)
    except Exception:                                        # noqa: BLE001
        _exch = None

    if _exch and _exch.get("status") in ("RECOVER", "PHANTOM", "MISMATCH",
                                         "PHANTOM_DUST"):
        _held = ", ".join(f"{v:g} {k}" for k, v in
                          list((_exch.get("exchange") or {}).items())[:3]) or "nothing"
        status_badge = '<span class="badge-status blocked">⚠ HOLDING ON EXCHANGE</span>'
        status_detail = (
            f"Indodax holds <b>{_held}</b> but the bot has no matching position. "
            f"Those coins have <b>no stop and no target</b>. "
            f"Trading is halted until they are sold or adopted."
        )
    elif _blocked and not pos:
        status_badge = '<span class="badge-status blocked">✕ NOT TRADING</span>'
        status_detail = (
            "Engine is refusing to run: <b>"
            + str(_blocked.get("reason") or "blocked") + "</b>"
            + " · no orders will be placed until this is resolved"
        )
    elif pos:
        status_badge = '<span class="badge-status holding">● HOLDING</span>'
        _lv = pos.get("exit_levels", {})
        status_detail = (
            f"Active Position: <b>{pos.get('coins')} {pos.get('symbol')}</b> @ {pos.get('entry_price', 0):,} IDR · "
            f"Target: <b>{_lv.get('target_price', 0):,} IDR</b> (+{_lv.get('target_pct', 0)*100:.0f}%) · "
            f"Stop: <b>{_lv.get('stop_trigger_price', 0):,} IDR</b> (-{_lv.get('stop_pct', 0)*100:.0f}%)"
        )
    elif order:
        status_badge = '<span class="badge-status open">● ORDER OPEN</span>'
        status_detail = (
            f"Resting Limit Buy: <b>{order.get('coin_amount')} {order.get('symbol')}</b> @ {order.get('limit_price', 0):,} IDR · "
            f"Checks: {order.get('checks_elapsed', 0)}/6"
        )
    else:
        status_badge = f'<span class="badge-status hunting">● {phase}</span>'
        # The header says which ledger it is quoting. It used to say
        # "Virtual Balance" unconditionally, which is true in paper and a lie in
        # live - and the number shown was the simulated one either way.
        if _exec_is_live():
            status_detail = ("Scanning 20 IDR Pairs · "
                             "<b>LIVE</b> · real balance on the exchange panel")
        else:
            status_detail = f"Scanning 20 IDR Pairs · Virtual Balance: <b>IDR {balance:,.0f}</b> · engine ready"

    def _open_row(p, o):
        """Live position/order row (pinned above the paginated history).

        11 cells, matching the history table columns exactly:
          Hunt ID | Time | Symbol | FW | Hold | Entry | Exit | Qty | Net PnL | Return % | Exit Reason
        """
        if p:
            _lv = p.get("exit_levels", {})
            hunt_id = p.get("hunt_id")
            entry_price = p.get("entry_price", 0)
            coins = p.get("coins", 0)

            # Floating (mark-to-market) PnL from the latest recorded tick.
            live_price, _bid, _ask = _live_prices_from_ticks(hunt_id, entry_price, coins)
            maker = p.get("maker_fee_pct", 0.1)
            taker = p.get("taker_fee_pct", 0.2)
            net, pct = (None, None)
            if live_price is not None:
                net, pct = _floating_pnl(entry_price, coins, live_price, maker, taker)

            if net is None:
                pnl_cells = (
                    "<td class='num'>—</td>"
                    "<td class='num'>—</td>"
                )
            else:
                sign = "+" if net >= 0 else ""
                cls = "pass" if net >= 0 else "fail"
                pnl_cells = (
                    f"<td class='num {cls}'><b>{sign}IDR {net:,.0f}</b></td>"
                    f"<td class='num {cls}'><b>{sign}{pct:.2f}%</b></td>"
                )

            target = _lv.get("target_price", 0)
            target_txt = f"{target:,.0f}" if target else "—"
            hid_disp = hunt_id or "•"
            hid_html = (
                f"<span class='hunt-link' onclick=\"openHuntPage('{hunt_id}', this)\">{hid_disp}</span>"
                if hunt_id else hid_disp
            )
            return (
                f"<tr class='open-trade'>"
                f"<td class='num'>{hid_html}</td>"
                f"<td>OPEN NOW</td>"
                f"<td><b>{p.get('symbol', '—')}</b></td>"
                f"<td class='num'><span class='fw-tag {'fw-exp' if p.get('framework','A') != 'A' else 'fw-a'}'>{p.get('framework','A')}</span></td>"
                f"<td class='num'>{_fmt_duration(p.get('entry_time'), time.time())}</td>"
                f"<td class='num'>{entry_price:,.0f}</td>"
                f"<td class='num'>{target_txt}</td>"
                f"<td class='num'>{coins}</td>"
                f"{pnl_cells}"
                f"<td><button class='sell-now' type='button'"
                f" onclick=\"sellNow('{p.get('symbol','')}', this)\">SELL NOW</button></td></tr>"
            )
        if o:
            return (
                f"<tr class='open-trade'>"
                f"<td class='num'>{o.get('hunt_id') or '•'}</td>"
                f"<td>ORDER OPEN</td>"
                f"<td><b>{o.get('symbol', '—')}</b></td>"
                f"<td class='num'><span class='fw-tag {'fw-exp' if o.get('framework','A') != 'A' else 'fw-a'}'>{o.get('framework','A')}</span></td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>{o.get('limit_price', 0):,.0f}</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>{o.get('coin_amount', 0)}</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>—</td>"
                f"<td style='color:#fbbf24;'>AWAITING FILL {o.get('checks_elapsed', 0)}/6</td></tr>"
            )
        return ""

    open_row = _open_row(pos, order)
    paper_log = trade_log
    # Live trades are recorded separately from paper ones: the two ledgers
    # must never be summed, because one is simulated money and one is not.
    live_log = state.get("live_trade_log", [])
    paper_total = len(paper_log)
    live_total = len(live_log)

    def _fw_cell(t):
        """Framework badge for a trade row. Rows predating framework tracking
        are shown as A* — they were all opened under framework A."""
        fw_name = t.get("framework")
        if not fw_name:
            return "<td class='num'><span class='fw-tag fw-legacy' title='opened before framework tracking'>A*</span></td>"
        cls = "fw-exp" if fw_name != "A" else "fw-a"
        return f"<td class='num'><span class='fw-tag {cls}'>{fw_name}</span></td>"

    def _rows(log):
        """Render paginated history rows for one trade log (newest first).

        11 cells: Hunt ID | Time | Symbol | Framework | Hold | Entry | Exit |
                  Qty | Net PnL | Return % | Exit Reason
        """
        out = ""
        for idx, t in enumerate(reversed(log), 1):
            ts = t.get("exit_time", t.get("entry_time", 0))
            net = t.get("net_pnl_idr", 0.0) or 0.0
            pct = t.get("net_pnl_pct", 0.0) or 0.0
            sign = "+" if net >= 0 else ""
            pnl_cls = "pass" if net >= 0 else "fail"
            hidden = "" if idx <= TRADES_PER_PAGE else " style='display:none;'"
            hold = _fmt_duration(t.get("entry_time"), t.get("exit_time"))
            out += (
                f"<tr class='trade-row'{hidden}>"
                f"<td class='num'><span class='hunt-link' onclick=\"openHuntPage('{t.get('hunt_id') or ''}', this)\">{t.get('hunt_id') or '—'}</span></td>"
                f"<td>{_fmt_epoch(ts)}</td>"
                f"<td><b>{t.get('symbol', '—')}</b></td>"
                + _fw_cell(t) +
                f"<td class='num'>{hold}</td>"
                f"<td class='num'>{t.get('entry_price', 0):,.0f}</td>"
                f"<td class='num'>{t.get('exit_price', 0):,.0f}</td>"
                f"<td class='num'>{t.get('quantity', 0)}</td>"
                f"<td class='num {pnl_cls}'><b>{sign}IDR {net:,.0f}</b></td>"
                f"<td class='num {pnl_cls}'><b>{sign}{pct:.2f}%</b></td>"
                f"<td>{t.get('exit_reason', '—')}</td></tr>"
            )
        return out

    def _pager(table_id: str, count: int):
        """Pagination bar — rendered only for a table that has trades."""
        if count == 0:
            return ""
        return (
            f'<div class="pagination-wrap" data-pager="{table_id}">'
            f'<div class="pg-info">Showing <b class="pg-from">1</b>–<b class="pg-to">'
            f'{min(TRADES_PER_PAGE, count)}</b> of <b>{count}</b> trades</div>'
            f'<div class="pg-controls">'
            f'<button class="pg-btn" data-page="prev" type="button">← Prev</button>'
            f'<span class="pg-nums"></span>'
            f'<button class="pg-btn" data-page="next" type="button">Next →</button>'
            f'</div>'
            f'</div>'
        )

    paper_rows = (open_row or "") + _rows(paper_log)
    if not paper_rows.strip():
        paper_rows = (
            "<tr><td colspan='12' style='text-align:center; color:#9ca3af; padding:24px;'>"
            f"No completed round-trip trades yet. Engine phase: <b>{phase}</b>.</td></tr>"
        )

    live_rows = _rows(live_log)
    if not live_rows.strip():
        live_rows = (
            "<tr><td colspan='12' style='text-align:center; color:#9ca3af; padding:20px;'>"
            "No live trades recorded. Real capital trading disabled until paper phase validation.</td></tr>"
        )

    paper_pager = _pager("paper", paper_total)
    live_pager = _pager("live", live_total)
    # Summary strip above each history table (completed trades only)
    paper_stats = _trade_stats_html(paper_log)
    live_stats = _trade_stats_html(live_log)

    return f"""
    <div class="section trade-panel">
        <div class="trade-header">
            <div>
                <h2>Trading Log & Execution Monitor</h2>
                <div class="trade-sub">{status_badge} {status_detail}</div>
            </div>
            <div class="mode-control-wrap">
                <!-- Two tabs, and the tab IS the mode. Selecting PAPER shows the
                     paper ledger and the paper START/STOP; selecting LIVE shows
                     the live ledger and the live START/STOP. One button on
                     screen at a time, matching the tab you are on.

                     The tab therefore also selects the executor. That is safe
                     only because actually STARTING is still a separate act: the
                     tab decides which button you see, and pressing that button
                     is what arms it. Switching tabs while running is refused. -->
                <div class="mode-btn-group" id="ledgerTabs" role="tablist">
                    <button class="mode-btn active" id="tabPaper" type="button"
                            role="tab" aria-selected="true" onclick="selectMode('paper')">
                        <span class="dot paper"></span> PAPER MODE
                    </button>
                    <button class="mode-btn" id="tabLive" type="button"
                            role="tab" aria-selected="false" onclick="selectMode('live')">
                        <span class="dot live"></span> LIVE MODE
                    </button>
                </div>
            </div>
        </div>

        <!-- Hunt control: exactly ONE button, and the selected tab decides
             what it does. On PAPER it starts/stops the paper engine; on LIVE it
             arms or stops real trading.

             Starting LIVE still goes to its own endpoint with a typed
             confirmation - the shared button only chooses WHICH action, never
             skips it. -->
        <div class="ctrl-wrap" id="ctrlWrap">
            <button class="ctrl-switch" id="ctrlSwitch" type="button" onclick="onControlSwitch()">
                <span class="led"></span><span id="ctrlLabel">● HUNTING</span>
            </button>
            <span id="ctrlHint" style="font-size:0.72em; color:#6b7280;"></span>
            <span id="liveBadge"></span>
        </div>

        <!-- Paper Mode Trade Log (DEFAULT view while paper validation is active) -->
        <div id="paperTradeWrap" class="trade-log-container">
            <div class="paper-active-banner">
                <span class="tag-paper">SIMULATION</span> Virtual Wallet Balance: <b>IDR {balance:,.0f}</b> · Sizing: 40% (min 10k IDR) · Single open position limit
            </div>
            {paper_stats}
            <div class="scroll-hint">⟵ swipe horizontally to view all columns ⟶</div>
            <div class="table-wrap">
                <table class="board" id="paperTradeTable">
                    <thead><tr>
                        <th>Hunt ID</th><th>Time (WIB)</th><th>Symbol</th><th>FW</th><th>Hold</th><th>Entry (IDR)</th>
                        <th>Exit (IDR)</th><th>Qty</th><th>Net PnL (IDR)</th><th>Return %</th>
                        <th>Exit Reason</th>
                    </tr></thead>
                    <tbody>{paper_rows}</tbody>
                </table>
            </div>
            {paper_pager}
        </div>

        <!-- Live Mode panel (hidden until LIVE is armed) -->
        <div id="liveTradeWrap" class="trade-log-container" style="display:none;">
            <div class="live-banner">
                <span class="tag-live">LIVE</span>
                <span class="live-banner-main">
                    Real capital on Indodax ·
                    IDR <b id="liveBalIdr">—</b> available
                    <span id="liveBalHold" class="live-bal-sub"></span>
                </span>
                <span id="liveBalErr" class="live-bal-err"></span>
            </div>
            {live_stats}
            <div class="scroll-hint">⟵ swipe horizontally to view all columns ⟶</div>
            <div class="table-wrap">
                <table class="board" id="liveTradeTable">
                    <thead><tr>
                        <th>Hunt ID</th><th>Time (WIB)</th><th>Symbol</th><th>FW</th><th>Hold</th><th>Entry (IDR)</th>
                        <th>Exit (IDR)</th><th>Qty</th><th>Net PnL (IDR)</th><th>Return %</th>
                        <th>Exit Reason</th>
                    </tr></thead>
                    <tbody>{live_rows}</tbody>
                </table>
            </div>
            {live_pager}
        </div>
    </div>
    """


def _settings_panel() -> str:
    """SETTINGS panel: run first-time setup actions from the web UI.

    Every heavy operation is a button, not something that happens by itself.
    State lives in data/setup_state.json, so the panel shows the truth after a
    restart rather than starting from blank. Buttons are rendered server-side
    from live state and updated by a small poller, so a page reload mid-backfill
    still shows it running.
    """
    import setup_state as S

    r = S.readiness()
    acts = r["actions"]

    STATE_TAG = {
        "not_run": ("<span class='su-tag su-idle'>not run</span>"),
        "running": ("<span class='su-tag su-run'>running</span>"),
        "done":    ("<span class='su-tag su-done'>done</span>"),
        "failed":  ("<span class='su-tag su-fail'>failed</span>"),
    }

    rows = []
    for name in S.ORDER:
        a = acts.get(name, {})
        meta = S.ACTIONS[name]
        state = a.get("state", "not_run")
        tag = STATE_TAG.get(state, STATE_TAG["not_run"])

        # a short line of detail: progress while running, the reason on failure
        detail = ""
        prog = a.get("progress")
        if state == "running" and prog and prog.get("total"):
            pct = int(prog["done"] / prog["total"] * 100)
            detail = (f"<div class='su-prog'><div class='su-bar' "
                      f"style='width:{pct}%'></div></div>"
                      f"<span class='su-dim'>{prog['done']}/{prog['total']}</span>")
        elif state == "failed" and a.get("detail"):
            detail = f"<span class='su-err'>{a['detail'][:120]}</span>"
        elif state == "done" and a.get("finished_at"):
            import datetime as _dt
            when = _dt.datetime.fromtimestamp(a["finished_at"]).strftime("%m-%d %H:%M")
            detail = f"<span class='su-dim'>{when}</span>"

        busy = r.get("busy")
        disabled = (state == "running") or (busy is not None and busy != name)
        btn_label = {"not_run": "RUN", "failed": "RETRY", "done": "RUN AGAIN",
                     "running": "RUNNING"}.get(state, "RUN")

        rows.append(
            f"<div class='su-row'>"
            f"<div class='su-main'>"
            f"<div class='su-name'>{meta['label']} {tag}</div>"
            f"<div class='su-desc'>{meta['desc']}</div>"
            f"{detail}"
            f"</div>"
            f"<button class='su-btn' onclick=\"runSetup('{name}')\""
            f"{' disabled' if disabled else ''}>{btn_label}</button>"
            f"</div>"
        )

    if r["ready"]:
        banner = ("<div class='su-banner ok'>Setup complete — every data "
                  "action has finished.</div>")
    else:
        missing = ", ".join(r["missing_labels"])
        banner = (f"<div class='su-banner warn'>Setup incomplete — still to do: "
                  f"{missing}</div>")

    busy = r.get("busy")
    lock = ""
    if busy:
        lock = (f"<div class='su-dim'>One heavy job at a time: "
                f"<b>{S.ACTIONS[busy]['label']}</b> is running.</div>")

    cron_html = _cron_switches_html()

    return f"""
<div class="panel" id="settingsPanel">
  <h2>Settings — First-Time Setup</h2>
  {banner}
  {lock}
  <div class="su-list">
    {''.join(rows)}
  </div>
  <div class="su-dim" style="margin-top:10px">
    Nothing here runs by itself. Each button starts one job; you can close the
    page and it keeps going.
  </div>
</div>

{cron_html}
"""


def _cron_switches_html() -> str:
    """Render toggle switches for every managed cron job."""
    import crontab_manager as cm
    try:
        crontab_err = cm.crontab_read_error()
    except AttributeError:
        crontab_err = ""
    try:
        states = cm.job_states() or {}
    except Exception:
        states = {}

    if crontab_err:
        import html as _html
        esc = _html.escape(crontab_err[:200])
        return (""
            + "<div class=\"panel\" id=\"cronPanel\">"
            + "  <h2>Cron Jobs</h2>"
            + "  <div class=\"su-banner fail\">Could not read the crontab.</div>"
            + "  <p class=\"su-dim\"><code>" + esc + "</code></p>"
            + "  <p class=\"su-dim\">If it mentions permission, run this on the server, then reload:</p>"
            + "  <pre style=\"background:#1f2937;padding:8px;border-radius:6px;overflow:auto;\">"
            + "sudo loginctl enable-linger $USER</pre>"
            + "</div>"
        )

    rows = []
    for name, schedule, desc in cm.JOBS:
        on = bool(states.get(name))
        checked = " checked" if on else ""
        rows.append(
            f"<div class='su-row'>"
            f"<div class='su-main'>"
            f"<div class='su-name'>{name}</div>"
            f"<div class='su-desc'>{desc} · <code>{schedule}</code></div>"
            f"</div>"
            f"<label class='cron-switch'>"
            f"<input type='checkbox'{checked} onchange=\"toggleCron('{name}', this)\">"
            f"<span class='cron-slider'></span>"
            f"</label>"
            f"</div>"
        )

    rows_html = "\n".join(rows)
    return (
        "<div class=\"panel\" id=\"cronPanel\">"
        "  <h2>Cron Jobs</h2>"
        "  <p class=\"su-dim\">Toggle each scheduled job on or off. "
        "The <b>scanning</b> job runs the trading engine; keep it off while you are fetching data.</p>"
        "  <div class=\"su-list\">" + rows_html + "</div>"
        "  <div id=\"cronStatus\" class=\"su-dim\" style=\"margin-top:8px\"></div>"
        "</div>"
    )


def _framework_panel(state: dict, control: dict) -> str:
    """FRAMEWORKS panel: per-framework cards with accumulated stats.

    Stats accumulate forever (Reza, 2026-11-03): grouping by the framework
    field on each trade means switching never erases history, so A and B stay
    comparable. Rows predating framework tracking are counted under A.
    """
    from frameworks import list_frameworks

    active = control.get("framework", "A")
    mode = control.get("mode", "RUNNING")
    can_switch = (mode == "STOPPED")

    trades = state.get("trade_log", [])
    by_fw = {}
    for t in trades:
        k = t.get("framework") or "A"
        d = by_fw.setdefault(k, {"n": 0, "w": 0, "pnl": 0.0})
        d["n"] += 1
        d["pnl"] += t.get("net_pnl_idr", 0.0) or 0.0
        if (t.get("net_pnl_idr", 0.0) or 0.0) > 0:
            d["w"] += 1

    def _stats_html(name):
        d = by_fw.get(name)
        if not d or d["n"] == 0:
            return ("<div class='fw-stats'>"
                    "<div><div class='k'>hunts</div><div class='v'>0</div></div>"
                    "<div><div class='k'>winrate</div><div class='v'>—</div></div>"
                    "<div><div class='k'>net pnl</div><div class='v'>—</div></div></div>")
        wr = d["w"] / d["n"] * 100
        pnl_cls = "pass" if d["pnl"] >= 0 else "fail"
        sign = "+" if d["pnl"] >= 0 else ""
        return (f"<div class='fw-stats'>"
                f"<div><div class='k'>hunts</div><div class='v'>{d['n']}</div></div>"
                f"<div><div class='k'>winrate</div><div class='v'>{wr:.0f}%</div></div>"
                f"<div><div class='k'>net pnl</div>"
                f"<div class='v {pnl_cls}'>{sign}{d['pnl']:,.0f}</div></div></div>")

    cards = ""
    for f in list_frameworks():
        is_active = (f["name"] == active)
        badges = ""
        if is_active:
            badges += "<span class='fw-badge-active'>ACTIVE</span>"
        if f["experimental"]:
            badges += "<span class='fw-badge-exp'>EXPERIMENTAL</span>"
        if is_active:
            action = "<button class='fw-activate' disabled>✓ ACTIVE</button>"
        else:
            dis = "" if can_switch else "disabled"
            action = (f"<button class='fw-activate' {dis} "
                      f"onclick=\"activateFramework('{f['name']}')\">[ ACTIVATE ]</button>")
        cards += (
            f"<div class='fw-card{' active' if is_active else ''}'>"
            f"<div class='fw-name'>{f['label']} <span style='color:#6b7280;font-size:0.8em'>"
            f"FW {f['name']}</span>{badges}</div>"
            f"<div class='fw-desc'>{f['description']}</div>"
            f"<div class='fw-meta'>v{f['version']}</div>"
            f"{_stats_html(f['name'])}"
            f"{action}</div>"
        )

    hint = ("" if can_switch else
            "<p class='fw-hint'>⏸ Stop hunting to switch framework. "
            "STOP liquidates any open position first.</p>")

    return f"""
        <div class="section">
            <h2>Frameworks — Entry, Exit &amp; Indicators</h2>
            <p class="board-note">Each framework bundles its own entry indicators, exit
            levels and runner management. Switching applies to <b>new</b> hunts only;
            trades already opened keep the framework they started under. Stats accumulate
            per framework and are never reset.</p>
            <div class="fw-grid">{cards}</div>
            {hint}
        </div>
    """


def _winner_card(winner, rows):
    """Render a big CLI-style winner card."""
    if not winner:
        return """<div class="section winner-card"><h2>TARGET ACQUISITION</h2><p style="color:#9ca3af;">NO ELIGIBLE PAIR · STAND BY</p><div class="winner-actions"><button class="toggle-board" id="toggleBoard" type="button">[+] Show Filter Board</button></div></div>"""
    row = next(r for r in rows if r["pair"] == winner)
    c_disp, f_disp = row["candles"][1], row["fresh"][1]
    s_disp = row["spread"][1]
    d_disp = row["depth"][1]
    cd_disp = row["cooldown"][1]
    t_disp = row["trend"][1]
    mom = row["momentum_disp"]
    total = row["total"]
    vol_b = row["vol"] / 1e9
    # If the engine is not running, "WINNER" is a preview of a decision nobody
    # makes. Say so on the card itself: this is the single element an operator
    # looks at, and it read WINNER for days while every tick was refused.
    _blk = None
    try:
        from control import load_control
        _blk = (load_control() or {}).get("blocked")
    except Exception:                                        # noqa: BLE001
        _blk = None

    _tag = ('<span class="winner-tag blocked">NOT TRADED — ENGINE OFF</span>'
            if _blk else '<span class="winner-tag">WINNER</span>')
    _note = (f'<div class="winner-blocked-note">The engine is refusing to run: '
             f'<b>{_blk.get("reason", "blocked")}</b>. Nothing will be bought '
             f'until this is resolved.</div>'
             if _blk else '')

    return f"""<div class="section winner-card{'' if not _blk else ' is-blocked'}">
    <h2>TARGET ACQUISITION</h2>
    <div class="winner-main">
        <span class="winner-name">{winner}</span>
        {_tag}
    </div>
    {_note}
    <div class="winner-grid">
        <div class="kv"><div class="k">Volume</div><div class="v">{vol_b:.2f}B IDR</div></div>
        <div class="kv"><div class="k">Candles</div><div class="v pass"> {c_disp}</div></div>
        <div class="kv"><div class="k">Freshness</div><div class="v pass"> {f_disp}</div></div>
        <div class="kv"><div class="k">Spread</div><div class="v pass">✓ {s_disp}</div></div>
        <div class="kv"><div class="k">Depth</div><div class="v pass">✓ {d_disp}</div></div>
        <div class="kv"><div class="k">Cooldown</div><div class="v pass">✓ {cd_disp}</div></div>
        <div class="kv"><div class="k">Trend</div><div class="v pass">✓ {t_disp}</div></div>
        <div class="kv"><div class="k">Momentum</div><div class="v">{mom}</div></div>
        <div class="kv"><div class="k">Total Score</div><div class="v pass">{total:.0f}</div></div>
    </div>
    <div class="winner-actions">
        <button class="toggle-board" id="toggleBoard" type="button">[+] Show Filter Board</button>
        <!-- BUY NOW: a manual market buy, LIVE only.
             It sits on the winner card because that is where the operator
             already is when they decide to act. It is a real order, so it asks
             for a typed confirmation and the engine re-checks every cap. -->
        <button class="buy-now" id="buyNow" type="button"
                style="display:none"
                onclick="buyNow('{winner}')">BUY {winner} NOW</button>
    </div>
</div>"""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict], factor: int) -> list[dict]:
    """Bucket every `factor` consecutive 15m bars into one candle."""
    if factor <= 1 or len(rows) <= 1:
        return rows
    merged = []
    for i in range(0, len(rows), factor):
        bucket = rows[i:i + factor]
        merged.append({
            "t": bucket[0]["t"],
            "o": bucket[0]["o"],
            "h": max(b["h"] for b in bucket),
            "l": min(b["l"] for b in bucket),
            "c": bucket[-1]["c"],
            "v": sum(b["v"] for b in bucket),
        })
    return merged


def _parse_date(date_str: str, end_of_day: bool = False):
    """Parse YYYY-MM-DD to unix ts (UTC). end_of_day -> 23:59:59 that day."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp())
    except ValueError:
        return None


def _trades_for(symbol: str) -> list:
    """Closed trades for one symbol, oldest first, with the fields a chart needs.

    Read from hunter_state.json's trade_log. Malformed rows are skipped rather
    than raised: a chart must not fail because one trade record is odd.
    """
    import state as _state
    try:
        st = _state.load_state()
    except Exception:
        return []
    out = []
    for t in (st.get("trade_log") or []):
        if not isinstance(t, dict):
            continue
        if (t.get("symbol") or "").upper() != symbol.upper():
            continue
        entry, exit_ = t.get("entry_time"), t.get("exit_time")
        if not entry or not exit_:
            continue
        try:
            out.append({
                "hunt_id": t.get("hunt_id"),
                "entry_time": float(entry),
                "exit_time": float(exit_),
                "entry_price": float(t.get("entry_price") or 0),
                "exit_price": float(t.get("exit_price") or 0),
                "net_pnl_idr": float(t.get("net_pnl_idr") or 0),
                "net_pnl_pct": float(t.get("net_pnl_pct") or 0),
                "exit_reason": t.get("exit_reason") or "",
                "framework": t.get("framework") or "",
            })
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: x["entry_time"])
    return out


@app.get("/api/ohlcv/{pair}")
async def api_ohlcv(pair: str, tf: str = DEFAULT_TF, from_date: str = "", to_date: str = ""):
    """
    Return candles for a pair as JSON.
      tf: 15m | 1h | 3h | 6h | 12h | 1d  (aggregated from stored 15m bars)
      from_date / to_date: YYYY-MM-DD inclusive; empty = no bound
    If the result exceeds MAX_CHART_POINTS it is aggregated further
    (flagged "aggregated": true so the UI can say so).
    """
    symbol = pair.upper()
    if tf not in TIMEFRAMES:
        return JSONResponse(
            {"error": f"unknown timeframe '{tf}' (valid: {', '.join(TIMEFRAMES)})"},
            status_code=400,
        )

    csv_path = OHLCV_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        return JSONResponse({"error": f"no data for {symbol}"}, status_code=404)

    from_ts = _parse_date(from_date, end_of_day=False)
    to_ts = _parse_date(to_date, end_of_day=True)

    rows = []
    with open(csv_path, newline="") as fh:
        for row in fh:
            parts = row.strip().split(",")
            if not parts or parts[0] == "timestamp":
                continue
            try:
                ts = int(parts[0])
                if from_ts is not None and ts < from_ts:
                    continue
                if to_ts is not None and ts > to_ts:
                    continue
                rows.append({
                    "t": ts,
                    "o": float(parts[1]),
                    "h": float(parts[2]),
                    "l": float(parts[3]),
                    "c": float(parts[4]),
                    "v": float(parts[5]),
                })
            except (ValueError, IndexError):
                continue

    if not rows:
        return JSONResponse({"error": f"no data for {symbol} in range"}, status_code=404)

    # 1) timeframe aggregation (e.g. 1h = 4 bars of 15m)
    factor = TIMEFRAMES[tf]
    rows = _aggregate(rows, factor)

    # 2) render cap — if still huge, aggregate further
    aggregated = factor > 1
    if len(rows) > MAX_CHART_POINTS:
        factor2 = -(-len(rows) // MAX_CHART_POINTS)  # ceil
        rows = _aggregate(rows, factor2)
        aggregated = True

    # Overlay data. The chart shades each trade's holding window so the user can
    # see what happened AFTER the bot exited - the whole point of the overlay.
    # Times are epoch seconds, so they survive any timeframe aggregation.
    return JSONResponse({
        "pair": symbol,
        "tf": tf,
        "candles": rows,
        "aggregated": aggregated,
        "total_bars": len(rows),
        "from": from_ts,
        "to": to_ts,
        "trades": _trades_for(symbol),
    })


def _style():
    return """
        :root {
            /* DARK (default). Every structural surface, border and text colour
               resolves through these, so a light theme is a change of VALUES,
               not a second stylesheet. Semantic colours (profit green, loss
               red, warning amber) stay literal on purpose: their meaning must
               not change with the palette. */
            --bg-primary: #0c0c0c;
            --bg-deep: #0b1220;
            --bg-secondary: #111827;
            --bg-card: #111827;
            --bg-elevated: #1f2937;
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --border: #374151;
            --border-strong: #4b5563;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-blue: #60a5fa;
            --border-color: #374151;
            --terminal-green: #22c55e;

            /* surfaces that carry a tint in dark mode */
            --row-hover: rgba(255,255,255,0.03);
            --shadow: 0 1px 3px rgba(0,0,0,.5);
            --chart-grid: rgba(255,255,255,0.06);
            --warn: #fbbf24;
        }


        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        html, body {
            width: 100%;
            max-width: 100vw;
            /* NOTE: do NOT leave overflow-x: hidden here — on iOS Safari it kills
               horizontal touch-scrolling on nested .table-wrap/.board-wrap
               containers (tables become unreachable). clip prevents body-level
               scroll without creating a scroll container; the hidden line below
               is only a fallback for browsers that don't understand clip. */
            overflow-x: hidden;
            overflow-x: clip;
            -webkit-text-size-adjust: 100%;
        }
        body {
            font-family: "Courier New", Courier, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 12px;
            line-height: 1.5;
        }
        h1 { color: var(--terminal-green); margin-bottom: 4px; font-size: 1.5rem; letter-spacing: 1px; }
        h1::before { content: "[~]$ "; color: var(--terminal-green); }
        .subtitle { color: var(--text-secondary); font-size: 0.85em; margin-top: 0; margin-bottom: 16px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin-bottom: 16px; }
        .card {
            background: var(--bg-card); border-radius: 8px; padding: 14px 10px;
            text-align: center; border: 1px solid var(--border-color);
        }
        .card .num { font-size: 1.8em; font-weight: 700; display: block; margin-bottom: 4px; }
        .card .label { font-size: 0.68em; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
        .card.total .num { color: var(--accent-blue); }
        .card.complete .num { color: #4ade80; }
        .card.stale .num { color: #f87171; }
        .card.size .num { color: #a78bfa; font-size: 1.3em; }
        .section {
            background: var(--bg-primary);
            border-radius: 0;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid var(--border);
            box-shadow: inset 0 0 20px rgba(34, 197, 94, 0.03);
        }
        .section h2 {
            color: var(--terminal-green); margin-top: 0; font-size: 1em;
            border-bottom: 1px dashed var(--border); padding-bottom: 10px; margin-bottom: 14px;
            font-family: "Courier New", Courier, monospace; text-transform: uppercase; letter-spacing: 1px;
        }
        .section h2::before { content: "# "; color: var(--terminal-green); }
        table { border-collapse: collapse; width: 100%; border: 1px solid var(--border); }
        th, td { border: 1px solid var(--border); padding: 10px 8px; text-align: left; font-family: "Courier New", Courier, monospace; }
        th { background: var(--bg-card); color: var(--terminal-green); font-weight: 700; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.3px; }
        tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
        tbody tr:hover { background: rgba(34, 197, 94, 0.08); }
        .stale { color: #f87171; font-weight: bold; }
        .fresh { color: #4ade80; font-weight: bold; }
        /* Stale table: compact, scrollable — never taller than 300px */
        .stale-scroll { max-height: 300px; overflow-y: auto; overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; border: 1px solid var(--border); }
        .stale-scroll table { font-size: 0.85em; min-width: 460px; }
        .stale-scroll th { position: sticky; top: 0; z-index: 2; }
        /* Top-20 pairs table: compact, scrollable — never taller than 300px */
        .pairs-scroll { max-height: 300px; overflow-y: auto; overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; border: 1px solid var(--border); }
        .pairs-scroll table { font-size: 0.85em; min-width: 480px; }
        .pairs-scroll th { position: sticky; top: 0; z-index: 2; }

        /* Control plane: START / STOP switch + framework picker */
        .ctrl-wrap { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; }

        .row-buy {
            display: none;
            padding: 4px 10px;
            border: 1px solid #ef4444;
            border-radius: 4px;
            background: transparent;
            color: #ef4444;
            font-family: inherit;
            font-size: 0.72em;
            font-weight: 700;
            letter-spacing: 0.06em;
            cursor: pointer;
            white-space: nowrap;
        }
        .row-buy:hover { background: #ef4444; color: #fff; }
        .row-buy:disabled { opacity: 0.5; cursor: wait; }
        .board td.act { text-align: center; min-width: 62px; }
        .action-note {
            display: block;
            margin: 8px 0 0;
            font-size: 0.72em;
            color: var(--text-dim);
            line-height: 1.5;
        }

        /* ---- live balance panel ------------------------------------------ */
        .buy-now {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-left: 10px;
            padding: 9px 16px;
            border: 1px solid #ef4444;
            border-radius: 6px;
            background: #ef4444;
            color: #fff;
            font-family: inherit;
            font-size: 0.8em;
            font-weight: 700;
            letter-spacing: 0.06em;
            cursor: pointer;
            transition: background 0.15s, transform 0.1s;
        }
        .buy-now:hover { background: #dc2626; }
        .buy-now:active { transform: scale(0.98); }
        .buy-now:disabled {
            background: #7f1d1d;
            border-color: #7f1d1d;
            cursor: wait;
        }
        @media (max-width: 720px) {
            .buy-now { margin-left: 0; margin-top: 8px; width: 100%; justify-content: center; min-height: 46px; }
        }

        .winner-tag.blocked {
            background: rgba(239,68,68,0.18);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        .winner-card.is-blocked { border-color: #ef4444; }
        .winner-blocked-note {
            margin: 10px 0 4px;
            padding: 9px 12px;
            border-left: 3px solid #ef4444;
            background: rgba(239,68,68,0.08);
            color: #fca5a5;
            font-size: 0.82em;
            line-height: 1.45;
        }

        .badge-status.blocked {
            background: rgba(239,68,68,0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        .ctrl-switch.blocked {
            background: #7f1d1d;
            border-color: #ef4444;
            color: #fff;
        }

        .live-banner {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            padding: 11px 16px;
            border: 1px solid #ef4444;
            border-left-width: 4px;
            border-radius: 6px;
            background: rgba(239,68,68,0.07);
            font-size: 0.82em;
            color: var(--text-primary);
        }
        .live-banner-main b { color: #fff; font-size: 1.15em; }
        .live-bal-sub { color: var(--text-dim); font-size: 0.9em; }
        .live-bal-err {
            color: #fca5a5;
            font-size: 0.9em;
            width: 100%;
        }
        .tag-live {
            background: #ef4444;
            color: #fff;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.78em;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        /* ---- LIVE execution ----------------------------------------------
           LIVE must never look like paper. A separate button colour, a red
           banner across the top and a bordered page edge: if a screenshot of
           this dashboard exists, "real money" should be unmissable in it. */
        .ctrl-wrap .ctrl-switch.ctrl-live {
            background: transparent;
            border: 1px solid #ef4444;
            color: #ef4444;
        }
        .ctrl-wrap .ctrl-switch.ctrl-live:hover { background: rgba(239,68,68,0.12); }
        .ctrl-wrap .ctrl-switch.live {
            background: #ef4444;
            border-color: #ef4444;
            color: #fff;
        }
        .live-badge {
            display: inline-block;
            margin-left: 10px;
            padding: 2px 8px;
            border-radius: 3px;
            background: #ef4444;
            color: #fff;
            font-size: 0.68em;
            font-weight: 700;
            letter-spacing: 0.08em;
            animation: livePulse 2s ease-in-out infinite;
        }
        @keyframes livePulse {
            0%, 100% { opacity: 1; }
            50%      { opacity: 0.45; }
        }
        body.live-armed { box-shadow: inset 0 0 0 3px #ef4444; }

        .ctrl-switch {
            display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
            font-family: "Courier New", Courier, monospace; font-size: 0.82em;
            font-weight: 700; letter-spacing: 0.5px; padding: 9px 16px;
            border: 1px solid #22c55e; background: rgba(34,197,94,0.14);
            color: var(--terminal-green); transition: all 0.15s ease; user-select: none;
        }
        .ctrl-switch:hover { background: rgba(34,197,94,0.24); }
        .ctrl-switch.stopped {
            border-color: var(--text-muted); background: rgba(107,114,128,0.14); color: var(--text-secondary);
        }
        .ctrl-switch.stopped:hover { background: rgba(107,114,128,0.24); }
        .ctrl-switch .led {
            width: 8px; height: 8px; border-radius: 50%; background: currentColor;
            box-shadow: 0 0 8px currentColor;
        }
        .ctrl-switch.stopped .led { box-shadow: none; }
        .ctrl-switch.busy { opacity: 0.55; pointer-events: none; }
        .fw-tag {
            display: inline-block; font-family: "Courier New", Courier, monospace;
            font-size: 0.78em; font-weight: 700; padding: 1px 6px; border: 1px solid;
        }
        .fw-tag.fw-a { color: var(--terminal-green); border-color: var(--terminal-green); background: rgba(34,197,94,0.12); }
        .fw-tag.fw-exp { color: var(--warn); border-color: var(--warn); background: rgba(251,191,36,0.12); }
        .fw-tag.fw-legacy { color: var(--text-muted); border-color: var(--border-strong); background: rgba(75,85,99,0.18); }

        /* Settings — first-time setup */
        .su-banner { padding: 9px 12px; margin-bottom: 12px; font-size: 0.8em;
                     border: 1px solid; }
        .su-banner.ok   { color: #86efac; border-color: var(--terminal-green); background: rgba(34,197,94,0.10); }
        .su-banner.warn { color: var(--warn); border-color: #b45309; background: rgba(251,191,36,0.10); }
        .su-list { display: flex; flex-direction: column; gap: 8px; }
        .su-row { display: flex; align-items: center; gap: 12px;
                  background: var(--bg-deep); border: 1px solid var(--bg-elevated); padding: 10px 12px; }
        .su-main { flex: 1; min-width: 0; }
        .su-name { font-family: "Courier New", Courier, monospace; font-size: 0.84em;
                   color: var(--text-primary); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .su-desc { font-size: 0.72em; color: var(--text-secondary); margin-top: 3px; }
        .su-tag { font-size: 0.66em; padding: 1px 6px; border: 1px solid; letter-spacing: 0.04em; }
        .su-idle { color: var(--text-secondary); border-color: var(--border-strong); }
        .su-run  { color: var(--accent-blue); border-color: #3b82f6; }
        .su-done { color: #86efac; border-color: var(--terminal-green); }
        .su-fail { color: #fca5a5; border-color: var(--accent-red); }
        .su-btn { font-family: "Courier New", Courier, monospace; font-size: 0.72em;
                  padding: 7px 14px; background: transparent; color: var(--terminal-green);
                  border: 1px solid #22c55e; cursor: pointer; white-space: nowrap; }
        .su-btn:hover:not(:disabled) { background: rgba(34,197,94,0.12); }
        .su-btn:disabled { color: var(--text-muted); border-color: var(--border-strong); cursor: default; }
        .su-prog { height: 4px; background: var(--bg-elevated); margin-top: 6px; width: 100%; max-width: 320px; }
        .su-bar { height: 100%; background: var(--accent-green); transition: width 0.4s; }
        .su-dim { font-size: 0.7em; color: var(--text-muted); }
        .su-err { font-size: 0.7em; color: #fca5a5; }

        /* Framework picker */
        .fw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }
        .fw-card {
            background: var(--bg-deep); border: 1px solid var(--bg-elevated); padding: 14px;
            display: flex; flex-direction: column; gap: 10px;
        }
        .fw-card.active { border-color: var(--terminal-green); box-shadow: 0 0 0 1px rgba(34,197,94,0.35) inset; }
        .fw-card .fw-name {
            font-family: "Courier New", Courier, monospace; font-size: 1.05em;
            font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;
        }
        .fw-card .fw-desc { font-size: 0.76em; color: var(--text-secondary); line-height: 1.45; }
        .fw-card .fw-meta { font-size: 0.7em; color: var(--text-muted); font-family: "Courier New", Courier, monospace; }
        .fw-card .fw-stats {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
            font-family: "Courier New", Courier, monospace; font-size: 0.72em;
            border-top: 1px solid var(--bg-elevated); padding-top: 9px;
        }
        .fw-card .fw-stats .k { color: var(--text-muted); font-size: 0.9em; }
        .fw-card .fw-stats .v { color: var(--text-primary); font-weight: 700; }
        .fw-badge-active {
            font-size: 0.62em; padding: 2px 7px; background: var(--accent-green); color: var(--bg-primary);
            letter-spacing: 0.6px; font-weight: 700;
        }
        .fw-badge-exp {
            font-size: 0.62em; padding: 2px 7px; border: 1px solid #fbbf24;
            color: var(--warn); letter-spacing: 0.6px; font-weight: 700;
        }
        .fw-activate {
            font-family: "Courier New", Courier, monospace; font-size: 0.78em;
            font-weight: 700; cursor: pointer; padding: 9px; border: 1px solid #22c55e;
            background: transparent; color: var(--terminal-green); letter-spacing: 0.5px;
        }
        .fw-activate:hover:not(:disabled) { background: rgba(34,197,94,0.16); }
        .fw-activate:disabled { border-color: var(--border); color: var(--border-strong); cursor: not-allowed; }
        .fw-hint { font-size: 0.72em; color: var(--warn); margin-top: 8px; }

        /* Winner card */
        .winner-card {
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-deep) 100%);
            border: 1px solid #22c55e;
            border-left: 6px solid #22c55e;
            border-radius: 8px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 0 18px rgba(34, 197, 94, 0.12);
            position: relative;
            overflow: hidden;
        }
        .winner-card::before {
            content: "> ";
            color: var(--terminal-green);
        }
        .winner-card h2 {
            margin: 0 0 12px 0;
            padding: 0;
            border: none;
            color: var(--terminal-green);
            font-size: 1rem;
            font-weight: 700;
        }
        .winner-main {
            display: flex;
            align-items: baseline;
            gap: 18px;
            flex-wrap: wrap;
        }
        .winner-name {
            font-family: "Courier New", Courier, monospace;
            font-size: 2.4em;
            font-weight: 700;
            color: #4ade80;
            letter-spacing: 1px;
        }
        .winner-tag {
            font-size: 0.75em;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-radius: 4px;
            padding: 2px 8px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .winner-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 16px;
        }
        .winner-grid .kv {
            background: var(--bg-deep);
            border: 1px solid var(--bg-elevated);
            border-radius: 4px;
            padding: 10px 12px;
        }
        .winner-grid .kv .k {
            font-size: 0.7em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 4px;
        }
        .winner-grid .kv .v {
            font-family: "Courier New", Courier, monospace;
            font-size: 1.1em;
            color: var(--text-primary);
            font-weight: 700;
        }
        .winner-grid .kv .v.pass { color: #4ade80; }
        .winner-grid .kv .v.fail { color: #f87171; }
        .toggle-board {
            margin-top: 14px;
            background: transparent;
            color: var(--terminal-green);
            border: 1px solid #22c55e;
            border-radius: 4px;
            padding: 8px 16px;
            cursor: pointer;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .toggle-board:hover { background: rgba(34, 197, 94, 0.1); }
        .board-collapsed { display: none; }

        .board-note { color: var(--text-muted); font-size: 0.75em; margin: 0 0 8px 0; font-family: "Courier New", Courier, monospace; }
        .board-summary { color: var(--text-secondary); font-size: 0.85em; margin: 0 0 10px 0; }
        .board-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; border: 1px solid var(--border); }
        .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; border: 1px solid var(--border); width: 100%; margin-bottom: 6px; }
        .board { min-width: 980px; font-size: 0.85em; }
        .board td { white-space: nowrap; font-family: "Courier New", Courier, monospace; }
        .board td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .board td.pass { color: #4ade80; }
        .board td.fail { color: #f87171; }
        .board tr.winner { background: rgba(96, 165, 250, 0.12); }
        .badge-win { background: var(--accent-green); color: var(--bg-primary); border-radius: 0; padding: 1px 7px; font-size: 0.75em; font-weight: 700; font-family: "Courier New", Courier, monospace; }
        .badge-ok { color: #4ade80; font-weight: 600; }
        .badge-no { color: #f87171; font-size: 0.85em; }

        /* ML veto layer */
        .ml-veto { color: #f87171; font-weight: 700; }
        .ml-up { color: #4ade80; font-weight: 700; }
        .ml-flat { color: var(--text-secondary); }
        .ml-majors { color: var(--text-muted); }
        .ml-none { color: var(--border-strong); }
        .board td.ml-veto, .board td.ml-up { font-weight: 700; }
        .winner-grid .kv .v .ml-veto { font-weight: 700; }

        /* Model report section */
        .report-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 14px; }
        .report-sub { color: var(--text-secondary); font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.8px; margin: 10px 0 6px 0; }
        .report-sub:first-child { margin-top: 0; }
        .report-tbl { font-size: 0.82em; width: 100%; }
        .report-tbl td:first-child { color: #d1d5db; }
        pre { overflow-x: auto; -webkit-overflow-scrolling: touch; font-size: 0.75em; }
        .footer { text-align: center; color: var(--text-secondary); font-size: 0.75em; margin-top: 20px; padding: 16px; }

        /* Chart controls */
        .chart-controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 12px; }
        .ctrl-group { display: flex; flex-direction: column; gap: 3px; flex: 1 1 170px; min-width: 0; }
        .ctrl-group label { color: var(--text-secondary); font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.5px; }
        .ctrl-group select, .ctrl-group input[type="date"] {
            background: var(--bg-primary); color: var(--text-primary); border: 1px solid var(--border);
            border-radius: 4px; font-size: 0.9em; color-scheme: dark;
            font-family: inherit;
            /* identical box: same height/width/padding for selects AND date inputs
               (native date inputs otherwise size themselves differently) */
            height: 40px; width: 100%; box-sizing: border-box;
            padding: 0 10px; margin: 0; line-height: normal;
        }
        .ctrl-group input[type="date"] {
            -webkit-appearance: none; appearance: none;
        }
        .reset-btn {
            background: var(--bg-primary); color: var(--terminal-green); border: 1px solid #22c55e;
            border-radius: 4px; padding: 0 14px; cursor: pointer; font-size: 0.85em;
            font-weight: 700; font-family: inherit; text-transform: uppercase;
            height: 40px;
        }
        .reset-btn:active { background: rgba(34, 197, 94, 0.15); color: #fff; }
        .chart-info { color: var(--text-secondary); font-size: 0.78em; margin-bottom: 6px; word-break: break-all; }
        .chart-tip { color: var(--text-muted); font-size: 0.75em; margin-bottom: 10px; line-height: 1.4; }
        .chart-wrap { position: relative; width: 100%; height: 440px; }
        #chart { width: 100%; height: 100%; display: block; cursor: crosshair; touch-action: pan-y; }
        #tooltip {
            position: absolute; pointer-events: none; display: none;
            background: rgba(11, 18, 32, 0.95); border: 1px solid var(--border-color); border-radius: 4px;
            padding: 6px 8px; font-size: 0.72em; color: var(--text-primary);
            white-space: nowrap; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.6);
        }
        .chart-tooltip {
            position: absolute; pointer-events: none; display: none;
            background: rgba(11, 18, 32, 0.95); border: 1px solid var(--border-color); border-radius: 4px;
            padding: 6px 8px; font-size: 0.72em; color: var(--text-primary);
            white-space: nowrap; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.6);
        }

        /* Top Bar & Burger Button */
        .top-bar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 12px; background: var(--bg-card); border: 1px solid var(--border);
            margin-bottom: 12px; border-left: 3px solid #22c55e;
            position: sticky; top: 0; z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .top-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
        .burger-btn {
            background: var(--bg-primary); border: 1px solid #22c55e; color: var(--terminal-green);
            min-width: 44px; height: 40px; display: flex; flex-direction: column;
            justify-content: center; align-items: center; gap: 4px; cursor: pointer;
            padding: 0; flex-shrink: 0; transition: all 0.2s;
        }
        .burger-btn:active { background: var(--accent-green); }
        .burger-btn:active span { background: var(--bg-primary); }
        .burger-btn span {
            display: block; width: 20px; height: 2px; background: var(--accent-green);
            transition: all 0.2s;
        }
        /* Refresh button (top bar) — re-fetches the page without F5 */
        .refresh-btn {
            background: var(--bg-primary); border: 1px solid #22c55e; color: var(--terminal-green);
            min-width: 44px; height: 40px; display: inline-flex;
            justify-content: center; align-items: center; cursor: pointer;
            padding: 0 8px; flex-shrink: 0; transition: all 0.2s;
            font-size: 1.1em; line-height: 1; font-family: inherit;
        }
        .refresh-btn:active { background: var(--accent-green); color: var(--bg-primary); }
        .refresh-btn.spinning { animation: hunter-spin 0.8s linear infinite; }
        @keyframes hunter-spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
        }
        /* Brief confirmation toast */
        .toast {
            position: fixed; left: 50%; transform: translateX(-50%);
            bottom: 24px; z-index: 999; background: var(--bg-card);
            border: 1px solid #22c55e; border-left: 3px solid #22c55e;
            color: var(--text-primary); padding: 9px 14px; font-size: 0.8em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.6); border-radius: 4px;
            opacity: 0; pointer-events: none; transition: opacity 0.25s;
        }
        .toast.show { opacity: 1; }

        /* ---- Single-hunt performance page ---- */
        .hunt-detail { padding: 12px 14px 16px; }
        .hunt-head { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
        .hunt-close-btn {
            margin-left: auto; width: 36px; height: 36px; display: inline-flex;
            justify-content: center; align-items: center; flex-shrink: 0;
            background: var(--bg-primary); border: 1px solid #ef4444; color: var(--accent-red);
            font-size: 1.55em; line-height: 1; cursor: pointer; font-family: inherit;
        }
        .hunt-close-btn:hover, .hunt-close-btn:active { background: #ef4444; color: var(--bg-primary); }
        .hunt-detail-row > td { padding: 8px; background: var(--bg-primary); }
        .hunt-detail-row .hunt-detail { margin: 0; }
        .hunt-title { display: flex; align-items: baseline; gap: 8px; min-width: 0; }
        .hunt-id-big { color: var(--text-primary); font-weight: bold; font-size: 1.15em; letter-spacing: 1px; }
        .hunt-sym { color: var(--terminal-green); font-weight: bold; font-size: 0.95em; }
        .hunt-kv {
            display: flex; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 12px;
            padding: 10px 12px; background: var(--bg-primary); border: 1px solid var(--border);
            border-left: 3px solid #22c55e; font-size: 0.82em;
        }
        .hunt-kv .k { color: var(--text-secondary); text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; }
        .hunt-kv .v { color: var(--text-primary); font-weight: 700; }
        .hunt-kv .v.pos { color: var(--terminal-green); }
        .hunt-kv .v.neg { color: var(--accent-red); }
        .hunt-chart-wrap { position: relative; width: 100%; height: 340px; }
        #huntChart { width: 100%; height: 100%; display: block; cursor: crosshair; touch-action: pan-y; }

        /* Clickable hunt ids in the trade tables */
        .hunt-link {
            color: var(--terminal-green); cursor: pointer; text-decoration: underline dotted;
            text-underline-offset: 2px; font-weight: 700;
        }
        .hunt-link:active { color: #fff; }

        /* Clickable pair names in the filter board */
        .pair-link {
            color: var(--text-primary); cursor: pointer; text-decoration: underline dotted;
            text-underline-offset: 2px;
        }
        .pair-link:active { color: var(--terminal-green); }
        .board-chart-row > td { padding: 8px; background: var(--bg-primary); }
        .bc-wrap { border-left: 3px solid #22c55e; padding: 8px 10px 10px; }
        .bc-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
        .bc-title { color: var(--text-secondary); font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.5px; }
        .bc-pair { color: var(--terminal-green); font-weight: bold; }
        .bc-tf {
            background: var(--bg-primary); color: var(--text-primary); border: 1px solid var(--border);
            border-radius: 4px; padding: 4px 8px; font-size: 0.8em; font-family: inherit;
            color-scheme: dark; height: 30px;
        }
        .bc-close {
            margin-left: auto; width: 30px; height: 30px; display: inline-flex;
            justify-content: center; align-items: center; flex-shrink: 0;
            background: var(--bg-primary); border: 1px solid #ef4444; color: var(--accent-red);
            font-size: 1.3em; line-height: 1; cursor: pointer; font-family: inherit;
        }
        .bc-close:hover, .bc-close:active { background: #ef4444; color: var(--bg-primary); }
        .bc-info { color: var(--text-secondary); font-size: 0.75em; margin-bottom: 6px; }
        .bc-chart-wrap { position: relative; width: 100%; height: 300px; }
        #boardChartCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; touch-action: pan-y; }
        .brand { display: flex; align-items: center; gap: 6px; min-width: 0; overflow: hidden; }
        .cli-prompt { color: var(--terminal-green); font-weight: bold; flex-shrink: 0; }
        .brand-name {
            color: var(--text-primary); font-weight: bold; letter-spacing: 1px; font-size: 1.05em;
            white-space: nowrap;
        }
        .active-crumb {
            color: var(--text-secondary); font-size: 0.8em; text-transform: uppercase;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .top-right {
            display: flex; align-items: center; gap: 8px; font-size: 0.78em; color: var(--text-secondary);
            flex-shrink: 0;
        }
        .conn-dot { color: var(--terminal-green); font-size: 1.1em; }

        /* Cron job toggle switches */
        .cron-switch { position: relative; display: inline-block; width: 46px; height: 24px; flex-shrink: 0; }
        .cron-switch input {
            opacity: 0; width: 0; height: 0;
            -webkit-appearance: none; -moz-appearance: none; appearance: none;
            position: absolute; margin: 0; padding: 0; border: none;
        }
        .cron-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
                       background-color: var(--accent-red); transition: .2s; border-radius: 24px; }
        .cron-slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px;
                              bottom: 3px; background-color: white; transition: .2s; border-radius: 50%; }
        .cron-switch input:checked + .cron-slider { background-color: var(--terminal-green); }
        .cron-switch input:checked + .cron-slider:before { transform: translateX(22px); }
        .cron-switch input:focus + .cron-slider { box-shadow: 0 0 0 2px rgba(34,197,94,0.3); }

        /* Nav Drawer & Overlay */
        .nav-drawer {
            position: fixed; top: 0; left: -320px; width: min(290px, 85vw);
            height: 100vh; height: 100dvh;
            background: var(--bg-secondary); border-right: 2px solid #22c55e; z-index: 1000;
            transition: left 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            padding: max(16px, env(safe-area-inset-top)) 14px max(16px, env(safe-area-inset-bottom)) 14px;
            display: flex; flex-direction: column; box-shadow: 10px 0 30px rgba(0,0,0,0.85);
            overflow-y: auto; -webkit-overflow-scrolling: touch;
        }
        .nav-drawer.open { left: 0; }
        .drawer-top {
            display: flex; justify-content: space-between; align-items: center;
            padding-bottom: 12px; border-bottom: 1px dashed var(--border); margin-bottom: 14px;
        }
        .drawer-title { color: var(--terminal-green); font-weight: bold; letter-spacing: 1px; font-size: 0.9em; }
        .drawer-close {
            background: transparent; border: 1px solid var(--border); color: var(--text-secondary);
            cursor: pointer; font-family: inherit; min-width: 36px; height: 36px;
            display: flex; align-items: center; justify-content: center; font-weight: bold;
        }
        .drawer-close:hover, .drawer-close:active { color: var(--accent-red); border-color: var(--accent-red); }
        .drawer-links { display: flex; flex-direction: column; gap: 8px; flex: 1; }
        .nav-item {
            background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary);
            padding: 14px 12px; text-align: left; cursor: pointer; display: flex;
            align-items: center; gap: 12px; font-family: inherit; transition: all 0.2s;
            min-height: 48px; border-radius: 4px;
        }
        .nav-item:hover, .nav-item:active { border-color: var(--terminal-green); background: rgba(34,197,94,0.08); }
        .nav-item.active { border-color: var(--terminal-green); background: rgba(34,197,94,0.15); border-left: 4px solid #22c55e; }
        .nav-item.active .nav-icon { color: var(--terminal-green); }
        .nav-icon { color: var(--text-muted); font-size: 0.9em; flex-shrink: 0; }
        .nav-title { font-weight: bold; font-size: 0.95em; color: var(--terminal-green); }
        .nav-desc { font-size: 0.72em; color: var(--text-secondary); margin-top: 3px; line-height: 1.3; }
        .drawer-bottom {
            border-top: 1px dashed var(--border); padding-top: 12px; font-size: 0.75em;
            color: var(--text-muted); line-height: 1.5; margin-top: auto;
        }
        .nav-overlay {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.65); backdrop-filter: blur(2px); z-index: 999;
            display: none;
        }
        .nav-overlay.open { display: block; }

        /* Trade Log Section & Mode Toggle */
        .trade-header {
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 10px; margin-bottom: 12px;
        }
        .trade-sub { font-size: 0.78em; color: var(--text-secondary); margin-top: 4px; line-height: 1.4; word-break: break-word; }
        .badge-status {
            display: inline-block; padding: 2px 8px; font-size: 0.75em;
            font-weight: bold; margin-right: 6px; border-radius: 3px;
        }
        .badge-status.hunting { background: rgba(34,197,94,0.15); color: var(--terminal-green); border: 1px solid #22c55e; }
        .badge-status.holding { background: rgba(96,165,250,0.15); color: var(--accent-blue); border: 1px solid #60a5fa; }
        .badge-status.open { background: rgba(251,191,36,0.15); color: var(--warn); border: 1px solid #fbbf24; }

        .mode-control-wrap { display: flex; align-items: center; }
        /* Ledger tabs. They choose which table you read; they do not choose
           which executor trades. */
        .mode-btn-group {
            display: flex;
            border: 1px solid var(--border);
            background: var(--bg-card);
            border-radius: 6px;
            overflow: hidden;
            margin-left: 10px;
        }
        .mode-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 8px 14px;
            font-family: inherit;
            font-size: 0.78em;
            font-weight: 700;
            letter-spacing: 0.05em;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            transition: background 0.15s, color 0.15s;
        }
        .mode-btn:hover { background: var(--bg-elevated); }
        .mode-btn.active { background: var(--bg-elevated); color: var(--text-primary); }
        .mode-btn .dot { width: 7px; height: 7px; border-radius: 50%; }
        .mode-btn .dot.paper { background: #60a5fa; }
        .mode-btn .dot.live { background: #ef4444; }
        #tabLive.active { color: #ef4444; box-shadow: inset 0 -2px 0 #ef4444; }
        #tabPaper.active { color: #60a5fa; box-shadow: inset 0 -2px 0 #60a5fa; }
        @media (max-width: 720px) {
            .mode-control-wrap { width: 100%; flex-wrap: wrap; gap: 8px; }
            .mode-btn-group { width: 100%; margin-left: 0; }
            .mode-btn { flex: 1; justify-content: center; padding: 11px 6px; min-height: 44px; }
        }

        /* A read-only badge, deliberately not shaped like a switch: no border
           between halves, no hover state, cursor stays default. */
        .exec-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 0.78em;
            font-weight: 700;
            letter-spacing: 0.10em;
            border: 1px solid var(--border);
            background: var(--panel-2);
            color: var(--text-dim);
            cursor: default;
            user-select: none;
        }
        .exec-badge .dot { width: 8px; height: 8px; border-radius: 50%; }
        .exec-badge .dot.paper { background: #60a5fa; }
        .exec-badge .dot.live  { background: #ef4444; }
        .exec-badge.is-paper { color: var(--accent-blue); border-color: #2563eb; }
        .exec-badge.is-live  { color: #ef4444; border-color: #ef4444;
                               background: rgba(239,68,68,0.10); }
        .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
        .dot.live { background: var(--accent-green); }
        .dot.paper { background: #60a5fa; }

        .live-standby-banner {
            display: flex; align-items: center; gap: 12px;
            background: rgba(34,197,94,0.05); border: 1px dashed #22c55e;
            padding: 12px 14px; margin-bottom: 10px;
        }
        .standby-icon { font-size: 1.4em; flex-shrink: 0; }
        .standby-title { color: var(--terminal-green); font-weight: bold; font-size: 0.85em; }
        .standby-sub { color: var(--text-secondary); font-size: 0.75em; margin-top: 2px; line-height: 1.3; }

        .paper-active-banner {
            background: rgba(96,165,250,0.06); border: 1px dashed #60a5fa;
            padding: 10px 14px; margin-bottom: 10px; font-size: 0.78em; color: var(--text-secondary); line-height: 1.4;
        }
        .tag-paper {
            background: #60a5fa; color: var(--bg-primary); padding: 1px 6px;
            font-size: 0.75em; font-weight: bold; margin-right: 6px; border-radius: 2px;
        }
        tr.open-trade { background: rgba(96, 165, 250, 0.08) !important; }

        /* Trade log pagination */
        .stats-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-bottom: 10px;
        }
        .stat-card {
            background: var(--bg-deep); border: 1px solid var(--border);
            padding: 8px 6px; text-align: center; border-radius: 3px;
        }
        .stat-card .sv {
            font-weight: 700; font-size: 1.0em; color: var(--text-primary);
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .stat-card .sv.pass { color: #4ade80; }
        .stat-card .sv.fail { color: #f87171; }
        .stat-card .sv.num-hunts { color: var(--accent-blue); }
        .stat-card .sl {
            font-size: 0.62em; color: var(--text-secondary); text-transform: uppercase;
            letter-spacing: 0.5px; margin-top: 3px;
        }
        .pagination-wrap {
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 10px; margin-top: 10px;
            padding: 8px 10px; background: var(--bg-deep); border: 1px solid var(--border);
        }
        .pg-info { font-size: 0.78em; color: var(--text-secondary); }
        .pg-info b { color: var(--text-primary); }
        .pg-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .pg-btn {
            background: var(--bg-primary); color: var(--terminal-green); border: 1px solid var(--border);
            padding: 6px 12px; cursor: pointer; font-family: inherit;
            font-size: 0.8em; font-weight: bold; min-height: 36px; min-width: 40px;
            transition: all 0.15s;
        }
        .pg-btn:hover:not(:disabled) { border-color: var(--terminal-green); background: rgba(34,197,94,0.1); }
        .pg-btn:disabled { color: var(--border-strong); border-color: var(--border); cursor: not-allowed; }
        .pg-num.active { background: var(--accent-green); color: var(--bg-primary); border-color: var(--terminal-green); }
        .pg-nums { display: inline-flex; gap: 4px; flex-wrap: wrap; }

        /* Mobile & Smartphone Responsive Styles */
        @media (max-width: 768px) {
            .scroll-hint { display: block; }
        }
        @media (max-width: 640px) {
            body { padding: 6px; font-size: 13px; }
            h1 { font-size: 1.2rem; }
            .stats-strip { grid-template-columns: repeat(2, 1fr); gap: 6px; }
            .stat-card .sv { font-size: 0.95em; }
            .top-bar { padding: 6px 10px; margin-bottom: 8px; }
            .burger-btn { min-width: 42px; height: 38px; }
            .refresh-btn { min-width: 42px; height: 38px; font-size: 1em; padding: 0 6px; }
            .cli-prompt { display: none; }
            .port-label { display: none; }
            .top-clock { font-size: 0.72em; }
            .section { padding: 10px 8px; margin-bottom: 10px; }
            .section h2 { font-size: 0.9em; padding-bottom: 8px; margin-bottom: 10px; }
            .cards { grid-template-columns: repeat(3, 1fr); gap: 6px; }
            .card { padding: 8px 4px; border-radius: 6px; }
            .card .num { font-size: 1.25em; }
            .card .label { font-size: 0.6em; }
            .card.size .num { font-size: 1.1em; }
            .chart-wrap { height: 280px; }
            .winner-card { padding: 12px 10px; }
            .winner-name { font-size: 1.6em; }
            .winner-grid { grid-template-columns: repeat(2, 1fr); gap: 6px; }
            .winner-grid .kv { padding: 8px 6px; }
            .winner-grid .kv .k { font-size: 0.65em; }
            .winner-grid .kv .v { font-size: 0.92em; }
            .winner-actions { flex-direction: column; gap: 8px; }
            .toggle-board { width: 100%; margin-top: 0; padding: 11px; font-size: 0.85em; }
            .trade-header { flex-direction: column; align-items: stretch; gap: 8px; }
            .mode-control-wrap { width: 100%; }            .live-standby-banner { flex-direction: column; align-items: flex-start; gap: 6px; padding: 10px; }
            .chart-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
            .ctrl-group select, .ctrl-group input[type="date"] {
                width: 100%; font-size: 16px; height: 46px; padding: 0 8px;
                /* 16px stops iOS Safari auto-zoom; fixed height keeps FROM/TO
                   the same box size as PAIR/TIMEFRAME (native date inputs
                   otherwise render taller with bigger internal font) */
                -webkit-appearance: none; appearance: none;
            }
            .reset-btn { grid-column: span 2; width: 100%; height: 46px; min-height: 46px; padding: 0; font-size: 0.85em; }
            .report-grid { grid-template-columns: 1fr; gap: 10px; }
            table { font-size: 0.78em; }
            th, td { padding: 7px 6px; }
            pre { font-size: 0.7em; overflow-x: auto; -webkit-overflow-scrolling: touch; }
        }
        @media (max-width: 380px) {
            .cards { grid-template-columns: repeat(2, 1fr); }
            .brand-name { font-size: 0.95em; }
            .active-crumb { display: none; }
        }
    """


CHART_JS = """
    let chartData = null;
    let boardChartData = null;          // { PAIR: {pair, tf, candles, aggregated} }
    let boardChartTarget = null;        // canvas id of the open board chart
    let boardChartRow = null;           // the <tr class="board-chart-row"> currently open
    let boardChartTrigger = null;       // the clicked pair cell

    // Sensible default date window per timeframe — switching TF auto-sets
    // these so each timeframe shows a visibly different (readable) chart.
    // null = full history (from the deep-backfill floor).
    const TF_WINDOW_DAYS = {"15m": 7, "1h": 30, "3h": 90, "6h": 180, "12h": 365, "1d": null};

    function tfWindow(tf) {
        const days = TF_WINDOW_DAYS[tf];
        // WIB calendar dates (UTC+7) regardless of browser timezone
        const wibNow = Math.floor(Date.now() / 1000) + 7 * 3600;
        const toStr = new Date(wibNow * 1000).toISOString().slice(0, 10);
        if (days === null || days === undefined) {
            return {from: "2024-01-01", to: toStr};  // full history
        }
        const fromStr = new Date((wibNow - days * 86400) * 1000).toISOString().slice(0, 10);
        return {from: fromStr, to: toStr};
    }

    function fmtPrice(p) {
        if (p >= 1000) return p.toLocaleString('en-US', {maximumFractionDigits: 0});
        if (p >= 1) return p.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 4});
        if (p >= 0.01) return p.toFixed(4);
        return p.toFixed(8);
    }

    function fmtTime(t, tf) {
        // WIB = UTC+7 — shift the epoch so the UTC string equals WIB wall-clock
        const d = new Date((t + 7 * 3600) * 1000);
        if (tf === '1d') return d.toISOString().slice(0, 10);
        return d.toISOString().slice(0, 16).replace('T', ' ');
    }

    async function loadPair(autoRange) {
        const pair = document.getElementById('pairSelect').value;
        const tf = document.getElementById('tfSelect').value;

        // On timeframe change, snap the date range to a sensible window for
        // that TF so the chart is readable (not 1400 sub-pixel points).
        if (autoRange) {
            const w = tfWindow(tf);
            document.getElementById('dateFrom').value = w.from;
            document.getElementById('dateTo').value = w.to;
        }

        const from = document.getElementById('dateFrom').value;
        const to = document.getElementById('dateTo').value;
        const info = document.getElementById('chartInfo');
        info.textContent = `Loading ${pair} ${tf}...`;
        try {
            let url = `/api/ohlcv/${pair}?tf=${tf}`;
            if (from) url += `&from_date=${from}`;
            if (to) url += `&to_date=${to}`;
            const res = await fetch(url);
            const data = await res.json();
            if (data.error) { info.textContent = data.error; chartData = null; drawChart(); return; }
            chartData = data;
            const agg = data.aggregated ? ' (aggregated)' : '';
            info.textContent = `${data.pair} ${data.tf}: ${data.candles.length} candles${agg} — ${fmtTime(data.candles[0].t, data.tf)} → ${fmtTime(data.candles[data.candles.length-1].t, data.tf)}`;
            drawChart();
        } catch (e) {
            info.textContent = 'Error: ' + e;
        }
    }

    function drawChart(mouseX, target) {
        // target: optional selector/id for a board-row chart. Default = the
        // Others-page candlestick chart.
        const canvas = target
            ? document.getElementById(target)
            : document.getElementById('chart');
        if (!canvas) return;
        const wrap = canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        const W = wrap.clientWidth, H = wrap.clientHeight;
        if (W === 0 || H === 0) return;
        canvas.width = W * dpr; canvas.height = H * dpr;
        canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        const data = target ? boardChartData : chartData;
        if (!data || !data.candles || !data.candles.length) {
            ctx.fillStyle = 'var(--text-secondary)';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No data', W / 2, H / 2);
            return;
        }

        const candles = data.candles;
        const tf = data.tf;
        const padL = 8, padR = 72, padT = 12, padB = 26, volH = 70;
        const chartW = W - padL - padR;
        const priceH = H - padT - padB - volH;

        let min = Infinity, max = -Infinity, maxVol = 0;
        for (const c of candles) {
            if (c.l < min) min = c.l;
            if (c.h > max) max = c.h;
            if (c.v > maxVol) maxVol = c.v;
        }
        const pad = (max - min) * 0.06 || max * 0.01 || 1;
        min -= pad; max += pad;

        const x = i => padL + (i + 0.5) * (chartW / candles.length);
        const y = p => padT + priceH - ((p - min) / (max - min)) * priceH;
        const volTop = padT + priceH + 10;

        // ---- Trade overlay: shade where the bot was IN a position ----
        //
        // Drawn BEFORE the candles so the shading sits behind them, and clipped
        // to the price pane so a tall band cannot bleed into the volume strip.
        //
        // Two problems this has to solve:
        //  1. A trade can be SHORTER than one candle (the sample median hold is
        //     2.6 bars and the minimum is 0.1). Scaling strictly by duration
        //     would make those invisible, so every band has a minimum width.
        //  2. The band must not hide the price action the user came to see,
        //     hence low alpha plus a thin, opaque entry/exit edge.
        const trades = (data.trades || []);
        const bandInfo = [];
        if (trades.length) {
            const times = candles.map(c => c.t);
            const barSec = times.length > 1
                ? Math.max(60, times[times.length - 1] - times[0]) / (times.length - 1)
                : 900;
            const tFirst = times[0], tLast = times[times.length - 1] + barSec;
            const timeToX = ts => padL + ((ts - tFirst) / (tLast - tFirst)) * chartW;

            ctx.save();
            ctx.beginPath();
            ctx.rect(padL, padT, chartW, priceH);
            ctx.clip();
            for (const t of trades) {
                if (t.exit_time < tFirst || t.entry_time > tLast) continue;
                let x1 = timeToX(Math.max(t.entry_time, tFirst));
                let x2 = timeToX(Math.min(t.exit_time, tLast));
                const minW = Math.max(3, (chartW / candles.length) * 0.6);
                if (x2 - x1 < minW) {
                    const mid = (x1 + x2) / 2;
                    x1 = mid - minW / 2; x2 = mid + minW / 2;
                }
                const win = t.net_pnl_idr >= 0;
                ctx.fillStyle = win ? 'rgba(34,197,94,0.16)' : 'rgba(239,68,68,0.16)';
                ctx.fillRect(x1, padT, x2 - x1, priceH);
                ctx.strokeStyle = win ? 'rgba(34,197,94,0.85)' : 'rgba(239,68,68,0.85)';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(x1, padT); ctx.lineTo(x1, padT + priceH); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x2, padT); ctx.lineTo(x2, padT + priceH); ctx.stroke();
                bandInfo.push({x1: x1, x2: x2, trade: t});
            }
            ctx.restore();

            if (bandInfo.length) {
                ctx.fillStyle = 'rgba(34,197,94,0.16)';
                ctx.fillRect(padL + 4, padT + 4, 10, 10);
                ctx.fillStyle = 'var(--text-secondary)';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText('win', padL + 18, padT + 12);
                ctx.fillStyle = 'rgba(239,68,68,0.16)';
                ctx.fillRect(padL + 48, padT + 4, 10, 10);
                ctx.fillStyle = 'var(--text-secondary)';
                ctx.fillText('loss', padL + 62, padT + 12);
                ctx.fillStyle = 'var(--text-muted)';
                ctx.fillText('shaded = bot in position', padL + 100, padT + 12);
            }
        }

        // Grid + price labels (right side)
        ctx.strokeStyle = 'rgba(51,65,85,0.6)';
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'left';
        for (let g = 0; g <= 5; g++) {
            const p = min + (max - min) * g / 5;
            const yy = y(p);
            ctx.beginPath();
            ctx.moveTo(padL, yy); ctx.lineTo(padL + chartW, yy);
            ctx.stroke();
            ctx.fillText(fmtPrice(p), padL + chartW + 6, yy + 3);
        }

        // Time labels (bottom)
        ctx.textAlign = 'center';
        const nLabels = Math.min(6, candles.length);
        for (let g = 0; g < nLabels; g++) {
            const i = Math.floor(g * (candles.length - 1) / Math.max(1, nLabels - 1));
            const xx = x(i);
            ctx.strokeStyle = 'rgba(51,65,85,0.3)';
            ctx.beginPath();
            ctx.moveTo(xx, padT); ctx.lineTo(xx, volTop + volH);
            ctx.stroke();
            ctx.fillStyle = 'var(--text-muted)';
            ctx.fillText(fmtTime(candles[i].t, tf).slice(5, 10), xx, H - 8);
        }

        // Candles
        const bw = Math.max(1, (chartW / candles.length) * 0.7);
        for (let i = 0; i < candles.length; i++) {
            const c = candles[i];
            const up = c.c >= c.o;
            const col = up ? '#26a69a' : '#ef5350';
            ctx.strokeStyle = col;
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.moveTo(x(i), y(c.h));
            ctx.lineTo(x(i), y(c.l));
            ctx.stroke();
            const yO = y(c.o), yC = y(c.c);
            ctx.fillRect(x(i) - bw / 2, Math.min(yO, yC), bw, Math.max(1, Math.abs(yC - yO)));
        }

        // Volume bars
        for (let i = 0; i < candles.length; i++) {
            const c = candles[i];
            const up = c.c >= c.o;
            ctx.fillStyle = up ? 'rgba(38,166,154,0.45)' : 'rgba(239,83,80,0.45)';
            const vh = maxVol > 0 ? (c.v / maxVol) * volH : 0;
            ctx.fillRect(x(i) - bw / 2, volTop + volH - vh, bw, vh);
        }

        // Crosshair + tooltip
        const tipId = target ? 'boardTooltip' : 'tooltip';
        const tooltip = document.getElementById(tipId);
        if (tooltip && mouseX !== undefined && mouseX !== null) {
            const idx = Math.round((mouseX - padL) / (chartW / candles.length) - 0.5);
            if (idx >= 0 && idx < candles.length) {
                const c = candles[idx];
                const xx = x(idx);
                ctx.strokeStyle = 'rgba(148,163,184,0.5)';
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(xx, padT); ctx.lineTo(xx, volTop + volH);
                ctx.moveTo(padL, y(c.c)); ctx.lineTo(padL + chartW, y(c.c));
                ctx.stroke();
                ctx.setLineDash([]);
                const pct = ((c.c - c.o) / c.o * 100).toFixed(2);
                const col = c.c >= c.o ? '#26a69a' : '#ef5350';

                // Which trade window is under the cursor? The overlay answers
                // "what did the bot do here", and the reason for the exit
                // matters more than the shading colour.
                const hit = bandInfo.find(b => xx >= b.x1 && xx <= b.x2);
                let tradeHtml = '';
                if (hit) {
                    const t = hit.trade;
                    const ok = t.net_pnl_idr >= 0;
                    const c2 = ok ? '#22c55e' : '#ef4444';
                    const mins = Math.max(0, Math.round((t.exit_time - t.entry_time) / 60));
                    const held = mins >= 60 ? `${(mins / 60).toFixed(1)}h` : `${mins}m`;
                    tradeHtml =
                        `<div style="margin-top:4px;padding-top:4px;border-top:1px solid rgba(148,163,184,.35)">` +
                        `<b style="color:${c2}">${t.hunt_id}</b> &nbsp;` +
                        // net_pnl_pct is ALREADY a percentage (-5.2249 means
                        // -5.22%), not a fraction. Multiplying by 100 here
                        // printed +128% for a +1.28% trade.
                        `<span style="color:${c2}">${t.net_pnl_pct >= 0 ? '+' : ''}` +
                        `${t.net_pnl_pct.toFixed(2)}%</span><br>` +
                        `${t.exit_reason}<br>` +
                        `held ${held} &nbsp; FW ${t.framework || '?'}` +
                        `</div>`;
                }

                tooltip.style.display = 'block';
                tooltip.innerHTML =
                    `<b>${fmtTime(c.t, tf)} WIB</b><br>` +
                    `O ${fmtPrice(c.o)} &nbsp; H ${fmtPrice(c.h)}<br>` +
                    `L ${fmtPrice(c.l)} &nbsp; C <span style="color:${col}">${fmtPrice(c.c)}</span><br>` +
                    `<span style="color:${col}">${pct >= 0 ? '+' : ''}${pct}%</span> &nbsp; Vol ${fmtPrice(c.v)}` +
                    tradeHtml;
                const tipW = hit ? 180 : 160;
                let tx = xx + 12;
                if (tx + tipW > W) {
                    tx = Math.max(4, xx - tipW - 12);
                }
                tooltip.style.left = tx + 'px';
                tooltip.style.top = Math.max(4, Math.min(H - (hit ? 120 : 85), y(c.c) - 10)) + 'px';
            } else {
                if (tooltip) tooltip.style.display = 'none';
            }
        } else {
            if (tooltip) tooltip.style.display = 'none';
        }
    }

    function resetFilters() {
        document.getElementById('pairSelect').value = "BTCIDR";
        document.getElementById('tfSelect').value = "15m";
        loadPair(true);  // reload with default pair/TF + auto date window
    }

    function initChart() {
        const canvas = document.getElementById('chart');
        if (!canvas) return;

        function updateHover(clientX) {
            const rect = canvas.getBoundingClientRect();
            drawChart(clientX - rect.left);
        }
        canvas.addEventListener('mousemove', e => updateHover(e.clientX));
        canvas.addEventListener('mouseleave', () => drawChart());

        // Smartphone touch interaction for candlestick inspection
        canvas.addEventListener('touchstart', e => {
            if (e.touches && e.touches.length > 0) updateHover(e.touches[0].clientX);
        }, {passive: true});
        canvas.addEventListener('touchmove', e => {
            if (e.touches && e.touches.length > 0) updateHover(e.touches[0].clientX);
        }, {passive: true});
        canvas.addEventListener('touchend', () => drawChart(), {passive: true});

        window.addEventListener('resize', () => drawChart());
        window.addEventListener('orientationchange', () => setTimeout(drawChart, 150));

        // Toggle filter board + model report
        function bindToggle(btnId, sectionId, showText, hideText) {
            const btn = document.getElementById(btnId);
            const sec = document.getElementById(sectionId);
            if (!btn || !sec) return;
            btn.addEventListener('click', () => {
                const expanding = sec.classList.contains('board-collapsed');

                // Anchor the button's position before the layout changes.
                const before = btn.getBoundingClientRect().top;

                if (expanding) {
                    sec.classList.remove('board-collapsed');
                    btn.textContent = hideText;
                } else {
                    sec.classList.add('board-collapsed');
                    btn.textContent = showText;
                }

                // Keep the BUTTON where it was on screen.
                //
                // The previous version scrolled to the expanded section, which for a
                // panel far down the page meant the view jumped away and the button
                // left the screen - indistinguishable from a click that did nothing.
                // Restoring the button's viewport position means the thing you just
                // pressed stays under your finger and the panel opens in place.
                const after = btn.getBoundingClientRect().top;
                if (after !== before) {
                    window.scrollBy(0, after - before);
                }
            });
        }

        // Scroll an element just below the sticky top bar, with a fallback for
        // browsers without smooth scrolling.
        function scrollSectionIntoView(el) {
            const bar = document.querySelector('.top-bar');
            const offset = (bar ? bar.offsetHeight : 0) + 12;
            const y = el.getBoundingClientRect().top + window.scrollY - offset;
            try {
                window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
            } catch (e) {
                window.scrollTo(0, Math.max(0, y));
            }
        }

        bindToggle('toggleBoard', 'boardSection', '[+] Show Filter Board', '[-] Hide Filter Board');

        document.getElementById('resetBtn').addEventListener('click', resetFilters);
        // Timeframe change -> auto-set a sensible date window for that TF.
        // Pair/date changes -> keep the current range as-is.
        document.getElementById('tfSelect').addEventListener('change', () => loadPair(true));
        document.getElementById('pairSelect').addEventListener('change', () => loadPair(false));
        document.getElementById('dateFrom').addEventListener('change', () => loadPair(false));
        document.getElementById('dateTo').addEventListener('change', () => loadPair(false));

        // Restore view from URL hash if set.
        // A #hunt-... hash is handled by openHuntPage() (its own initialiser),
        // so do NOT force switchPage('home') there — that would overwrite the
        // breadcrumb and hide the hunt page underneath it.
        //
        // Every page switchPage() can navigate to must be listed here. This
        // knew only about '#others' while switchPage writes '#settings' and
        // '#analytics', so a plain F5 on Settings or Analytics fell through to
        // the else branch and landed on Home.
        if (window.location.hash === '#others' || window.location.hash === '#analytics') {
            switchPage('others', false);
        } else if (window.location.hash === '#settings') {
            switchPage('settings', false);
        } else if (/^#hunt-/i.test(window.location.hash || '')) {
            // leave the view to the hunt-page initialiser
        } else {
            switchPage('home', false);
        }
    }

    function toggleNav() {
        const drawer = document.getElementById('navDrawer');
        const overlay = document.getElementById('navOverlay');
        if (!drawer || !overlay) return;
        drawer.classList.toggle('open');
        overlay.classList.toggle('open');
    }

    function switchPage(page, closeDrawer = true) {
        const pages = {
            home: document.getElementById('pageHome'),
            others: document.getElementById('pageOthers'),
            settings: document.getElementById('pageSettings')
        };
        const navs = {
            home: document.getElementById('navHome'),
            others: document.getElementById('navOthers'),
            settings: document.getElementById('navSettings')
        };
        const activeCrumb = document.getElementById('activeCrumb');
        const drawer = document.getElementById('navDrawer');
        const overlay = document.getElementById('navOverlay');
        const crumbMap = {home: '/ home', others: '/ analytics', settings: '/ settings'};
        const hashMap = {home: '#home', others: '#analytics', settings: '#settings'};

        Object.values(pages).forEach(el => { if (el) el.style.display = 'none'; });
        Object.values(navs).forEach(el => { if (el) el.classList.remove('active'); });

        if (pages[page]) pages[page].style.display = 'block';
        if (navs[page]) navs[page].classList.add('active');
        if (activeCrumb) activeCrumb.textContent = crumbMap[page] || '/ home';
        history.replaceState(null, '', hashMap[page] || '#home');

        if (page === 'others') {
            setTimeout(() => {
                if (typeof drawChart === 'function') {
                    if (!chartData) { loadPair(true); } else { drawChart(); }
                }
            }, 60);
        }

        if (closeDrawer) {
            if (drawer) drawer.classList.remove('open');
            if (overlay) overlay.classList.remove('open');
        }
    }


    // ----------------------------------------------------------------------
    // Control plane: START / STOP hunting
    // ----------------------------------------------------------------------
    let ctrlState = { mode: 'RUNNING', framework: 'A' };

    // ---------------------------------------------------------------------
    // The TAB is the mode. Selecting PAPER shows the paper ledger and a
    // paper START/STOP; selecting LIVE shows the live ledger and a live
    // START/STOP. Exactly one button on screen at a time.
    //
    // This is safe because the tab only decides WHICH button you see - pressing
    // it is still what arms anything, and LIVE still goes to its own endpoint
    // with a typed confirmation.
    // ---------------------------------------------------------------------
    var selectedMode = 'paper';
    var currentLedger = 'paper';

    function selectMode(which) {
        var want = (which === 'live') ? 'live' : 'paper';
        var runningExec = (ctrlState.mode === 'RUNNING') ? ctrlState.exec : null;

        // Refuse to hide the stop button of a running engine by switching tabs.
        if (runningExec && runningExec !== (want === 'live' ? 'LIVE' : 'PAPER')) {
            window.alert('The ' + runningExec + ' engine is running. Stop it before '
                         + 'switching to ' + (want === 'live' ? 'LIVE' : 'PAPER') + '.');
            return;
        }
        selectedMode = want;
        currentLedger = want;
        paintControl();
        refreshBuyButton();
    }

    function paintControl() {
        var sw = document.getElementById('ctrlSwitch');
        var lbl = document.getElementById('ctrlLabel');
        var hint = document.getElementById('ctrlHint');
        if (!sw || !lbl) return;

        var isLive = (ctrlState.exec === 'LIVE');
        var wantLive = (selectedMode === 'live');
        var running = (ctrlState.mode === 'RUNNING');
        var blocked = ctrlState.blocked && ctrlState.blocked.reason;
        var thisOneRuns = running && (isLive === wantLive);

        // the page carries the live treatment regardless of which tab is open
        document.body.classList.toggle('live-armed', isLive);
        var lb = document.getElementById('liveBadge');
        if (lb) {
            lb.innerHTML = isLive
                ? '<span class="live-badge">\u25CF LIVE \u00B7 REAL MONEY</span>' : '';
        }

        // panels + tab highlight follow the selection (or the running engine)
        var shown = running ? (isLive ? 'live' : 'paper') : selectedMode;
        currentLedger = shown;
        var paperWrap = document.getElementById('paperTradeWrap');
        var liveWrap = document.getElementById('liveTradeWrap');
        var tabP = document.getElementById('tabPaper');
        var tabL = document.getElementById('tabLive');
        if (paperWrap) paperWrap.style.display = (shown === 'paper') ? 'block' : 'none';
        if (liveWrap) liveWrap.style.display = (shown === 'live') ? 'block' : 'none';
        if (tabP) { tabP.classList.toggle('active', shown === 'paper');
                    tabP.setAttribute('aria-selected', shown === 'paper'); }
        if (tabL) { tabL.classList.toggle('active', shown === 'live');
                    tabL.setAttribute('aria-selected', shown === 'live'); }
        if (shown === 'live') loadLiveBalance();

        // ONE button, meaning decided by the tab
        if (thisOneRuns) {
            sw.style.display = '';
            sw.classList.remove('stopped');
            sw.classList.toggle('live', isLive);
            sw.classList.toggle('blocked', !!blocked);
            sw.title = 'Running. Click to stop.';
            lbl.textContent = blocked
                ? '\u25A0 STOP \u2014 ENGINE BLOCKED'
                : (isLive ? '\u25A0 STOP LIVE' : '\u25A0 STOP PAPER');
        } else if (running) {
            sw.style.display = 'none';
        } else {
            sw.style.display = '';
            sw.classList.remove('live', 'blocked');
            sw.classList.add('stopped');
            sw.classList.toggle('ctrl-live', wantLive);
            sw.title = wantLive ? 'Arm LIVE trading (asks for confirmation).'
                                : 'Start the paper engine.';
            lbl.textContent = wantLive ? '\u25B6 START LIVE' : '\u25B6 START PAPER';
        }

        if (hint) {
            hint.textContent = blocked
                ? ('NOT TRADING: ' + blocked)
                : (thisOneRuns
                   ? ('framework ' + ctrlState.framework + ' \u00B7 every 60s'
                      + (isLive ? ' \u00B7 LIVE' : ' \u00B7 paper'))
                   : (running
                      ? ('the ' + (isLive ? 'LIVE' : 'PAPER') + ' engine is running \u2014 '
                         + 'stop it to use this tab')
                      : ('framework ' + ctrlState.framework + ' \u00B7 stopped')));
        }
    }

    function refreshControl() {
        fetch('/api/control', { cache: 'no-store' })
            .then(r => r.json())
            .then(d => { ctrlState = d; paintControl(); })
            .catch(() => {});
    }

    function loadLiveBalance() {
        // Real exchange balance, fetched server-side. Shown only while LIVE is
        // armed; the paper balance belongs to the SIMULATION panel and must
        // never be presented as if it were real money.
        if (ctrlState.exec !== 'LIVE') return;
        var main = document.getElementById('liveBalIdr');
        var hold = document.getElementById('liveBalHold');
        var err = document.getElementById('liveBalErr');
        fetch('/api/live/balance', { cache: 'no-store' })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (!d.ok) {
                    if (main) main.textContent = 'unavailable';
                    if (err) err.textContent = d.error || 'cannot read balance';
                    return;
                }
                if (err) err.textContent = '';
                if (main) {
                    main.textContent = 'IDR ' + Number(d.idr || 0)
                        .toLocaleString(undefined, {maximumFractionDigits: 0});
                }
                if (hold) {
                    hold.textContent = (d.idr_on_hold > 0)
                        ? (' (IDR ' + Number(d.idr_on_hold)
                             .toLocaleString(undefined, {maximumFractionDigits: 0})
                           + ' locked in open orders)')
                        : '';
                }
            })
            .catch(function (e) {
                if (main) main.textContent = 'unavailable';
                if (err) err.textContent = String(e);
            });
    }

    function refreshBuyButton() {
        // A manual buy only exists meaningfully in LIVE with nothing open: in
        // paper it would be a simulated order dressed as a real one, and a
        // second position is refused by the engine anyway.
        var can = (ctrlState.exec === 'LIVE')
                  && (ctrlState.mode === 'RUNNING')
                  && !(ctrlState.position && ctrlState.position.symbol)
                  && !ctrlState.blocked;

        var b = document.getElementById('buyNow');
        if (b) {
            b.style.display = can ? 'inline-flex' : 'none';
            if (can) b.title = 'Market buy at the best ask. Real money.';
        }

        // One BUY per row in the filter board, same rule. Shown for EVERY pair,
        // not just the eligible ones - choosing a pair the funnel rejected is
        // the operator's decision to make, and the engine re-checks the caps
        // regardless of what the table thinks.
        Array.from(document.querySelectorAll('.row-buy')).forEach(function (rb) {
            rb.style.display = can ? 'inline-block' : 'none';
            if (can && !rb.title) {
                rb.title = 'Market buy ' + rb.getAttribute('data-sym')
                           + ' at the best ask. Real money.';
            }
        });
        var note = document.getElementById('actionNote');
        if (note) {
            note.style.display = can ? 'none' : 'block';
        }
    }

    function buyNow(symbol) {
        var NL = String.fromCharCode(10);
        var msg = 'MARKET BUY ' + symbol + '?' + NL + NL;
        msg += 'This spends REAL money at whatever the book gives.' + NL;
        msg += 'A market order cannot be undone.' + NL + NL;
        msg += 'Type ' + symbol + ' to confirm.';
        var typed = window.prompt(msg, '');
        if (typed === null) return;
        if (String(typed).trim().toUpperCase() !== symbol) {
            window.alert('Not bought: you must type ' + symbol + ' exactly.');
            return;
        }
        var b = document.getElementById('buyNow');
        if (b) { b.disabled = true; b.textContent = 'BUYING...'; }
        fetch('/api/position/buy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: symbol, confirm: symbol})
        })
        .then(function (r) { return r.json().then(function (d) {
            return {status: r.status, d: d}; }); })
        .then(function (res) {
            if (res.status === 409 && res.d.busy) {
                setTimeout(function () { buyNow(symbol); }, 2500);
                return;
            }
            if (!res.d.ok) {
                window.alert('Not bought: ' + (res.d.error || 'unknown'));
                if (b) { b.disabled = false; b.textContent = 'BUY ' + symbol + ' NOW'; }
                refreshControl();
                return;
            }
            window.alert(res.d.message || 'bought');
            refreshDashboard();
        })
        .catch(function (e) {
            window.alert('Buy failed: ' + e);
            if (b) { b.disabled = false; b.textContent = 'BUY ' + symbol + ' NOW'; }
        });
    }

    function onStartLive() {
        // Two stages: fetch the preflight, show exactly what will happen, then
        // require the word LIVE. A single confirm() is too easy to click through
        // when the thing behind it spends real money.
        var NL = String.fromCharCode(10);
        fetch('/api/live/preflight', { cache: 'no-store' })
            .then(function (r) { return r.json(); })
            .then(function (pf) {
                if (!pf.ok) {
                    var problems = (pf.problems || []).join(NL);
                    window.alert('Cannot arm LIVE:' + NL + NL + problems + NL + NL
                                 + 'Fix these, then try again.');
                    return;
                }
                var size = pf.trade_size_idr || 0;
                var msg = 'ARM LIVE TRADING?' + NL + NL;
                msg += 'Executor:  REAL ORDERS on Indodax' + NL;
                msg += 'Per trade: IDR ' + size.toLocaleString(undefined, {maximumFractionDigits: 0}) + NL;
                msg += 'Balance:   verified' + NL;
                msg += 'Clock:     verified' + NL;
                (pf.warnings || []).forEach(function (w) { msg += NL + 'WARNING: ' + w; });
                msg += NL + NL + 'Type the word LIVE to confirm.';
                var typed = window.prompt(msg, '');
                if (typed === null) return;
                if (String(typed).trim().toUpperCase() !== 'LIVE') {
                    window.alert('Not armed: you must type LIVE exactly.');
                    return;
                }
                fetch('/api/control/start-live', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({confirm: 'LIVE'})
                })
                .then(function (r) { return r.json().then(function (d) {
                    return {status: r.status, d: d}; }); })
                .then(function (res) {
                    if (!res.d.ok) {
                        window.alert('Not armed: ' + (res.d.error || 'unknown'));
                        refreshControl();
                        return;
                    }
                    window.alert(res.d.message || 'LIVE armed.');
                    refreshDashboard();
                })
                .catch(function (e) {
                    window.alert('Live start failed: ' + e);
                });
            })
            .catch(function (e) { window.alert('Preflight failed: ' + e); });
    }

    function stopLiveWithChoice() {
        // "C": never decide for the operator what happens to an open position.
        var NL = String.fromCharCode(10);
        var p = (ctrlState.position || {});
        var label = p.symbol
            ? (p.symbol + ' - ' + (p.coins || 0) + ' coins - entry ' + p.entry_price)
            : 'an open position';
        var sell = window.confirm(
            'STOP LIVE' + NL + NL + 'There is ' + label + '.' + NL + NL
            + 'OK     = sell it now at market, then stop' + NL
            + 'Cancel = stop and LEAVE IT OPEN on the exchange');
        return sell ? 'sell' : 'leave';
    }

    function onControlSwitch() {
        const sw = document.getElementById('ctrlSwitch');
        const goingDown = (ctrlState.mode === 'RUNNING');
        var stopChoice = null;

        // Starting: the SELECTED TAB decides which engine. LIVE keeps its own
        // endpoint and its typed confirmation - the shared button only chooses
        // the action, it never skips the check.
        if (!goingDown) {
            if (selectedMode === 'live') {
                onStartLive();
                return;
            }
        }

        if (goingDown) {
            if (ctrlState.exec === 'LIVE') {
                stopChoice = stopLiveWithChoice();
                if (stopChoice === null) return;
            } else {
                const pos = document.querySelector('tr.open-trade');
                const warn = pos
                    ? 'STOP will market-sell the open position NOW, then pause hunting.'
                    : 'Stop and Pause Hunting?';
                if (!window.confirm(warn)) return;
            }
        }
        if (sw) sw.classList.add('busy');
        const url = goingDown ? '/api/control/stop' : '/api/control/start';
        const init = { method: 'POST' };
        if (stopChoice) {
            init.headers = {'Content-Type': 'application/json'};
            init.body = JSON.stringify({position: stopChoice});
        }
        fetch(url, init)
            .then(r => r.json().then(d => ({ status: r.status, d })))
            .then(({ status, d }) => {
                if (sw) sw.classList.remove('busy');
                if (status === 409 && d.busy) {
                    // A tick holds the engine lock — retry once shortly.
                    setTimeout(onControlSwitch, 2500);
                    return;
                }
                if (!d.ok) {
                    window.alert('Control failed: ' + (d.error || 'unknown'));
                    refreshControl();
                    return;
                }
                if (d.message) window.alert(d.message);
                // Reload so the trade table reflects the liquidation.
                // Goes through refreshDashboard() so the current page (Settings,
                // Analytics or Home) is preserved - a bare location.reload()
                // dropped the user back to Home.
                refreshDashboard();
            })
            .catch(e => {
                if (sw) sw.classList.remove('busy');
                window.alert('Control request failed: ' + e);
            });
    }

    refreshControl();

    // ----------------------------------------------------------------------
    // Framework activation (only permitted while STOPPED)
    // ----------------------------------------------------------------------
    function activateFramework(name) {
        fetch('/api/frameworks/' + encodeURIComponent(name) + '/activate',
              { method: 'POST' })
            .then(r => r.json().then(d => ({ status: r.status, d })))
            .then(({ status, d }) => {
                if (status === 409) {
                    window.alert(d.error || 'Stop hunting before switching framework.');
                    return;
                }
                if (!d.ok) {
                    window.alert('Switch failed: ' + (d.error || 'unknown'));
                    return;
                }
                ctrlState.framework = d.framework;
                // same reason as above: keep the user where they were
                refreshDashboard();
            })
            .catch(e => window.alert('Switch request failed: ' + e));
    }

    // ----------------------------------------------------------------------
    // Settings — first-time setup actions
    // ----------------------------------------------------------------------
    let setupPolling = null;

    function setCronStatus(msg, kind) {
        const el = document.getElementById("cronStatus");
        if (!el) return;
        el.textContent = msg;
        el.style.color = kind === "error" ? "#f87171" : "var(--text-secondary)";
    }

    function toggleCron(job, checkbox) {
        const enabled = checkbox.checked;
        const action = enabled ? "enable" : "disable";
        setCronStatus("Updating " + job + "...");
        fetch("/api/cron/" + job + "/" + action, {method: "POST"})
            .then(r => r.json().then(data => ({ok: r.ok, data})))
            .then(resp => {
                if (resp.ok && resp.data.ok) {
                    setCronStatus(job + " " + (enabled ? "enabled" : "disabled"));
                } else {
                    checkbox.checked = !enabled;
                    setCronStatus(resp.data.error || ("Failed to " + action + " " + job), "error");
                }
            })
            .catch(err => {
                checkbox.checked = !enabled;
                setCronStatus(String(err), "error");
            });
    }

    function runSetup(action) {
        const btns = document.querySelectorAll('#settingsPanel .su-btn');
        btns.forEach(b => b.disabled = true);       // no double-click launches

        fetch('/api/setup/' + encodeURIComponent(action) + '/run', { method: 'POST' })
            .then(r => r.json().then(d => ({ status: r.status, d })))
            .then(({ status, d }) => {
                if (!d.ok) {
                    // 409 means another heavy job holds the lock; say which.
                    window.alert(d.error || 'Could not start.');
                    btns.forEach(b => b.disabled = false);
                    return;
                }
                showToast('Started: ' + action);
                pollSetup();
            })
            .catch(e => {
                window.alert('Start request failed: ' + e);
                btns.forEach(b => b.disabled = false);
            });
    }

    function renderSetup(d) {
        // Update just the state tags and progress bars in place, so a poll does
        // not reload the page out from under the user.
        const panel = document.getElementById('settingsPanel');
        if (!panel) return;

        const TAG = { not_run: 'su-idle', running: 'su-run',
                      done: 'su-done', failed: 'su-fail' };
        const WORD = { not_run: 'not run', running: 'running',
                       done: 'done', failed: 'failed' };

        (d.order || []).forEach((name, i) => {
            const row = panel.querySelectorAll('.su-row')[i];
            if (!row) return;
            const a = (d.actions || {})[name] || {};
            const tag = row.querySelector('.su-tag');
            if (tag) {
                tag.className = 'su-tag ' + (TAG[a.state] || 'su-idle');
                tag.textContent = WORD[a.state] || a.state;
            }
            const btn = row.querySelector('.su-btn');
            if (btn) {
                const busyElsewhere = d.busy && d.busy !== name;
                btn.disabled = (a.state === 'running') || !!busyElsewhere;
                btn.textContent = { not_run: 'RUN', failed: 'RETRY',
                                    done: 'RUN AGAIN', running: 'RUNNING' }[a.state] || 'RUN';
            }
            // progress bar
            let prog = row.querySelector('.su-prog');
            const p = a.progress;
            if (a.state === 'running' && p && p.total) {
                if (!prog) {
                    prog = document.createElement('div');
                    prog.className = 'su-prog';
                    prog.innerHTML = "<div class='su-bar'></div>";
                    row.querySelector('.su-main').appendChild(prog);
                }
                const pct = Math.round(p.done / p.total * 100);
                prog.querySelector('.su-bar').style.width = pct + '%';
            } else if (prog && a.state !== 'running') {
                prog.remove();
            }
        });

        const banner = panel.querySelector('.su-banner');
        if (banner) {
            if (d.ready) {
                banner.className = 'su-banner ok';
                banner.textContent = 'Setup complete — every data action has finished.';
            } else {
                banner.className = 'su-banner warn';
                banner.textContent = 'Setup incomplete — still to do: ' +
                    (d.missing_labels || []).join(', ');
            }
        }
    }

    function pollSetup() {
        if (setupPolling) clearInterval(setupPolling);
        const tick = () => {
            fetch('/api/setup')
                .then(r => r.json())
                .then(d => {
                    renderSetup(d);
                    // stop polling when nothing is running and everything is done
                    const anyRunning = Object.values(d.actions || {})
                        .some(a => a.state === 'running');
                    if (!anyRunning && setupPolling) {
                        clearInterval(setupPolling);
                        setupPolling = null;
                    }
                })
                .catch(() => {});
        };
        tick();
        setupPolling = setInterval(tick, 3000);
    }

    // Resume polling on load if a job is already running (page reload mid-job).
    document.addEventListener('DOMContentLoaded', () => {
        fetch('/api/setup')
            .then(r => r.json())
            .then(d => {
                const anyRunning = Object.values(d.actions || {})
                    .some(a => a.state === 'running');
                if (anyRunning) pollSetup();
            })
            .catch(() => {});
    });

    // ----------------------------------------------------------------------
    // Refresh — re-fetch the page from the server without an F5.
    // Preserves page (home/others), trading mode (paper/live) and scroll.
    // ----------------------------------------------------------------------
    function showToast(msg) {
        const t = document.getElementById('toast');
        if (!t) return;
        t.textContent = msg;
        t.classList.add('show');
        clearTimeout(showToast._tid);
        showToast._tid = setTimeout(() => t.classList.remove('show'), 1600);
    }

    function currentPage() {
        // Every page must be listed here. This knew only about 'others' and
        // 'home', so a refresh pressed while on Settings recorded the page as
        // 'home' and the reload dropped the user back to Home.
        const pages = ['pageSettings', 'pageOthers', 'pageHome'];
        const names = {pageSettings: 'settings', pageOthers: 'others', pageHome: 'home'};
        for (const id of pages) {
            const el = document.getElementById(id);
            if (el && getComputedStyle(el).display !== 'none') return names[id];
        }
        return 'home';
    }

    function refreshDashboard() {
        const btn = document.getElementById('refreshBtn');
        if (btn) btn.classList.add('spinning');

        // Remember where the user was, so the reload lands back on it.
        // NOT the executor: that is a fact from the control file, and storing
        // it here is what made the old view-toggle look like a real switch.
        try {
            sessionStorage.setItem('hunterRestore', JSON.stringify({
                page: currentPage(),
                scrollY: window.scrollY || 0
            }));
        } catch (e) {}

        // cache: 'reload' bypasses the browser cache so data is fresh.
        fetch(window.location.pathname, { cache: 'reload' })
            .then(() => window.location.reload())
            .catch(() => window.location.reload());
    }

    // Restore the pre-refresh view (only set by the refresh button).
    (function restoreAfterRefresh() {
        let raw = null;
        try { raw = sessionStorage.getItem('hunterRestore'); } catch (e) {}
        if (!raw) return;
        try { sessionStorage.removeItem('hunterRestore'); } catch (e) {}
        let st;
        try { st = JSON.parse(raw); } catch (e) { return; }

        if (st.page === 'others') {
            switchPage('others', false);
        } else if (st.page === 'settings') {
            switchPage('settings', false);
        } else {
            switchPage('home', false);
        }
        if (st.scrollY) {
            // let layout settle before restoring scroll
            requestAnimationFrame(() => window.scrollTo(0, st.scrollY));
            setTimeout(() => window.scrollTo(0, st.scrollY), 120);
        }
        showToast('Data refreshed · ' + new Date().toLocaleTimeString('en-GB'));
    })();

    // ----------------------------------------------------------------------
    // Single-hunt performance page — PnL % over the life of one hunt
    // ----------------------------------------------------------------------
    let huntChartData = null;
    let activeHuntDetailRow = null;
    let activeHuntTrigger = null;

    function openHuntPage(huntId, trigger) {
        if (!huntId || huntId === '—' || huntId === '•' || !trigger) return;
        const parentRow = trigger.closest('tr');
        if (!parentRow) return;

        // Clicking the same Hunt ID closes its inline graph; clicking another
        // moves the single expanded panel to the new hunt.
        if (activeHuntDetailRow && activeHuntTrigger === trigger) {
            closeHuntPage();
            return;
        }
        if (activeHuntDetailRow) closeHuntPage();

        const table = parentRow.closest('table');
        const colCount = table ? table.querySelectorAll('thead th').length : 10;
        const detailRow = document.createElement('tr');
        detailRow.className = 'hunt-detail-row';
        const cell = document.createElement('td');
        cell.colSpan = colCount;
        const template = document.getElementById('huntDetailTemplate');
        cell.appendChild(template.content.cloneNode(true));
        detailRow.appendChild(cell);
        parentRow.insertAdjacentElement('afterend', detailRow);
        activeHuntDetailRow = detailRow;
        activeHuntTrigger = trigger;
        trigger.setAttribute('aria-expanded', 'true');

        const root = detailRow;
        root.querySelector('#huntTitleId').textContent = huntId;
        root.querySelector('#huntTitleSym').textContent = '';
        root.querySelector('#huntKv').innerHTML = '<span style="color:var(--text-muted);">loading…</span>';
        huntChartData = null;
        initHuntChart(root.querySelector('#huntChart'));
        drawHuntChart();

        fetch('/api/hunt/' + encodeURIComponent(huntId))
            .then(r => r.json())
            .then(d => {
                if (!activeHuntDetailRow || !activeHuntDetailRow.isConnected) return;
                huntChartData = d;
                renderHuntMeta(d);
                drawHuntChart();
            })
            .catch(() => {
                if (activeHuntDetailRow && activeHuntDetailRow.isConnected) {
                    activeHuntDetailRow.querySelector('#huntKv').innerHTML =
                        '<span style="color:var(--accent-red);">failed to load hunt data</span>';
                }
            });
    }

    function closeHuntPage() {
        if (activeHuntDetailRow) activeHuntDetailRow.remove();
        activeHuntDetailRow = null;
        huntChartData = null;
        if (activeHuntTrigger) {
            activeHuntTrigger.setAttribute('aria-expanded', 'false');
            activeHuntTrigger.focus({preventScroll: true});
        }
        activeHuntTrigger = null;
    }

    function renderHuntMeta(d) {
        const root = activeHuntDetailRow;
        if (!root) return;
        if (d.error) {
            root.querySelector('#huntKv').innerHTML =
                '<span style="color:var(--accent-red);">' + d.error + '</span>';
            return;
        }
        root.querySelector('#huntTitleSym').textContent = d.symbol || '';
        const kv = [];
        const add = (k, v, cls) => kv.push(
            '<span><span class="k">' + k + '</span> <span class="v ' + (cls || '') + '">' + v + '</span></span>');

        add('Entry', Number(d.entry_price || 0).toLocaleString('en-US', {maximumFractionDigits: 2}));
        if (d.target_price) add('Target', Number(d.target_price).toLocaleString('en-US'));
        if (d.stop_trigger_price) add('Stop', Number(d.stop_trigger_price).toLocaleString('en-US'));
        add('Ticks', (d.ticks || []).length);

        // How long until price reached +5% above entry. Measured from the 15m
        // candles, because the tick log ends at exit and cannot see anything
        // after the bot sold - which is exactly the interesting part.
        if (d.t5) {
            const t5 = d.t5;
            if (t5.reached) {
                const mins = Math.round((t5.duration_s || 0) / 60);
                const dur = mins >= 60
                    ? Math.floor(mins / 60) + 'h ' + (mins % 60) + 'm'
                    : mins + 'm';
                const cls = t5.during_hold ? 'pos' : 'warn';
                const label = t5.during_hold ? 'during hold' : 'after exit';
                const extra = (!t5.during_hold && t5.after_exit_s != null)
                    ? ' (+' + Math.round(t5.after_exit_s / 60) + 'm late)'
                    : '';
                add('Time to +5%', dur + '  ' + label + extra, cls);
            } else {
                const peak = t5.last_high_pct != null ? 'peak ' + t5.last_high_pct.toFixed(2) + '%' : '';
                add('Time to +5%', 'never — ' + peak, 'neg');
            }
        }

        const pnl = d.net_pnl_pct;
        if (pnl != null) {
            const s = pnl >= 0 ? '+' : '';
            add('Result', s + pnl.toFixed(2) + '%', pnl >= 0 ? 'pos' : 'neg');
            const idr = d.net_pnl_idr;
            if (idr != null) add('Net PnL', (idr >= 0 ? '+' : '') + 'IDR ' +
                Number(idr).toLocaleString('en-US', {maximumFractionDigits: 0}),
                idr >= 0 ? 'pos' : 'neg');
        } else {
            add('Result', 'OPEN', '');
        }
        if (d.exit_reason) add('Exit', d.exit_reason);

        // peak / trough of the recorded path
        const t = d.ticks || [];
        if (t.length) {
            const pcts = t.map(x => x.pct).filter(v => v != null);
            if (pcts.length) {
                const hi = Math.max.apply(null, pcts);
                const lo = Math.min.apply(null, pcts);
                add('Peak', (hi >= 0 ? '+' : '') + hi.toFixed(2) + '%', hi >= 0 ? 'pos' : 'neg');
                add('Trough', (lo >= 0 ? '+' : '') + lo.toFixed(2) + '%', lo >= 0 ? 'pos' : 'neg');
            }
        }
        root.querySelector('#huntKv').innerHTML = kv.join('');
        root.querySelector('#huntChartLegend').style.display = 'block';
    }

    function drawHuntChart(hoverX) {
        const canvas = activeHuntDetailRow ? activeHuntDetailRow.querySelector('#huntChart') : null;
        if (!canvas) return;
        const wrap = canvas.parentElement;
        const W = wrap.clientWidth, H = wrap.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = W * dpr; canvas.height = H * dpr;
        canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        const padL = 46, padR = 12, padT = 14, padB = 26;
        const plotW = W - padL - padR, plotH = H - padT - padB;

        const data = huntChartData;
        if (!data || !data.ticks || !data.ticks.length) {
            ctx.fillStyle = 'var(--text-muted)';
            ctx.font = '13px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('no tick data for this hunt', W / 2, H / 2);
            return;
        }

        const ticks = data.ticks;
        const pcts = ticks.map(x => (x.pct == null ? 0 : x.pct));
        // include the target/stop levels in the scale so both are always visible
        const entry = data.entry_price || 0;
        let lo = Math.min.apply(null, pcts);
        let hi = Math.max.apply(null, pcts);
        let tgtPct = null, stopPct = null;
        if (entry > 0) {
            if (data.target_price) { tgtPct = (data.target_price - entry) / entry * 100; hi = Math.max(hi, tgtPct); }
            if (data.stop_trigger_price) { stopPct = (data.stop_trigger_price - entry) / entry * 100; lo = Math.min(lo, stopPct); }
        }
        lo = Math.min(lo, 0); hi = Math.max(hi, 0);
        const span = (hi - lo) || 1;
        const pad = span * 0.12;
        lo -= pad; hi += pad;

        const X = i => padL + (ticks.length === 1 ? plotW / 2 : (i / (ticks.length - 1)) * plotW);
        const Y = p => padT + (1 - (p - lo) / (hi - lo)) * plotH;

        // --- zero line (entry) ---
        ctx.strokeStyle = 'var(--border-strong)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();

        // --- target / stop guides ---
        const guide = (p, color, label) => {
            if (p == null) return;
            ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
            ctx.beginPath(); ctx.moveTo(padL, Y(p)); ctx.lineTo(W - padR, Y(p)); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = color; ctx.font = '10px monospace'; ctx.textAlign = 'left';
            ctx.fillText(label, padL + 4, Y(p) - 3);
        };
        guide(tgtPct, '#fbbf24', 'target +' + (tgtPct != null ? tgtPct.toFixed(1) : '') + '%');
        guide(stopPct, 'var(--text-secondary)', 'stop ' + (stopPct != null ? stopPct.toFixed(1) : '') + '%');

        // --- y axis labels ---
        ctx.fillStyle = 'var(--text-muted)'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
        for (let g = 0; g <= 4; g++) {
            const p = lo + (hi - lo) * (g / 4);
            ctx.fillText(p.toFixed(1) + '%', padL - 6, Y(p) + 3);
        }

        // --- filled area, coloured by profit/loss at each point ---
        for (let i = 0; i < ticks.length - 1; i++) {
            const p0 = pcts[i], p1 = pcts[i + 1];
            const x0 = X(i), x1 = X(i + 1);
            const y0 = Y(p0), y1 = Y(p1);
            const yz = Y(0);
            ctx.fillStyle = (p0 >= 0) ? 'rgba(34,197,94,0.16)' : 'rgba(239,68,68,0.16)';
            ctx.beginPath();
            ctx.moveTo(x0, yz); ctx.lineTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x1, yz);
            ctx.closePath(); ctx.fill();
        }

        // --- the PnL % line ---
        ctx.lineWidth = 2;
        for (let i = 0; i < ticks.length - 1; i++) {
            ctx.strokeStyle = (pcts[i] >= 0) ? '#22c55e' : '#ef4444';
            ctx.beginPath();
            ctx.moveTo(X(i), Y(pcts[i]));
            ctx.lineTo(X(i + 1), Y(pcts[i + 1]));
            ctx.stroke();
        }

        // --- entry marker + exit marker ---
        ctx.fillStyle = 'var(--text-primary)';
        ctx.beginPath(); ctx.arc(X(0), Y(pcts[0]), 3.5, 0, Math.PI * 2); ctx.fill();
        if (!data.open) {
            const li = ticks.length - 1;
            ctx.fillStyle = (pcts[li] >= 0) ? '#22c55e' : '#ef4444';
            ctx.beginPath(); ctx.arc(X(li), Y(pcts[li]), 4.5, 0, Math.PI * 2); ctx.fill();
        }

        // --- hover crosshair ---
        const tip = activeHuntDetailRow ? activeHuntDetailRow.querySelector('#huntTip') : null;
        if (typeof hoverX === 'number' && hoverX >= padL && hoverX <= W - padR && ticks.length > 1) {
            const frac = (hoverX - padL) / plotW;
            let idx = Math.round(frac * (ticks.length - 1));
            idx = Math.max(0, Math.min(ticks.length - 1, idx));
            const x = X(idx), y = Y(pcts[idx]);
            ctx.strokeStyle = 'var(--text-muted)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#fff';
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();

            const tk = ticks[idx];
            const tsec = tk.t || 0;
            const wib = new Date((tsec + 7 * 3600) * 1000).toISOString().slice(11, 19);
            const sign = pcts[idx] >= 0 ? '+' : '';
            tip.innerHTML =
                '<b>' + wib + ' WIB</b><br>' +
                'PnL: <b style="color:' + (pcts[idx] >= 0 ? '#22c55e' : '#ef4444') + '">' +
                sign + pcts[idx].toFixed(2) + '%</b><br>' +
                'last ' + Number(tk.last || 0).toLocaleString('en-US') +
                ' · bid ' + Number(tk.bid || 0).toLocaleString('en-US') +
                ' · ask ' + Number(tk.ask || 0).toLocaleString('en-US');
            tip.style.display = 'block';
            const tw = tip.offsetWidth || 140;
            let tx = x + 12;
            if (tx + tw > W) tx = x - tw - 12;
            tip.style.left = Math.max(2, tx) + 'px';
            tip.style.top = Math.max(2, y - 34) + 'px';
        } else if (tip) {
            tip.style.display = 'none';
        }
    }

    function initHuntChart(canvas) {
        if (!canvas || canvas.dataset.bound === '1') return;
        canvas.dataset.bound = '1';
        const pos = e => {
            const r = canvas.getBoundingClientRect();
            const cx = (e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX;
            return cx - r.left;
        };
        canvas.addEventListener('mousemove', e => drawHuntChart(pos(e)));
        canvas.addEventListener('mouseleave', () => drawHuntChart());
        canvas.addEventListener('touchstart', e => drawHuntChart(pos(e)), { passive: true });
        canvas.addEventListener('touchmove', e => drawHuntChart(pos(e)), { passive: true });
        canvas.addEventListener('touchend', () => drawHuntChart());
    }

    window.addEventListener('resize', () => {
        if (activeHuntDetailRow && activeHuntDetailRow.isConnected) drawHuntChart();
    });

    // ----------------------------------------------------------------------
    // Filter-board pair chart — click a Pair to expand its OHLCV chart inline
    // ----------------------------------------------------------------------
    function closeBoardChart() {
        if (boardChartRow) boardChartRow.remove();
        boardChartRow = null;
        boardChartTarget = null;
        boardChartData = null;
        boardChartTrigger = null;
    }

    function boardChartRange(tf) {
        const w = tfWindow(tf);
        return {from: w.from, to: w.to};
    }

    async function loadBoardChart(pair, tf, canvasId, infoId) {
        const info = document.getElementById(infoId);
        if (info) info.textContent = `Loading ${pair} ${tf}...`;
        const r = boardChartRange(tf);
        let url = `/api/ohlcv/${pair}?tf=${tf}&from_date=${r.from}&to_date=${r.to}`;
        try {
            const res = await fetch(url);
            const data = await res.json();
            if (!boardChartRow) return;           // closed while loading
            if (data.error) {
                if (info) info.textContent = data.error;
                boardChartData = null;
                drawChart(undefined, canvasId);
                return;
            }
            boardChartData = data;
            if (info) {
                const agg = data.aggregated ? ' (aggregated)' : '';
                const c = data.candles;
                info.textContent =
                    `${data.pair} ${data.tf}: ${c.length} candles${agg} — ` +
                    `${fmtTime(c[0].t, data.tf)} → ${fmtTime(c[c.length - 1].t, data.tf)}`;
            }
            drawChart(undefined, canvasId);
        } catch (e) {
            if (info) info.textContent = 'Error: ' + e;
        }
    }

    function openBoardChart(pair, trigger) {
        if (!pair || !trigger) return;
        // clicking the same pair closes it
        if (boardChartRow && boardChartTrigger === trigger) {
            closeBoardChart();
            return;
        }
        closeBoardChart();

        const parentRow = trigger.closest('tr');
        if (!parentRow) return;
        const table = parentRow.closest('table');
        const colCount = table ? table.querySelectorAll('thead th').length : 13;

        const tr = document.createElement('tr');
        tr.className = 'board-chart-row';
        const td = document.createElement('td');
        td.colSpan = colCount;

        const tpl = document.getElementById('boardChartTemplate');
        td.appendChild(tpl.content.cloneNode(true));
        tr.appendChild(td);
        parentRow.insertAdjacentElement('afterend', tr);

        boardChartRow = tr;
        boardChartTrigger = trigger;
        trigger.setAttribute('aria-expanded', 'true');

        // header
        tr.querySelector('.bc-pair').textContent = pair;
        tr.querySelector('#bcTf').value = '15m';
        const canvasId = 'boardChartCanvas';
        const infoId = 'boardChartInfo';

        // tf select -> reload
        tr.querySelector('#bcTf').addEventListener('change', (e) => {
            loadBoardChart(pair, e.target.value, canvasId, infoId);
        });

        // hover / touch inspection
        const canvas = tr.querySelector('#' + canvasId);
        const pos = ev => {
            const rect = canvas.getBoundingClientRect();
            const cx = (ev.touches && ev.touches[0]) ? ev.touches[0].clientX : ev.clientX;
            return cx - rect.left;
        };
        canvas.addEventListener('mousemove', ev => drawChart(pos(ev), canvasId));
        canvas.addEventListener('mouseleave', () => drawChart(undefined, canvasId));
        canvas.addEventListener('touchstart', ev => drawChart(pos(ev), canvasId), {passive: true});
        canvas.addEventListener('touchmove', ev => drawChart(pos(ev), canvasId), {passive: true});
        canvas.addEventListener('touchend', () => drawChart(undefined, canvasId), {passive: true});

        loadBoardChart(pair, '15m', canvasId, infoId);
    }

    // ----------------------------------------------------------------------
    // Trade log pagination (client-side, works for Live and Paper independently)
    // ----------------------------------------------------------------------
    const TRADES_PER_PAGE = __PAGE_SIZE__;

    function initPager(tableId, pagerKey) {
        const table = document.getElementById(tableId);
        const pager = document.querySelector(`.pagination-wrap[data-pager="${pagerKey}"]`);
        if (!table || !pager) return;

        const rows = Array.from(table.querySelectorAll('tbody tr.trade-row'));
        const total = rows.length;
        if (total === 0) return;

        const totalPages = Math.max(1, Math.ceil(total / TRADES_PER_PAGE));
        let page = 1;

        const pgFrom = pager.querySelector('.pg-from');
        const pgTo = pager.querySelector('.pg-to');
        const numsBox = pager.querySelector('.pg-nums');
        const prevBtn = pager.querySelector('button[data-page="prev"]');
        const nextBtn = pager.querySelector('button[data-page="next"]');

        function render() {
            const start = (page - 1) * TRADES_PER_PAGE;
            const end = Math.min(start + TRADES_PER_PAGE, total);

            rows.forEach((tr, i) => {
                tr.style.display = (i >= start && i < end) ? '' : 'none';
            });

            if (pgFrom) pgFrom.textContent = total === 0 ? 0 : start + 1;
            if (pgTo) pgTo.textContent = end;
            if (prevBtn) prevBtn.disabled = (page === 1);
            if (nextBtn) nextBtn.disabled = (page === totalPages);

            if (numsBox) {
                numsBox.innerHTML = '';
                for (let p = 1; p <= totalPages; p++) {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'pg-btn pg-num' + (p === page ? ' active' : '');
                    b.textContent = p;
                    b.addEventListener('click', () => { page = p; render(); });
                    numsBox.appendChild(b);
                }
            }
        }

        if (prevBtn) prevBtn.addEventListener('click', () => { if (page > 1) { page--; render(); } });
        if (nextBtn) nextBtn.addEventListener('click', () => { if (page < totalPages) { page++; render(); } });

        render();
    }

    function initPagers() {
        initPager('liveTradeTable', 'live');
        initPager('paperTradeTable', 'paper');
    }

    document.addEventListener('DOMContentLoaded', () => {
        initChart();
        initPagers();
    });
"""

# ---------------------------------------------------------------------------
# Page-level actions that must work on EVERY page, not just Analytics.
#
# These used to sit inside CHART_JS, which is emitted only on the Analytics
# page and - worse - inside initChart(). Anything defined there is invisible to
# the Home page, which is where the SELL NOW button and the theme switch live.
# Keeping them in their own constant makes the scoping explicit.
# ---------------------------------------------------------------------------
PAGE_JS = """
    // ---- Manual market exit ----
    // Sells the open position at the best bid and keeps hunting. NOT the same as
    // STOP: STOP liquidates and parks the engine. A confirmation is deliberate -
    // there is no undo on a market sell.
    function sellNow(symbol, btn) {
        var msg = 'Market sell ' + symbol + ' now?';
        msg += String.fromCharCode(10, 10);
        msg += 'This exits at the best bid, records the trade and sets a cooldown on the pair. ';
        msg += 'Hunting continues.';
        if (!confirm(msg)) return;
        if (btn) { btn.disabled = true; btn.textContent = 'SELLING...'; }
        fetch('/api/position/sell', { method: 'POST' })
            .then(function (r) { return r.json().then(function (d) {
                return { status: r.status, d: d }; }); })
            .then(function (res) {
                var status = res.status, d = res.d;
                if (status === 409 && d.busy) {
                    setTimeout(function () { sellNow(symbol, btn); }, 2500);
                    return;
                }
                if (!d.ok) {
                    window.alert('Sell failed: ' + (d.error || 'unknown'));
                    if (btn) { btn.disabled = false; btn.textContent = 'SELL NOW'; }
                    refreshDashboard();
                    return;
                }
                window.alert(d.message || 'sold');
                refreshDashboard();
            })
            .catch(function (e) {
                window.alert('Sell request failed: ' + e);
                if (btn) { btn.disabled = false; btn.textContent = 'SELL NOW'; }
            });
    }

    // wire the switch once the document is ready

"""


def _pair_options():
    """Build dropdown options with DEFAULT_PAIR selected."""
    pairs = list_ohlcv_pairs()
    if DEFAULT_PAIR not in pairs and pairs:
        pairs.append(DEFAULT_PAIR)
        pairs.sort()
    options = []
    for p in pairs:
        sel = " selected" if p == DEFAULT_PAIR else ""
        options.append(f'<option value="{p}"{sel}>{p}</option>')
    return "".join(options)


def _tf_options():
    options = []
    for tf in TIMEFRAMES:
        sel = " selected" if tf == DEFAULT_TF else ""
        options.append(f'<option value="{tf}"{sel}>{tf.upper()}</option>')
    return "".join(options)


def _first_touch(symbol: str, entry_ts: float, entry_price: float,
                 pct: float, exit_ts: float = None,
                 lookahead_s: float = 24 * 3600):
    """When did `symbol` first reach entry_price * (1 + pct) at or after entry?

    Read from the stored 15m candles, which continue past the trade, so this can
    answer questions the tick log cannot: the tick log ENDS at exit, so anything
    that happened after the bot sold has no tick row at all.

    Accuracy is one candle (15m): the high of a bar is the best resolution we
    store. The bar CONTAINING entry is excluded - its high may have printed
    before the entry - so a hit is never credited to a candle that started
    earlier than the position did.

    Returns a dict:
        reached        True if the price met the level at any point after entry
        at_ts          the candle timestamp that first met it
        duration_s     entry -> that candle
        during_hold    reached at or before exit (only if exit_ts is given)
        after_exit_s   seconds after exit (None if during hold or no exit_ts)
        last_high      highest high seen after entry, for context
    """
    path = OHLCV_DIR / f"{symbol.upper()}.csv"
    if not path.exists() or not entry_price:
        return {"reached": False, "reason": "no candles for this pair"}

    level = entry_price * (1.0 + pct)
    # A hit four days after the bot sold is not actionable, and would make the
    # "peak" figure meaningless. Bound the look-ahead: from entry while holding,
    # then a fixed window after exit.
    horizon_end = (float(exit_ts) + lookahead_s) if exit_ts else (entry_ts + lookahead_s)
    reached_ts = None
    last_high = 0.0
    try:
        with open(path, newline="") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if not parts or parts[0] == "timestamp":
                    continue
                try:
                    ts = int(parts[0])
                    high = float(parts[2])
                except (ValueError, IndexError):
                    continue
                if ts < entry_ts or ts > horizon_end:
                    continue
                if high > last_high:
                    last_high = high
                if reached_ts is None and high >= level:
                    reached_ts = ts
    except OSError:
        return {"reached": False, "reason": "cannot read candles"}

    out = {
        "reached": reached_ts is not None,
        "level": round(level, 8),
        "pct": pct,
        "at_ts": reached_ts,
        "last_high": last_high,
        "last_high_pct": (round((last_high - entry_price) / entry_price * 100, 4)
                          if last_high else None),
        "horizon_end": int(horizon_end),
        "lookahead_h": round(lookahead_s / 3600, 1),
    }
    if reached_ts is not None:
        out["duration_s"] = int(reached_ts - entry_ts)
        if exit_ts:
            out["during_hold"] = reached_ts <= exit_ts
            out["after_exit_s"] = (int(reached_ts - exit_ts)
                                   if reached_ts > exit_ts else None)
    return out


@app.get("/api/hunt/{hunt_id}")
async def api_hunt(hunt_id: str):
    """
    Return one hunt's recorded tick path plus its outcome.

    Path  : data/hunt_ticks/<HUNT_ID>.jsonl  (one row per 60s while HOLDING)
    Result: the matching trade_log entry from hunter_state.json, if closed.

    Used by the per-hunt PnL performance graph.
    """
    hid = hunt_id.strip().upper()
    if not re.fullmatch(r"H-\d{4,}", hid):
        return JSONResponse({"error": f"bad hunt id '{hunt_id}'"}, status_code=400)

    tick_path = HUNT_TICKS_DIR / f"{hid}.jsonl"
    ticks = []
    if tick_path.exists():
        try:
            with open(tick_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ticks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return JSONResponse({"error": f"cannot read ticks for {hid}"}, status_code=500)

    # find the trade record (closed) for this hunt
    state = {}
    state_file = DATA_DIR / "hunter_state.json"
    if state_file.exists():
        try:
            with open(state_file) as fh:
                state = json.load(fh)
        except Exception:
            state = {}
    trade = None
    for t in state.get("trade_log", []):
        if (t.get("hunt_id") or "").upper() == hid:
            trade = t
            break

    if not ticks and not trade:
        return JSONResponse({"error": f"no data for {hid}"}, status_code=404)

    symbol = (ticks[0].get("symbol") if ticks else trade.get("symbol")) or "—"
    entry_price = (ticks[0].get("entry_price") if ticks
                   else (trade or {}).get("entry_price"))
    coins = (trade or {}).get("quantity")
    target = ticks[0].get("target") if ticks else None
    stop = ticks[0].get("stop") if ticks else None
    _entry_ts = (ticks[0].get("t") if ticks
                 else (trade or {}).get("entry_time"))

    return JSONResponse({
        "hunt_id": hid,
        "symbol": symbol,
        "entry_price": entry_price,
        "target_price": target,
        "stop_trigger_price": stop,
        "entry_time": ticks[0].get("t") if ticks else (trade or {}).get("entry_time"),
        "exit_time": (trade or {}).get("exit_time"),
        "open": trade is None,
        "exit_reason": (trade or {}).get("exit_reason"),
        "net_pnl_idr": (trade or {}).get("net_pnl_idr"),
        "net_pnl_pct": (trade or {}).get("net_pnl_pct"),
        # every recorded tick: t, last, bid, ask, pct  (pct = gross % from entry)
        "ticks": [{k: r.get(k) for k in ("t", "last", "bid", "ask", "pct")}
                  for r in ticks],
        # How long until price reached +5% above entry, from the 15m candles (the
        # tick log STOPS at exit, so it cannot answer this once the bot has sold).
        # Reported whether or not the trade hit its own TP: the point is to expose
        # targets the bot abandoned early.
        "t5": _first_touch(
            symbol,
            float(_entry_ts) if _entry_ts else 0.0,
            float(entry_price or 0),
            0.05,
            (trade or {}).get("exit_time"),
        ),
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Root favicon so browsers that request /favicon.ico directly still get it."""
    from fastapi.responses import FileResponse
    p = STATIC_DIR / "favicon.ico"
    if p.exists():
        return FileResponse(str(p), media_type="image/x-icon")
    return JSONResponse({"error": "no favicon"}, status_code=404)


# ---------------------------------------------------------------------------
# Control plane — START / STOP hunting, and framework selection
# ---------------------------------------------------------------------------
# These are unauthenticated POSTs reachable over the LAN. Acceptable while the
# bot trades paper money on a single-user box; MUST be revisited before live
# money or public exposure. See the vault note "hunter control plane".
# ---------------------------------------------------------------------------

# The engine's schedule is the app's own crontab job "scanning", not a Hermes
# cron entry. The old hardcoded Hermes job id did NOTHING on an installed app:
# pause/resume silently failed, so STOP only worked through the control file and
# the scanning job kept firing every minute for nothing.
SCAN_JOB = "scanning"
STOP_SCRIPT = BASE_DIR / "scripts" / "run_hunter.py"


def _resolve_engine_python() -> str:
    """
    Find an interpreter that can actually run run_hunter.py.

    The dashboard service runs under .venv (fastapi/uvicorn only); the engine
    needs `requests`, which lives in the system python the engine job uses via
    hunter.sh. Using sys.executable here fails with ModuleNotFoundError, so
    probe candidates once and cache the first that can import requests.
    Override with HUNTER_PYTHON if the layout ever changes.
    """
    import paths
    candidates = []
    env_py = os.environ.get("HUNTER_PYTHON")
    if env_py:
        candidates.append(env_py)

    # The installed layout puts venv/ beside app/ - and that venv has `requests`.
    # Probing it FIRST is what makes 'Fetch the top-20 pairs' work on a fresh
    # install, where the dashboard runs under a different interpreter that
    # cannot import requests and fetch_market.py dies in 0s.
    install_venv = os.path.join(os.path.dirname(str(paths.APP_ROOT)), "venv", "bin", "python")
    if os.path.exists(install_venv):
        candidates.append(install_venv)

    candidates += ["python3", "python", "/usr/bin/python3", sys.executable]

    for cand in candidates:
        if not cand:
            continue
        try:
            probe = subprocess.run([cand, "-c", "import requests"],
                                   capture_output=True, text=True, timeout=15)
            if probe.returncode == 0:
                return cand
        except (OSError, subprocess.TimeoutExpired):
            continue
    return sys.executable      # last resort; the error will surface to the UI


ENGINE_PYTHON = _resolve_engine_python()


def _run_cron(action: str) -> tuple:
    """Best-effort pause/resume of the engine's own scheduled job.

    The engine runs from the app's crontab job "scanning" (crontab_manager), not
    from a Hermes cron entry. An earlier version called
    `hermes cron pause 75a5da3dafb6` - a hardcoded id that only ever existed on
    the developer's machine. On a real install that call failed and the job kept
    firing every minute; the control file still stopped trading, so the app
    looked correct while spawning pointless work forever.

    The control file remains authoritative: if this fails, the engine still
    stops or starts correctly. Returns (ok, detail) - never raises.
    """
    try:
        import crontab_manager as cm
    except Exception as e:                                  # noqa: BLE001
        return False, f"crontab_manager unavailable: {e}"

    enable = (action == "resume")
    try:
        res = cm.set_job(SCAN_JOB, enable)
    except Exception as e:                                  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    # set_job returns the full job->enabled map on success and raises on failure,
    # so the only thing worth reporting is whether THIS job ended up as asked.
    if isinstance(res, dict) and SCAN_JOB in res:
        if bool(res[SCAN_JOB]) == enable:
            return True, ("enabled" if enable else "disabled")
        return False, f"{SCAN_JOB} did not change state"
    return True, ("enabled" if enable else "disabled")


@app.get("/api/control")
async def api_control_get():
    """Current control mode, active framework, and when it last changed."""
    from control import load_control
    from frameworks import list_frameworks
    c = load_control()
    st = _dash_state()
    return {
        "mode": c["mode"],
        # the executor is part of the control state, not a UI preference - the
        # buttons paint from this and nothing else
        "exec": c.get("exec", "PAPER"),
        "framework": c.get("framework", "A"),
        "changed_at": c.get("changed_at"),
        "changed_by": c.get("changed_by"),
        "phase": st.get("phase"),
        "position": _position_summary(),
        "blocked": c.get("blocked"),
        # what the EXCHANGE holds, not just what the bot thinks. A coin in the
        # account that the bot does not know about has no stop, and reporting
        # "HUNTING / no position" while holding real coins is how the operator
        # ends up with an unprotected position.
        "exchange": _exchange_summary(st),
        "frameworks": list_frameworks(),
    }


def _dash_state():
    """The engine state, or {} if it cannot be read. Never raises."""
    try:
        import state as _state
        return _state.load_state() or {}
    except Exception:                                        # noqa: BLE001
        return {}


def _exec_is_live() -> bool:
    """True if the control file says LIVE. Never raises; failure means paper."""
    try:
        from control import load_control, is_live
        return is_live(load_control())
    except Exception:                                        # noqa: BLE001
        return False


def _open_position() -> bool:
    """True if a position is currently held. Blocks arming or switching LIVE."""
    st = _dash_state()
    return st.get("phase") == "HOLDING" and bool(st.get("position"))


def _open_order() -> bool:
    """True if a resting order is working. Also blocks a live switch: the order
    was placed by one executor and would be filled against another's accounting."""
    st = _dash_state()
    return st.get("phase") == "ORDER_OPEN" and bool(st.get("order"))


def _exchange_summary(state: dict) -> dict:
    """Real holdings, or an error. Never raises; never blocks the page."""
    try:
        from control import is_live, load_control
        if not is_live(load_control()):
            return {"available": False, "reason": "paper mode"}
    except Exception:                                        # noqa: BLE001
        return {"available": False, "reason": "control unreadable"}
    try:
        import executor_live
        rec = executor_live.reconcile_position(state)
        return {"available": True, **rec}
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "reason": str(e)[:200]}


def _position_summary() -> dict:
    """The open position as the confirmation dialog needs to show it."""
    st = _dash_state()
    p = st.get("position") or {}
    if not p:
        return {}
    try:
        px = float(p.get("entry_price") or 0)
        qty = float(p.get("coins") or 0)
        return {"symbol": p.get("symbol"), "coins": qty,
                "entry_price": px, "value_idr": round(px * qty, 2),
                "hunt_id": p.get("hunt_id")}
    except Exception:                                        # noqa: BLE001
        return {"symbol": p.get("symbol")}


def _trade_size() -> float:
    """The IDR amount one trade would use right now, matching the engine.

    Read from the same config key and sizing fraction the engine uses, so the
    number quoted in the START LIVE confirmation is the number that will be
    spent - not an estimate computed differently in the UI.
    """
    try:
        import config as cfgmod
        cfg = cfgmod.load_config() or {}
        frac = float(cfg.get("trading", {}).get("sizing_fraction", 0.40))
    except Exception:                                        # noqa: BLE001
        frac = 0.40
    try:
        st = _dash_state()
        bal = float(st.get("balance_idr") or 0)
    except Exception:                                        # noqa: BLE001
        bal = 0.0
    if not bal:
        try:
            import executor_live
            bal = executor_live.read_balance_idr()
        except Exception:                                    # noqa: BLE001
            bal = 0.0
    return round(bal * frac, 2)


@app.get("/api/live/balance")
async def api_live_balance():
    """The REAL exchange balance.

    Read server-side because the browser must never hold the API secret. Cached
    for a few seconds so a page that polls does not hammer the exchange.

    Returns ok=False with the exchange's own message on failure - a balance the
    app cannot read must be shown as unknown, never as zero.
    """
    global _LIVE_BAL_CACHE
    now = time.time()
    try:
        if _LIVE_BAL_CACHE["at"] and (now - _LIVE_BAL_CACHE["at"]) < 10:
            return _LIVE_BAL_CACHE["data"]
    except Exception:
        pass

    try:
        import executor_live
        client = executor_live.RealClient()
        info = client.get_info()
        ret = info.get("return") or {}
        bal = ret.get("balance") or {}
        hold = ret.get("balance_hold") or {}
        idr = float(bal.get("idr") or 0)
        held = float(hold.get("idr") or 0)
        data = {
            "ok": True,
            "idr": idr,
            "idr_on_hold": held,
            "spendable": idr,
            "assets": {k.upper(): float(v) for k, v in bal.items()
                       if k.lower() != "idr" and float(v or 0) > 0},
            "fetched_at": now,
        }
    except Exception as e:                                   # noqa: BLE001
        data = {"ok": False, "idr": None, "error": str(e)[:300], "fetched_at": now}

    _LIVE_BAL_CACHE = {"at": now, "data": data}
    return data


@app.get("/api/live/preflight")
async def api_live_preflight():
    """Everything that must be true before START LIVE is allowed.

    Read-only: it authenticates and reads balances, and places no order.
    """
    try:
        import executor_live
        import config as cfgmod
        size = 0.0
        try:
            size = float((cfgmod.load_config() or {}).get("trade_size_idr") or 0)
        except Exception:
            pass
        if not size:
            try:
                bal = executor_live.read_balance_idr()
                size = bal * 0.40
            except Exception:
                size = 0.0
        pf = executor_live.preflight(trade_size_idr=size)
        return {"ok": pf["ok"], "checks": pf["checks"],
                "problems": pf["problems"], "warnings": pf.get("warnings", []),
                "trade_size_idr": size}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "checks": [], "trade_size_idr": 0.0,
                "problems": [f"{type(e).__name__}: {str(e)[:300]}"], "warnings": []}


@app.post("/api/control/start-live")
async def api_control_start_live(request: Request):
    """Arm LIVE and start hunting.

    Deliberately a SEPARATE endpoint from /api/control/start. A single endpoint
    with an {exec: "live"} body can be reached by a stale flag or a bug; this
    one cannot. It also requires the literal word LIVE in the body, so no
    accidental click or replayed request can arm it.
    """
    from control import MODE_RUNNING, EXEC_LIVE, set_mode, load_control
    import executor_live

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    if str(body.get("confirm", "")).strip().upper() != "LIVE":
        return JSONResponse({"ok": False,
                             "error": "type LIVE to confirm"}, status_code=400)

    ctrl = load_control()
    if ctrl.get("mode") == "RUNNING":
        return JSONResponse({"ok": False,
                             "error": "already running - stop first"},
                            status_code=409)

    # never arm LIVE while a position or resting order exists: the position
    # would have been opened by a different executor than the one that closes it
    if _open_position() or _open_order():
        return JSONResponse({"ok": False,
                             "error": "a position or order is open - close it before switching to LIVE"},
                            status_code=409)

    try:
        pf = executor_live.preflight(trade_size_idr=_trade_size())
    except Exception as e:                                   # noqa: BLE001
        return JSONResponse({"ok": False,
                             "error": f"preflight failed: {str(e)[:300]}"},
                            status_code=400)
    if not pf["ok"]:
        return JSONResponse({"ok": False, "error": "; ".join(pf["problems"])[:500],
                             "problems": pf["problems"]}, status_code=400)

    try:
        set_mode(MODE_RUNNING, by="dashboard", exec_mode=EXEC_LIVE)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    cron_ok, cron_detail = _run_cron("resume")
    warn = ("; ".join(pf.get("warnings") or []))
    return {"ok": True,
            "message": ("LIVE trading is armed. Real orders will be placed."
                        + (f"\n\nNote: {warn}" if warn else "")),
            "exec": "LIVE", "cron": cron_detail}


@app.post("/api/control/start")
async def api_control_start():
    """Resume hunting: re-enable the engine job and set mode RUNNING."""
    from control import MODE_RUNNING, set_mode
    set_mode(MODE_RUNNING, by="dashboard")
    cron_ok, cron_detail = _run_cron("resume")
    return {"ok": True, "mode": MODE_RUNNING,
            "cron_ok": cron_ok, "cron_detail": cron_detail}


@app.post("/api/position/buy")
async def api_position_buy(request: Request):
    """Market-buy a pair NOW, as a manual operator action.

    Only available in LIVE, and only while flat. It spends real money, so it
    requires the symbol in the body AND a typed confirmation phrase, and the
    engine re-checks every cap itself - this endpoint is not the safety layer.

    A market order is used rather than a limit because the button's whole point
    is "buy it now". A limit can rest unfilled for minutes, which is the
    opposite of an immediate buy.
    """
    try:
        from control import load_control, is_live
        ctrl = load_control()
    except Exception as e:                                   # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"cannot read control: {e}"},
                            status_code=500)

    if not is_live(ctrl):
        return JSONResponse(
            {"ok": False,
             "error": "market buy is LIVE only - start LIVE first, or use the "
                      "paper engine's own entries"},
            status_code=400)

    try:
        body = await request.json()
    except Exception:
        body = {}

    symbol = str(body.get("symbol") or "").strip().upper()
    if not symbol:
        return JSONResponse({"ok": False, "error": "symbol is required"},
                            status_code=400)
    if str(body.get("confirm", "")).strip().upper() != symbol:
        return JSONResponse(
            {"ok": False,
             "error": f"type {symbol} to confirm"},
            status_code=400)

    if _open_position() or _open_order():
        return JSONResponse(
            {"ok": False,
             "error": "a position or order is already open - close it first"},
            status_code=409)

    try:
        proc = subprocess.run(
            [ENGINE_PYTHON, str(STOP_SCRIPT), "--buy-now", symbol],
            capture_output=True, text=True, timeout=90,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "buy timed out"},
                            status_code=504)
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                            status_code=500)

    if proc.returncode == 2:
        return JSONResponse({"ok": False, "busy": True,
                             "error": "engine tick in progress - retry"},
                            status_code=409)

    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(
            {"ok": False,
             "error": (proc.stderr or proc.stdout or "no output").strip()[:400]},
            status_code=500)

    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    return result


@app.post("/api/position/sell")
async def api_position_sell():
    """Market-sell the open position and keep hunting.

    Different from /api/control/stop, which liquidates AND parks the engine.
    This is the "get me out of this one trade" action: the engine sells at the
    best bid, records the trade, sets a cooldown on that pair and stays RUNNING.

    Runs scripts/run_hunter.py --sell-now as a subprocess so the sell takes the
    engine's own flock and reuses its fee/PnL arithmetic. Re-implementing the
    sell here would let the dashboard and the engine disagree about the same
    fill. Exit code 2 means a tick is in flight; the UI should retry.
    """
    try:
        proc = subprocess.run(
            [ENGINE_PYTHON, str(STOP_SCRIPT), "--sell-now"],
            capture_output=True, text=True, timeout=60,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "sell timed out"}, status_code=504)
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                            status_code=500)

    if proc.returncode == 2:
        return JSONResponse(
            {"ok": False, "busy": True,
             "error": "engine tick in progress — retry in a moment"},
            status_code=409)
    if proc.returncode == 3:
        # nothing to sell: not an error, the position closed on its own
        return JSONResponse({"ok": False, "nothing_to_sell": True,
                             "error": "no open position"}, status_code=409)

    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(
            {"ok": False, "error": (proc.stderr or proc.stdout or "no output").strip()[:400]},
            status_code=500)
    return result


@app.post("/api/control/stop")
async def api_control_stop(request: Request):
    """
    Stop hunting: liquidate anything open, park the engine, pause the engine job.

    Runs scripts/run_hunter.py --stop-now as a subprocess so the stop takes
    the engine's own flock and reuses its money-math. Exit code 2 means a tick
    is in flight; the UI should retry.

    LIVE stops are different in two ways, both requested explicitly:

      * resting orders are CANCELLED first. An order left on the book after the
        process stops is a position you do not know you own.
      * an open position is not closed silently. The caller must say what to do
        with it: sell it, or leave it open on the exchange. Guessing either way
        is wrong - one force-sells at market, the other strands real money.
    """
    # which executor is running decides what "stop" means
    try:
        from control import load_control, is_live, MODE_STOPPED, set_mode
        ctrl = load_control()
        live = is_live(ctrl)
    except Exception as e:                                   # noqa: BLE001
        return JSONResponse({"ok": False,
                             "error": f"cannot read control: {e}"},
                            status_code=500)

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    if live and _open_position():
        # "C": ask. The UI sends either "sell" or "leave" after the operator
        # chooses; without one of those, refuse rather than pick for them.
        choice = str(body.get("position") or "").strip().lower()
        if choice not in ("sell", "leave"):
            return JSONResponse(
                {"ok": False, "needs_choice": True,
                 "error": "a live position is open - choose 'sell' or 'leave'",
                 "position": _position_summary()},
                status_code=409)
        if choice == "leave":
            import control as _ctl
            _ctl.set_mode(MODE_STOPPED, by="dashboard")
            cron_ok, cron_detail = _run_cron("pause")
            try:
                import executor_live
                exs = executor_live.cancel_all_open_orders()
                note = (f"cancelled {len(exs['cancelled'])} order(s)"
                        if exs["ok"] else
                        f"WARNING: could not cancel {len(exs['failed'])} order(s)")
            except Exception as e:                           # noqa: BLE001
                note = f"WARNING: could not cancel open orders: {str(e)[:120]}"
            return {"ok": True, "exec": "LIVE",
                    "message": (f"Stopped. THE POSITION IS STILL OPEN on the "
                                f"exchange - you must sell it yourself. {note}."),
                    "cron": cron_detail}

    cancel_note = ""
    if live:
        # an orphaned resting order is a position you don't know you own
        try:
            import executor_live
            exs = executor_live.cancel_all_open_orders()
            cancel_note = (f"cancelled {len(exs['cancelled'])} order(s)"
                           if exs["ok"] else
                           f"WARNING: {len(exs['failed'])} order(s) could not be cancelled")
        except Exception as e:                               # noqa: BLE001
            cancel_note = f"WARNING: cancel failed: {str(e)[:120]}"

    try:
        proc = subprocess.run(
            [ENGINE_PYTHON, str(STOP_SCRIPT), "--stop-now"],
            capture_output=True, text=True, timeout=60,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "stop timed out"}, status_code=504)
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                            status_code=500)

    if proc.returncode == 2:
        return JSONResponse(
            {"ok": False, "busy": True,
             "error": "engine tick in progress — retry in a moment"},
            status_code=409)

    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(
            {"ok": False, "error": (proc.stderr or proc.stdout or "no output").strip()[:400]},
            status_code=500)

    if result.get("ok"):
        cron_ok, cron_detail = _run_cron("pause")
        result["cron_ok"] = cron_ok
        result["cron_detail"] = cron_detail
        if cancel_note:
            result["message"] = ((result.get("message") or "")
                                 + f" {cancel_note}.")
    return result


@app.get("/api/frameworks")
async def api_frameworks_list():
    """All registered frameworks plus which one is active."""
    from control import load_control
    from frameworks import list_frameworks
    return {"active": load_control().get("framework", "A"),
            "frameworks": list_frameworks()}


@app.post("/api/cron/{job}/{action}")
async def api_cron_toggle(job: str, action: str):
    """Enable or disable a single managed cron job from the Settings page.

    action must be 'enable' or 'disable'. The job name must be one of the
    managed jobs known to crontab_manager.
    """
    import crontab_manager as cm
    if action not in ("enable", "disable"):
        return JSONResponse({"ok": False, "error": "invalid action"}, status_code=400)
    if not any(j[0] == job for j in cm.JOBS):
        return JSONResponse({"ok": False, "error": f"unknown job: {job}"}, status_code=400)
    try:
        cm.set_job(job, action == "enable")
        return {"ok": True, "job": job, "enabled": action == "enable"}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/api/cron")
async def api_cron_list():
    """Return the current on/off state of every managed cron job."""
    import crontab_manager as cm
    try:
        states = cm.job_states() or {}
        return {"jobs": {name: bool(states.get(name)) for name, _, _ in cm.JOBS}}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/frameworks/{name}/activate")
async def api_framework_activate(name: str):
    """
    Switch the active framework.

    Reza's rule (2026-11-03): framework may only change while STOPPED.
    Hunting, or holding a position, or a resting order -> 409. The user must
    press STOP first, which liquidates/cancels, then switch.
    """
    from control import load_control, save_control, MODE_STOPPED
    from frameworks import framework_exists

    key = (name or "").upper()
    if not framework_exists(key):
        return JSONResponse({"ok": False, "error": f"unknown framework '{name}'"},
                            status_code=404)

    c = load_control()
    if c["mode"] != MODE_STOPPED:
        return JSONResponse({
            "ok": False,
            "error": "stop hunting before switching framework",
            "mode": c["mode"],
        }, status_code=409)

    c["framework"] = key
    c["changed_by"] = "dashboard"
    save_control(c)
    return {"ok": True, "framework": key, "mode": c["mode"]}


@app.get("/api/setup")
async def api_setup_get():
    """Readiness plus every action's state, for the Settings page.

    Lives on disk (setup_state.json), so it survives a dashboard restart, a
    browser reload and an ssh disconnect.
    """
    import setup_state
    r = setup_state.readiness()
    r["order"] = setup_state.ORDER
    r["labels"] = {n: setup_state.ACTIONS[n] for n in setup_state.ORDER}
    return r


@app.post("/api/setup/{action}/run")
async def api_setup_run(action: str):
    """Start one setup action, detached.

    Returns immediately: a deep backfill runs for a long time and must not hold
    the request open. A second heavy job is refused with 409 rather than queued,
    because two backfills writing the same CSVs corrupt data.
    """
    import setup_state
    ok, message = setup_state.start(action, background=True)
    if not ok:
        status = 409 if "still running" in message else 400
        return JSONResponse({"ok": False, "error": message}, status_code=status)
    return {"ok": True, "action": action, "message": message}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # Load Hunter state
    state_file = DATA_DIR / "hunter_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file) as fh:
                state = json.load(fh)
        except Exception:
            pass

    ticks = get_tick_map()
    pairs_html = get_pairs_info(ticks)
    summary, stale_html = get_ohlcv_summary_and_stale(ticks)

    total = summary.get("total_pairs", 0)
    complete = summary.get("complete_count", 0)
    stale = summary.get("stale_count", 0)

    depth_size = get_folder_size(DATA_DIR / "depth")
    ohlcv_size = get_folder_size(OHLCV_DIR)

    today = time.strftime("%Y-%m-%d", time.gmtime(time.time() + WIB_OFFSET))
    now_wib = time.strftime("%H:%M:%S WIB", time.gmtime(time.time() + WIB_OFFSET))

    # Filter board
    board_rows, winner, eligible_n = build_filter_board()

    # Control plane (start/stop + active framework) for the UI
    try:
        from control import load_control
        control = load_control()
    except Exception:
        control = {"mode": "RUNNING", "framework": "A"}
    # A verdict is only meaningful if the engine would act on it. While the
    # engine refuses to tick, "WINNER" is a preview of a decision nobody makes -
    # and it reads as a promise. The word is withheld and the reason shown
    # instead, so the row can never be mistaken for "the bot is buying this".
    _funnel_blocked = (control or {}).get("blocked")

    board_html_rows = ""
    for i, r in enumerate(board_rows, 1):
        reason = _block_reason(r)
        if _funnel_blocked and (r["pair"] == winner or r["eligible"]):
            verdict = '<span class="badge-no" title="the engine is not running">✗ BLOCKED</span>'
            tr_cls = ""
        elif r["pair"] == winner:
            verdict = '<span class="badge-win">WINNER</span>'
            tr_cls = ' class="winner"'
        elif r["eligible"]:
            verdict = '<span class="badge-ok">TRADEABLE</span>'
            tr_cls = ""
        else:
            verdict = f'<span class="badge-no">✗ {reason}</span>'
            tr_cls = ""
        total_disp = f"{r['total']:.0f}" if r["total"] is not None else "—"
        total_cls = "pass" if (r["total"] is not None and r["total"] >= 60) else "fail"
        board_html_rows += (
            f"<tr{tr_cls}><td class='num'>{i}</td>"
            f"<td><b><span class='pair-link' onclick=\"openBoardChart('{r['pair']}', this)\">{r['pair']}</span></b></td>"
            f"<td class='num'>{r['vol'] / 1e9:.2f}B</td>"
            + _cell(*r["candles"]) + _cell(*r["fresh"]) + _cell(*r["spread"])
            + _cell(*r["depth"]) + _cell(*r["cooldown"])
            + _cell(*r["trend"])
            + f"<td>{r['momentum_disp']}</td>"
            + f"<td class='num {total_cls}'><b>{total_disp}</b></td>"
            f"<td>{verdict}</td>"
            # Per-pair manual buy. Hidden by default and revealed only when the
            # executor is LIVE and nothing is open - the same rule the winner
            # card's button follows, applied to every row so the operator can
            # buy a pair the funnel did NOT pick.
            f"<td class='act'><button class='row-buy' data-sym='{r['pair']}' "
            f"type='button' onclick=\"buyNow('{r['pair']}')\">BUY</button></td></tr>"
        )
    winner_text = f" · winner: <b>{winner}</b>" if winner else " · no eligible pair right now"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, viewport-fit=cover">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#0c0c0c">
    <title>The Hunter Dashboard</title>
    <link rel="icon" href="/static/favicon.ico" sizes="any">
    <link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">
    <link rel="icon" type="image/png" sizes="192x192" href="/static/favicon-192.png">
    <link rel="apple-touch-icon" href="/static/favicon-180.png">
    
    <style>{_style()}</style>
</head>
<body>
    <!-- Top Navigation Bar with Burger Button -->
    <div class="top-bar">
        <div class="top-left">
            <button class="burger-btn" onclick="toggleNav()" type="button" title="Open navigation menu" aria-label="Menu">
                <span></span><span></span><span></span>
            </button>
            <div class="brand">
                <span class="cli-prompt">[~]$</span>
                <span class="brand-name">THE HUNTER</span>
                <span class="active-crumb" id="activeCrumb">/ home</span>
            </div>
        </div>
        <div class="top-right">
            <span class="conn-dot">●</span>
            <span class="port-label">PORT 8082</span>
            <span class="top-clock">{now_wib}</span>
            <button class="refresh-btn" id="refreshBtn" type="button"
                    onclick="refreshDashboard()" title="Reload data (no F5 needed)"
                    aria-label="Refresh">⟳</button>
        </div>
    </div>

    <!-- Toast for refresh feedback -->
    <div class="toast" id="toast"></div>

    <!-- Navigation Drawer -->
    <div class="nav-drawer" id="navDrawer">
        <div class="drawer-top">
            <span class="drawer-title">SYSTEM NAVIGATION</span>
            <button class="drawer-close" onclick="toggleNav()" type="button" aria-label="Close menu">[✕]</button>
        </div>
        <div class="drawer-links">
            <button class="nav-item active" id="navHome" type="button" onclick="switchPage('home')">
                <span class="nav-icon">▶</span>
                <div class="nav-meta">
                    <div class="nav-title">HOME</div>
                    <div class="nav-desc">Trade logs, live/paper mode, target acquisition</div>
                </div>
            </button>
            <button class="nav-item" id="navOthers" type="button" onclick="switchPage('others')">
                <span class="nav-icon">▶</span>
                <div class="nav-meta">
                    <div class="nav-title">ANALYTICS</div>
                    <div class="nav-desc">Pipeline health, charts, stale data, top 20</div>
                </div>
            </button>
            <button class="nav-item" id="navSettings" type="button" onclick="switchPage('settings')">
                <span class="nav-icon">▶</span>
                <div class="nav-meta">
                    <div class="nav-title">SETTINGS</div>
                    <div class="nav-desc">First-time setup, cron jobs, framework config</div>
                </div>
            </button>
        </div>
        <div class="drawer-bottom">
            <div><b>The Hunter Trading Engine</b></div>
            <div>Hunting engine · Paper Mode</div>
            <div>ZeroTier: 10.36.20.175:8082</div>
        </div>
    </div>
    <div class="nav-overlay" id="navOverlay" onclick="toggleNav()"></div>

    <!-- PAGE A: HOME VIEW -->
    <div id="pageHome">
        {_trading_log_section(state)}

        {_winner_card(winner, board_rows)}

        <div class="section board-collapsed" id="boardSection">
            <h2>Filter Board — Prey Selection (live preview)</h2>
            <p class="board-note">Live preview of The Hunter's funnel · trade size computed from the CURRENT balance and sizing · depth shown = min(ask, bid) cover within ±2% · signal scoring is a preview · cooldown reads hunter_state.json · "!OB" = overbought penalty applied</p>
            <p class="board-summary"><b>{eligible_n} of 20</b> pairs fully eligible{winner_text}</p>
            <div class="scroll-hint">⟵ swipe horizontally to view all columns ⟶</div>
            <div class="board-wrap">
                <table class="board">
                    <thead><tr>
                        <th>#</th><th>Pair</th><th>Vol</th><th>Candles</th><th>Fresh</th>
                        <th>Spread</th><th>Depth</th><th>Cooldown</th><th>Trend</th>
                        <th>Mom</th><th>Total</th><th>Verdict</th><th>Action</th>
                    </tr></thead>
                    <tbody>{board_html_rows}</tbody>
                </table>
                <p class="action-note" id="actionNote">
                    BUY buttons appear only in LIVE with no position open.
                    The Action column is present either way so the layout does
                    not shift when you arm live.
                </p>
            </div>
        </div>

    </div>

    <!-- PAGE C: SETTINGS VIEW -->
    <div id="pageSettings" style="display:none;">
        {_settings_panel()}
        {_framework_panel(state, control)}
    </div>

    <!-- PAGE B: OTHERS VIEW -->
    <div id="pageOthers" style="display:none;">
        <div class="section">
            <h2>OHLCV Data Summary</h2>
            <div class="cards">
                <div class="card total"><span class="num">{total}</span><span class="label">Total Pairs</span></div>
                <div class="card complete"><span class="num">{complete}</span><span class="label">Complete</span></div>
                <div class="card stale"><span class="num">{stale}</span><span class="label">Stale</span></div>
            </div>
            <div class="cards">
                <div class="card size"><span class="num">{depth_size}</span><span class="label">Depth Folder</span></div>
                <div class="card size"><span class="num">{ohlcv_size}</span><span class="label">OHLCV Folder</span></div>
            </div>
        </div>

        <div class="section">
            <h2>Candlestick Chart</h2>
            <div class="chart-controls">
                <div class="ctrl-group">
                    <label for="pairSelect">Pair</label>
                    <select id="pairSelect">{_pair_options()}</select>
                </div>
                <div class="ctrl-group">
                    <label for="tfSelect">Timeframe</label>
                    <select id="tfSelect">{_tf_options()}</select>
                </div>
                <div class="ctrl-group">
                    <label for="dateFrom">From</label>
                    <input type="date" id="dateFrom" value="{DEFAULT_FROM}">
                </div>
                <div class="ctrl-group">
                    <label for="dateTo">To</label>
                    <input type="date" id="dateTo" value="{today}">
                </div>
                <button class="reset-btn" id="resetBtn" title="Reset pair, timeframe, and dates to defaults">Reset</button>
            </div>
            <div class="chart-info" id="chartInfo">Loading...</div>
            <div class="chart-tip">Tip: Touch & drag across chart to inspect OHLCV candles. Timeframe auto-sets range (15m→7d, 1h→30d, 1D→full).</div>
            <div class="chart-wrap">
                <canvas id="chart"></canvas>
                <div id="tooltip"></div>
            </div>
        </div>

        <div class="section">
            <h2>Stale Data Identification</h2>
            <div class="scroll-hint">⟵ swipe horizontally to view all columns ⟶</div>
            <div class="stale-scroll">{stale_html}</div>
        </div>

        <div class="section">
            <h2>Top 20 Pairs (24h Volume)</h2>
            <div class="scroll-hint">⟵ swipe horizontally to view all columns ⟶</div>
            <div class="pairs-scroll">{pairs_html}</div>
        </div>
    </div>

        <!-- Click a Pair in the filter board to expand its OHLCV chart inline -->
        <template id="boardChartTemplate">
            <div class="bc-wrap">
                <div class="bc-head">
                    <span class="bc-title"><span class="bc-pair">—</span> · OHLCV</span>
                    <select id="bcTf" class="bc-tf" aria-label="Timeframe">
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="3h">3h</option>
                        <option value="6h">6h</option>
                        <option value="12h">12h</option>
                        <option value="1d">1D</option>
                    </select>
                    <button class="bc-close" type="button" onclick="closeBoardChart()" title="Close chart" aria-label="Close chart">×</button>
                </div>
                <div class="bc-info" id="boardChartInfo">Loading…</div>
                <div class="bc-chart-wrap">
                    <canvas id="boardChartCanvas"></canvas>
                    <div id="boardTooltip" class="chart-tooltip"></div>
                </div>
            </div>
        </template>

    <!-- Cloned inline directly beneath its history row when a Hunt ID is clicked -->
    <template id="huntDetailTemplate">
        <div class="section hunt-detail">
            <div class="hunt-head">
                <div class="hunt-title">
                    <span class="hunt-id-big" id="huntTitleId">H-</span>
                    <span class="hunt-sym" id="huntTitleSym"></span>
                </div>
                <button class="hunt-close-btn" type="button" onclick="closeHuntPage()" title="Close graph" aria-label="Close graph">×</button>
            </div>
            <div class="hunt-kv" id="huntKv"><span style="color:#64748b;">loading…</span></div>
            <div class="chart-tip" id="huntChartLegend" style="display:none;">
                PnL % from entry over the life of the hunt · <b style="color:#22c55e;">green</b> = profit,
                <b style="color:#ef4444;">red</b> = loss ·
                <b style="color:#fbbf24;">——</b> target (limit sell),
                <b style="color:#9ca3af;">——</b> stop (market sell) · drag across the chart to inspect each tick
            </div>
            <div class="hunt-chart-wrap">
                <canvas id="huntChart"></canvas>
                <div id="huntTip" class="chart-tooltip" style="display:none;"></div>
            </div>
        </div>
    </template>

    <div class="footer">The Hunter Dashboard · built {time.strftime("%Y-%m-%d %H:%M:%S WIB", time.gmtime(time.time() + WIB_OFFSET))}</div>

    <script>{CHART_JS.replace("__PAGE_SIZE__", str(TRADES_PER_PAGE))}</script>
    <script>{PAGE_JS}</script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
