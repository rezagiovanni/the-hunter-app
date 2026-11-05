#!/usr/bin/env python3
"""
features.py — the feature maths, in pure standard library.

Why this exists
---------------
`model/predict.py` must run on the installed machine. It previously imported
pandas, which is a ~45 MB compiled dependency whose only jobs here were reading
a CSV and holding columns. The maths is ordinary arithmetic, so pandas bought
nothing at runtime and cost install size and an aarch64 wheel risk.

This module is now the single implementation of the feature computation:

    model/predict.py        -> reads CSV with `csv`, calls pair_features()
    model/extract_features.py -> reads with pandas, calls pair_features(),
                                 wraps the result back into a DataFrame

Both paths therefore compute features the same way. Before this, the two had
separate copies of the maths, which is exactly how a trainer and a predictor
drift apart.

The surface
-----------
`pair_features(cols)` takes a dict of column name -> list of floats, in
timestamp order, and returns a dict of column name -> list of floats (or None
for undefined values). It is deliberately not pandas-shaped: the caller adapts.

Precision
---------
Every operation mirrors what pandas did, so results agree to within floating
point noise:

    ewm(span=N, adjust=False)   -> recursive y[i] = a*x[i] + (1-a)*y[i-1], y[0]=x[0]
    ewm(alpha=k, adjust=False)  -> same recursion with a = k
    shift(n)                    -> index offset, None where undefined
    diff()                      -> x[i] - x[i-1]
    clip(lower=0)               -> max(x, 0)
    rolling(w).min()/.max()     -> None until w values exist, then window extremes
    rolling(w).mean()           -> None until w values exist, then window mean
    rolling(w).std()            -> sample standard deviation (ddof=1)
    pct_change()                -> (x[i] - x[i-1]) / x[i-1]

A window containing an undefined value yields None rather than a partial result,
which is what pandas does with its default min_periods.

No third-party imports. No trading logic. Feature values only.
"""

import csv
import math
import os

# Must match train_model.py, or the model receives different features than it
# was fitted on.
FWD_BARS = 4
DEADZONE_PCT = 0.3     # extract_features.py: |return| below this is "flat"
WARMUP_BARS = 100      # extract_features.py: leading bars dropped per pair

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

# ---------------------------------------------------------------------------
# Coin tiers
# ---------------------------------------------------------------------------
# Copied verbatim from extract_features.py. It lives here as well because the
# predictor needs tier_of() and importing it from extract_features would pull
# pandas into the installed app - the very thing this module exists to avoid.
# The self-check asserts the two definitions agree, so they cannot drift apart.
MAJORS = {"BTC", "ETH", "XRP", "SOL"}
MEMES = {
    "PEPE", "DOGE", "BOME", "SUNDOG", "PIPPIN", "PENGU", "TROLLSOL",
    "WIF", "BONK", "FLOKI", "MEW", "POPCAT", "BRETT", "MOG",
}


def tier_of(symbol):
    """1 = major (model is near coin-flip, never vetoed), 3 = meme, 2 = other."""
    s = symbol.upper().replace("IDR", "")
    if s in MAJORS:
        return 1
    if s in MEMES:
        return 3
    return 2


# ---------------------------------------------------------------------------
# CSV reading (stdlib)
# ---------------------------------------------------------------------------
def read_ohlcv_csv(path, limit=None):
    """Read an OHLCV csv into {column: [floats]}.

    Returns (cols, timestamps) where timestamps is a list of ints. Rows that
    cannot be parsed are skipped rather than aborting the read — a single
    malformed line should not make a pair unscorable.
    """
    cols = {k: [] for k in REQUIRED_COLUMNS}
    ts = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return cols, ts
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"csv missing columns: {missing}")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            try:
                cols["timestamp"].append(int(float(row["timestamp"])))
                for k in ("open", "high", "low", "close", "volume"):
                    cols[k].append(float(row[k]))
            except (TypeError, ValueError):
                # drop the partially-appended row
                cols["timestamp"].pop()
                for k in ("open", "high", "low", "close", "volume"):
                    if len(cols[k]) > len(cols["timestamp"]):
                        cols[k].pop()
                continue
    ts = cols["timestamp"]
    return cols, ts


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _is_nan(x):
    return x is None or (isinstance(x, float) and x != x)


def ewm(values, alpha):
    """Recursive exponential weighted mean, adjust=False.

    y[0] = x[0]; y[i] = alpha*x[i] + (1-alpha)*y[i-1].
    Undefined input propagates: once a value is missing the recursion cannot
    continue, which matches how the training data is built (no gaps).
    """
    out = [None] * len(values)
    prev = None
    for i, x in enumerate(values):
        if _is_nan(x):
            prev = None
            continue
        prev = x if prev is None else alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def shift(values, n):
    out = [None] * len(values)
    for i in range(len(values)):
        j = i - n
        if 0 <= j < len(values):
            out[i] = values[j]
    return out


def diff(values):
    out = [None] * len(values)
    for i in range(1, len(values)):
        a, b = values[i], values[i - 1]
        if not _is_nan(a) and not _is_nan(b):
            out[i] = a - b
    return out


def pct_change(values):
    out = [None] * len(values)
    for i in range(1, len(values)):
        a, b = values[i], values[i - 1]
        if not _is_nan(a) and not _is_nan(b) and b != 0:
            out[i] = (a - b) / b
    return out


def _window(values, i, w):
    """The w values ending at i, or None if the window is incomplete/undefined."""
    if i + 1 < w:
        return None
    win = values[i + 1 - w:i + 1]
    for x in win:
        if _is_nan(x):
            return None
    return win


def rolling_min(values, w):
    out = [None] * len(values)
    for i in range(len(values)):
        win = _window(values, i, w)
        if win is not None:
            out[i] = min(win)
    return out


def rolling_max(values, w):
    out = [None] * len(values)
    for i in range(len(values)):
        win = _window(values, i, w)
        if win is not None:
            out[i] = max(win)
    return out


def rolling_mean(values, w):
    out = [None] * len(values)
    for i in range(len(values)):
        win = _window(values, i, w)
        if win is not None:
            out[i] = sum(win) / w
    return out


def rolling_std(values, w):
    """Sample standard deviation (ddof=1), matching pandas' default."""
    out = [None] * len(values)
    for i in range(len(values)):
        win = _window(values, i, w)
        if win is None or w < 2:
            continue
        m = sum(win) / w
        var = sum((x - m) ** 2 for x in win) / (w - 1)
        out[i] = math.sqrt(var) if var > 0 else 0.0
    return out


def _div(a, b, nan_on_zero=True):
    if _is_nan(a) or _is_nan(b):
        return None
    if b == 0:
        return None if nan_on_zero else 0.0
    return a / b


# ---------------------------------------------------------------------------
# The feature set
# ---------------------------------------------------------------------------
def pair_features(cols):
    """Compute every feature column from OHLCV columns.

    `cols` maps column name to a list of floats. Returns a dict of
    column name -> list, with None where a value is undefined. Includes
    `label`, which training uses and the predictor ignores.
    """
    c = cols["close"]
    h = cols["high"]
    low = cols["low"]
    v = cols["volume"]
    n = len(c)

    ema9 = ewm(c, 2.0 / (9 + 1))
    ema21 = ewm(c, 2.0 / (21 + 1))

    out = {}

    # --- trend ---
    out["ema_spread_pct"] = [
        None if (ema9[i] is None or ema21[i] is None) else
        (ema9[i] - ema21[i]) / ema21[i] * 100.0 if ema21[i] != 0 else None
        for i in range(n)
    ]

    ema21_shift8 = shift(ema21, 8)
    out["ema_slope_pct_hr"] = [
        None if (ema21[i] is None or ema21_shift8[i] is None) else
        (ema21[i] - ema21_shift8[i]) / ema21_shift8[i] * 100.0 / 2.0
        if ema21_shift8[i] != 0 else None
        for i in range(n)
    ]

    # --- Wilder's RSI(14) ---
    delta = diff(c)
    gain = [None if _is_nan(x) else max(x, 0.0) for x in delta]
    loss = [None if _is_nan(x) else max(-x, 0.0) for x in delta]
    ag = ewm(gain, 1.0 / 14.0)
    al = ewm(loss, 1.0 / 14.0)

    rsi = []
    for i in range(n):
        if ag[i] is None or al[i] is None:
            rsi.append(None)
            continue
        if al[i] == 0:
            # pandas: ag / al.replace(0, NaN) -> NaN, then the explicit
            # out.loc[al == 0] = 100.0 overwrites it.
            rsi.append(100.0)
            continue
        rs = ag[i] / al[i]
        rsi.append(100.0 - 100.0 / (1.0 + rs))
    out["rsi14"] = rsi

    # --- position in the 24h range ---
    low24 = rolling_min(low, 96)
    high24 = rolling_max(h, 96)
    out["range_pos"] = [
        None if (low24[i] is None or high24[i] is None) else
        _div(c[i] - low24[i], high24[i] - low24[i])
        for i in range(n)
    ]

    # --- volume ---
    avg_vol20 = rolling_mean(v, 20)
    out["vol_ratio"] = [None if avg_vol20[i] is None else _div(v[i], avg_vol20[i])
                        for i in range(n)]

    # --- higher highs ---
    h_prev = shift(h, 1)
    # pandas computes (h > h.shift(1)), which is a boolean Series: row 0 compares
    # against NaN and yields False. So an undefined previous high is 0.0 here,
    # not a missing value -- otherwise the first window would never complete and
    # hh_frac would start much later than the model expects.
    is_hh = [0.0 if h_prev[i] is None else (1.0 if h[i] > h_prev[i] else 0.0)
             for i in range(n)]
    out["hh_frac"] = rolling_mean(is_hh, 24)

    # --- returns ---
    pc = pct_change(c)
    out["ret1_pct"] = [None if x is None else x * 100.0 for x in pc]

    c_shift4 = shift(c, 4)
    out["ret4_pct"] = [
        None if (c_shift4[i] is None or c_shift4[i] == 0) else
        (c[i] / c_shift4[i] - 1.0) * 100.0
        for i in range(n)
    ]

    c_shift96 = shift(c, 96)
    out["ret24_pct"] = [
        None if (c_shift96[i] is None or c_shift96[i] == 0) else
        (c[i] / c_shift96[i] - 1.0) * 100.0
        for i in range(n)
    ]

    out["vol_std24"] = [None if x is None else x * 100.0
                        for x in rolling_std(pc, 24)]

    # --- training label: forward 4-bar return with a deadzone ---
    c_fwd = shift(c, -FWD_BARS)
    label = []
    for i in range(n):
        if c_fwd[i] is None or c[i] == 0:
            label.append(0)
            continue
        fwd = (c_fwd[i] - c[i]) / c[i] * 100.0
        if fwd > DEADZONE_PCT:
            label.append(1)
        elif fwd < -DEADZONE_PCT:
            label.append(-1)
        else:
            label.append(0)
    out["label"] = label

    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    failures = []

    def check(label, cond):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)

    print("=== features.py self-check ===")
    print()

    # --- 1. ewm matches the recursive definition ---
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = 2.0 / 10          # span=9
    got = ewm(vals, a)
    exp = [1.0]
    for x in vals[1:]:
        exp.append(a * x + (1 - a) * exp[-1])
    check("ewm matches the recursion", all(abs(g - e) < 1e-15 for g, e in zip(got, exp)))
    check("ewm seeds at the first value", got[0] == 1.0)

    # --- 2. shift / diff / pct_change ---
    check("shift(1)", shift([1, 2, 3], 1) == [None, 1, 2])
    check("shift(-1)", shift([1, 2, 3], -1) == [2, 3, None])
    check("diff", diff([1, 3, 6]) == [None, 2, 3])
    check("pct_change", pct_change([100, 110])[1] == 0.10000000000000009 or
          abs(pct_change([100, 110])[1] - 0.1) < 1e-12)

    # --- 3. rolling windows need a full window ---
    check("rolling_min undefined until w values",
          rolling_min([5, 4, 3], 3) == [None, None, 3])
    check("rolling_max", rolling_max([1, 5, 3], 3) == [None, None, 5])
    check("rolling_mean", rolling_mean([2.0, 4.0], 2) == [None, 3.0])

    # --- 4. rolling_std is the sample std (ddof=1) ---
    g = rolling_std([1.0, 2.0, 3.0, 4.0], 4)[3]
    # mean 2.5, sum sq dev 5, /3 = 1.6667, sqrt = 1.29099
    check("rolling_std uses ddof=1", abs(g - math.sqrt(5.0 / 3.0)) < 1e-12)

    # a window containing None yields None, not a partial result
    check("window with None yields None",
          rolling_mean([1.0, None, 3.0], 3) == [None, None, None])

    # --- 5. rsi: a monotonic rise has no losses, so RSI is 100 ---
    up = [float(i) for i in range(1, 40)]
    f = pair_features({"timestamp": list(range(len(up))), "open": up,
                       "high": up, "low": up, "close": up,
                       "volume": [1.0] * len(up)})
    check("rsi is 100 on a pure uptrend", f["rsi14"][-1] == 100.0)

    # --- 6. a flat series has no movement, so RSI is undefined or neutral ---
    flat = [5.0] * 40
    f2 = pair_features({"timestamp": list(range(40)), "open": flat, "high": flat,
                        "low": flat, "close": flat, "volume": [1.0] * 40})
    check("flat series does not crash", f2["rsi14"][-1] is None or
          isinstance(f2["rsi14"][-1], float))

    # --- 7. length agreement: every column matches the input length ---
    n = 250
    cols = {
        "timestamp": list(range(n)),
        "open": [100.0 + i * 0.5 for i in range(n)],
        "high": [101.0 + i * 0.5 for i in range(n)],
        "low": [99.0 + i * 0.5 for i in range(n)],
        "close": [100.0 + i * 0.5 for i in range(n)],
        "volume": [10.0 + (i % 7) for i in range(n)],
    }
    feats = pair_features(cols)
    bad = [k for k, v in feats.items() if len(v) != n]
    check("every feature column has the input length", not bad)

    # --- 8. warm-up: nothing is defined before its window is full ---
    # ema_spread_pct IS defined at bar 0 and equals exactly 0.0: both EMAs seed
    # at close[0] with adjust=False, so their difference is zero. That matches
    # pandas, so asserting None here would be wrong.
    check("ema_spread is exactly 0 at bar 0 (both EMAs seed at close[0])",
          feats["ema_spread_pct"][0] == 0.0)
    check("range_pos undefined before 96 bars",
          all(x is None for x in feats["range_pos"][:95]))
    check("range_pos defined at bar 95", feats["range_pos"][95] is not None)

    # --- 9. label deadzone ---
    check("label is one of -1/0/1",
          set(feats["label"]) <= {-1, 0, 1})

    # --- 10. tier table agrees with extract_features.py ---
    # extract_features cannot be imported here (it pulls in pandas), so read its
    # source and compare the literal set definitions.
    import re as _re
    ef = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "extract_features.py")).read()

    def _set_literal(name, text):
        m = _re.search(name + r"\s*=\s*\{([^}]*)\}", text, _re.S)
        if not m:
            return None
        return set(_re.findall(r'"([^"]+)"', m.group(1)))

    ef_majors = _set_literal("MAJORS", ef)
    ef_memes = _set_literal("MEMES", ef)
    check("MAJORS agrees with extract_features.py", ef_majors == MAJORS)
    check("MEMES agrees with extract_features.py", ef_memes == MEMES)

    # --- 11. no third-party imports ---
    src = open(__file__).read().split("def _self_check")[0]
    for mod in ("pandas", "numpy", "sklearn", "joblib"):
        check(f"no {mod} import", f"import {mod}" not in src)

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for x in failures:
            print("  -", x)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
