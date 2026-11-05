#!/usr/bin/env python3
"""
ML spike — Step 3: live predictor.

Loads model.joblib, computes features for the LAST bar of every pair in the
top-20 list (same computation as training — full history so EMAs match),
applies market regime features, and writes model/predictions.json.

Runs under model/venv/bin/python (has sklearn). The dashboard reads
predictions.json and refreshes it in the background when stale.

Veto rules:
    tier 1 (majors)      -> no signal (model is ~coin-flip on them: 51.6%)
    tier 2/3, P(down) >= 60% -> VETO  (out-of-sample precision: 72.3%)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib

# The model was fitted on a DataFrame and warns when handed a bare list. That
# warning is suppressed because main() enforces the same property up front: it
# compares FEATURES against the model's own feature_names_in_ and refuses to
# run if they differ. Passing a DataFrame would silence sklearn but would
# require pandas. Left unsuppressed this prints 20 lines per run, which cron
# would mail out every time the predictor refreshes.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# Paths from scripts/paths.py so the predictor runs from any install location.
# predict.py lives in model/, so add scripts/ to the path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paths import (  # noqa: E402
    DATA_DIR as _DATA, OHLCV_DIR as _OHLCV, PAIRS_FILE as _PAIRS,
    MODEL_DIR as _MODEL,
)

BASE = Path(__file__).resolve().parent.parent
OHLCV_DIR = Path(_OHLCV)
PAIRS_FILE = Path(_PAIRS)
MODEL_PATH = Path(_MODEL) / "model.joblib"
OUT_JSON = Path(_MODEL) / "predictions.json"

VETO_THRESHOLD = 0.60

sys.path.insert(0, str(BASE / "model"))
# Feature maths is now shared, pure-stdlib code (model/features.py). Dropping
# pandas here removes a ~45 MB compiled dependency from the installed app while
# computing exactly the same numbers.
from features import pair_features, read_ohlcv_csv, tier_of  # noqa: E402

FEATURES = [
    "ema_spread_pct", "ema_slope_pct_hr", "rsi14", "range_pos", "vol_ratio",
    "hh_frac", "ret1_pct", "ret4_pct", "ret24_pct", "vol_std24",
    "btc_trend", "market_vol", "tier",
]

# The subset computed per pair by features.py; btc_trend, market_vol and tier
# are added afterwards from other pairs and the tier table.
FEATURE_COLS = [f for f in FEATURES if f not in ("btc_trend", "market_vol", "tier")]

PRED_LOG = BASE / "model" / "prediction_log.jsonl"


def scored_bar(symbol: str):
    """(features_row, timestamp) of the pair's last bar — full-history EMA-correct.

    `features_row` is a dict of column -> value. Full history is used (not a
    tail window) so the EMAs match training, which is why this reads the whole
    file rather than the last N rows.
    """
    csv_path = OHLCV_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        return None, None
    try:
        cols, ts = read_ohlcv_csv(csv_path)
        if len(ts) < 200:
            return None, None
        feats = pair_features(cols)
        row = {k: v[-1] for k, v in feats.items()}
        # a missing feature means this bar cannot be scored (window not full)
        if any(row[k] is None for k in FEATURE_COLS):
            return None, None
        return row, int(ts[-1])
    except Exception as exc:
        print(f"[predict] {symbol}: {exc}")
        return None, None


def main():
    t0 = time.time()
    if not MODEL_PATH.exists():
        print("[predict] model.joblib missing — run train_model.py first")
        return 1
    clf = joblib.load(MODEL_PATH)

    # Guard against feature drift. If FEATURES is reordered or renamed without
    # retraining, the model would be fed the right numbers under the wrong names
    # and produce confident nonsense. Fail loudly instead.
    fitted = list(getattr(clf, "feature_names_in_", []))
    if fitted and fitted != FEATURES:
        print("[predict] FATAL: feature order does not match the fitted model")
        print(f"[predict]   model expects: {fitted}")
        print(f"[predict]   this file has: {FEATURES}")
        print("[predict]   refusing to write predictions from mismatched features")
        return 1

    with open(PAIRS_FILE) as fh:
        pairs = json.load(fh)
    top20 = pairs[:20]
    symbols = [p["symbol"].upper() for p in top20]

    # ---- per-pair last-bar features ----
    scored = []
    for sym in symbols:
        row, ts = scored_bar(sym)
        if row is None:
            print(f"[predict] {sym}: no usable last bar — skipped")
            continue
        scored.append((sym, row, ts))

    if not scored:
        print("[predict] no pairs could be scored")
        return 1

    # ---- market regime features ----
    # BTC trend: ema_spread_pct at each pair's last timestamp (latest BTC bar <= t)
    btc_by_ts = {}
    csv_path = OHLCV_DIR / "BTCIDR.csv"
    if csv_path.exists():
        btc_cols, btc_ts = read_ohlcv_csv(csv_path)
        btc_series = pair_features(btc_cols)["ema_spread_pct"]
        # sorted timestamp -> value, for the nearest-prior lookup
        btc_by_ts = {t: v for t, v in zip(btc_ts, btc_series)}

    _btc_sorted = sorted(btc_by_ts)

    def btc_trend_at(ts: int) -> float:
        if not btc_by_ts:
            return 0.0
        # nearest BTC bar at or before ts (bisect over the sorted timestamps)
        import bisect
        idx = bisect.bisect_right(_btc_sorted, ts) - 1
        if idx < 0:
            return 0.0
        val = btc_by_ts[_btc_sorted[idx]]
        if val is None or (isinstance(val, float) and val != val):
            return 0.0
        return float(val)

    # market_vol: mean vol_std24 across scored pairs sharing the same timestamp
    vol_by_ts = {}
    for sym, row, ts in scored:
        vol_by_ts.setdefault(ts, []).append(float(row["vol_std24"]))

    # ---- predict ----
    predictions = {}
    for sym, row, ts in scored:
        feats = {
            "ema_spread_pct": float(row["ema_spread_pct"]),
            "ema_slope_pct_hr": float(row["ema_slope_pct_hr"]),
            "rsi14": float(row["rsi14"]),
            "range_pos": float(row["range_pos"]),
            "vol_ratio": float(row["vol_ratio"]),
            "hh_frac": float(row["hh_frac"]),
            "ret1_pct": float(row["ret1_pct"]),
            "ret4_pct": float(row["ret4_pct"]),
            "ret24_pct": float(row["ret24_pct"]),
            "vol_std24": float(row["vol_std24"]),
            "btc_trend": btc_trend_at(ts),
            "market_vol": float(sum(vol_by_ts[ts]) / len(vol_by_ts[ts])),
            "tier": tier_of(sym),
        }
        # A bare list is deliberate. Feeding the model a DataFrame would supply
        # the column names it was fitted with and silence its warning, but that
        # requires pandas, which this module exists to avoid. The names are
        # enforced instead by the feature-drift guard in main(), which compares
        # FEATURES against the model's own feature_names_in_ and refuses to run
        # if they differ. So the ordering is checked up front rather than at
        # every call.
        X = [[feats[name] for name in FEATURES]]
        p_down = float(clf.predict_proba(X)[0][1])
        tier = feats["tier"]
        veto = tier != 1 and p_down >= VETO_THRESHOLD
        predictions[sym] = {
            "p_down": round(p_down, 3),
            "tier": tier,
            "veto": veto,
            "bar_ts": ts,
        }

    out = {
        "generated_at": time.time(),
        "veto_threshold": VETO_THRESHOLD,
        "predictions": predictions,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- append to prediction log (one line per pair per NEW bar) ----
    # This is the drift monitor's raw material: check_drift.py scores these
    # against what actually happened 1h later.
    last_logged = {}
    if PRED_LOG.exists():
        try:
            with open(PRED_LOG) as fh:
                for ln in fh:
                    try:
                        rec = json.loads(ln)
                        last_logged[rec["pair"]] = rec["bar_ts"]
                    except Exception:
                        continue
        except Exception:
            pass
    new_lines = []
    for sym, v in predictions.items():
        if last_logged.get(sym) == v["bar_ts"]:
            continue  # same bar already logged — no new prediction
        new_lines.append(json.dumps({
            "logged_at": int(time.time()),
            "pair": sym,
            "bar_ts": v["bar_ts"],
            "p_down": v["p_down"],
            "tier": v["tier"],
            "veto": v["veto"],
        }))
    if new_lines:
        with open(PRED_LOG, "a") as fh:
            fh.write("\n".join(new_lines) + "\n")

    vetoed = [s for s, v in predictions.items() if v["veto"]]
    print(f"[predict] scored {len(predictions)} pairs in {time.time() - t0:.1f}s")
    print(f"[predict] vetoes (P(down) >= {VETO_THRESHOLD:.0%}): {vetoed if vetoed else 'none'}")
    for sym, v in sorted(predictions.items(), key=lambda kv: -kv[1]["p_down"])[:5]:
        tag = " VETO" if v["veto"] else (" (majors)" if v["tier"] == 1 else "")
        print(f"  {sym}: P(down)={v['p_down']:.0%}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
