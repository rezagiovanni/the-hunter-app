#!/usr/bin/env python3
"""
ML spike — Step 1: feature extraction for The Hunter's directional veto model.

Reads every 15m OHLCV CSV in data/ohlcv/, computes per-bar features + forward
1h direction label (±0.3% deadzone), adds coin tier and market regime features,
and writes one tidy dataset to model/features.csv.

Features (all normalized so they compare across coins):
    ema_spread_pct    (EMA9 - EMA21) / EMA21 * 100
    ema_slope_pct_hr  EMA21 change over 8 bars, as %/hour
    rsi14             Wilder's RSI
    range_pos         position of close in the last 96-bar (24h) range, 0..1
    vol_ratio         last bar volume / 20-bar average volume
    hh_frac           fraction of higher-highs over the last 24 bars
    ret1_pct          last 1-bar return %
    ret4_pct          last 4-bar (1h) return %
    ret24_pct         last 96-bar (24h) return %
    vol_std24         std of 1-bar returns over 24 bars (volatility)
    tier              1 = majors, 2 = mid caps, 3 = memes/small caps
    btc_trend         BTC's ema_spread_pct at the same timestamp (market regime)
    market_vol        mean vol_std24 across all pairs at the same timestamp

Label: forward 4-bar (1h) return — +1 up (> +0.3%), -1 down (< -0.3%), 0 = dropped
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Paths from scripts/paths.py so this runs from any install location.
# Training-time scripts (train_model, check_drift) deliberately keep their own
# paths: they are development tools and do not ship with the app.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paths import OHLCV_DIR as _OHLCV, MODEL_DIR as _MODEL  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OHLCV_DIR = Path(_OHLCV)
OUT_CSV = Path(_MODEL) / "features.csv"

# Coin tiers (symbol suffix IDR stripped). Verified against data/ohlcv listing.
MAJORS = {"BTC", "ETH", "XRP", "SOL"}
MEMES = {
    "PEPE", "DOGE", "BOME", "SUNDOG", "PIPPIN", "PENGU", "TROLLSOL",
    "WIF", "BONK", "FLOKI", "MEW", "POPCAT", "BRETT", "MOG",
}

WARMUP_BARS = 100          # drop this many leading bars per pair (indicator warmup)
FWD_BARS = 4               # label horizon: 4 x 15m = 1h
DEADZONE_PCT = 0.3         # |return| below this is "flat" -> sample dropped


def tier_of(symbol: str) -> int:
    s = symbol.upper().replace("IDR", "")
    if s in MAJORS:
        return 1
    if s in MEMES:
        return 3
    return 2


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling features on one pair's bars."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["ema_spread_pct"] = (ema9 - ema21) / ema21 * 100
    out["ema_slope_pct_hr"] = (ema21 - ema21.shift(8)) / ema21.shift(8) * 100 / 2

    # Wilder's RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)
    out.loc[al == 0, "rsi14"] = 100.0

    low24 = l.rolling(96).min()
    high24 = h.rolling(96).max()
    out["range_pos"] = (c - low24) / (high24 - low24)

    avg_vol20 = v.rolling(20).mean()
    out["vol_ratio"] = v / avg_vol20.replace(0, np.nan)

    out["hh_frac"] = (h > h.shift(1)).rolling(24).mean()

    out["ret1_pct"] = c.pct_change() * 100
    out["ret4_pct"] = (c / c.shift(4) - 1) * 100
    out["ret24_pct"] = (c / c.shift(96) - 1) * 100
    out["vol_std24"] = c.pct_change().rolling(24).std() * 100

    # Label: forward 4-bar return with deadzone
    fwd = (c.shift(-FWD_BARS) - c) / c * 100
    out["label"] = np.where(fwd > DEADZONE_PCT, 1,
                            np.where(fwd < -DEADZONE_PCT, -1, 0))
    return out


def main():
    t0 = time.time()
    files = sorted(OHLCV_DIR.glob("*.csv"))
    print(f"[extract] {len(files)} pair CSVs found")

    frames = []
    for i, f in enumerate(files, 1):
        symbol = f.stem.upper()
        df = pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        if len(df) < WARMUP_BARS + FWD_BARS + 10:
            print(f"[extract] {symbol}: only {len(df)} bars — skipped")
            continue
        feats = pair_features(df)
        feats["timestamp"] = df["timestamp"].values
        feats["pair"] = symbol
        feats["tier"] = tier_of(symbol)
        feats = feats.iloc[WARMUP_BARS:]          # drop indicator warmup
        feats = feats[feats["label"] != 0]        # drop flat / no-label rows
        frames.append(feats)
        if i % 10 == 0 or i == len(files):
            print(f"[extract] {i}/{len(files)} pairs done ({time.time() - t0:.0f}s)")

    data = pd.concat(frames, ignore_index=True)
    print(f"[extract] per-pair features: {len(data):,} labeled samples")

    # ---- market regime features (global pass) ----
    # BTC trend at each timestamp
    btc = data[data["pair"] == "BTCIDR"][["timestamp", "ema_spread_pct"]]
    btc = btc.rename(columns={"ema_spread_pct": "btc_trend"})
    data = data.drop(columns=["btc_trend"], errors="ignore").merge(btc, on="timestamp", how="left")

    # Market-wide average volatility at each timestamp
    mkt = data.groupby("timestamp")["vol_std24"].mean().rename("market_vol").reset_index()
    data = data.drop(columns=["market_vol"], errors="ignore").merge(mkt, on="timestamp", how="left")

    data = data.dropna(subset=["btc_trend", "market_vol"])

    cols = ["timestamp", "pair", "tier", "ema_spread_pct", "ema_slope_pct_hr", "rsi14",
            "range_pos", "vol_ratio", "hh_frac", "ret1_pct", "ret4_pct", "ret24_pct",
            "vol_std24", "btc_trend", "market_vol", "label"]
    data = data[cols]
    data.to_csv(OUT_CSV, index=False)

    print(f"[extract] saved {OUT_CSV.name}: {len(data):,} rows, {data['pair'].nunique()} pairs")
    print(f"[extract] label balance: up={int((data['label'] == 1).sum()):,} "
          f"down={int((data['label'] == -1).sum()):,}")
    ts = pd.to_datetime(data["timestamp"], unit="s")
    print(f"[extract] range: {ts.min()} → {ts.max()}")
    print(f"[extract] tiers: {data.groupby('tier')['pair'].nunique().to_dict()}")
    print(f"[extract] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
