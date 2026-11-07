#!/usr/bin/env python3
"""
ml_report.py — did the model get it right?

Why this exists
---------------
predict.py logs every prediction to model/prediction_log.jsonl:

    {"logged_at": ..., "pair": "BTCIDR", "bar_ts": ..., "p_down": 0.552,
     "tier": 1, "veto": false}

...but it never records what happened next. So the model's real accuracy was
unknowable from the app: `hunter status` could only say "no predictions yet" or
how old the file was. Every accuracy figure had to be reconstructed by hand by
joining timestamps to CSVs - which is not something the owner of the bot should
need to do.

This module closes that gap. It:

  outcome   append the actual result once a prediction's horizon has closed,
            as a separate record so the log stays append-only and a re-run is
            idempotent (keyed by pair + bar_ts)
  report    rolling precision, calibration by p_down band, and the veto's
            win/loss record

The label rule matches training exactly (model/features.py):

    fwd = (close[t + FWD_BARS] - close[t]) / close[t] * 100
    fwd < -DEADZONE  -> down   (the model's positive class)
    fwd > +DEADZONE  -> up
    otherwise        -> flat

Everything here is read-only against the CSVs and append-only against the log.
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths                                    # noqa: E402
from ohlcv_io import read_csv                   # noqa: E402

FWD_BARS = 4            # must match model/features.py
DEADZONE = 0.3          # percent; must match model/features.py
VETO_THRESHOLD = 0.60

PRED_LOG = os.path.join(paths.MODEL_DIR, "prediction_log.jsonl")
OUTCOME_LOG = os.path.join(paths.MODEL_DIR, "prediction_outcomes.jsonl")


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------
def read_records(path=PRED_LOG):
    """Every well-formed record in the log, as a list of dicts.

    The file holds two kinds of line: 'prediction' records (written by
    predict.py) and 'outcome' records (written here). Old lines predate the
    'kind' field and are predictions.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("pair"):
                rec.setdefault("kind", "prediction")
                out.append(rec)
    return out


def load_outcomes(path=OUTCOME_LOG):
    """{(pair, bar_ts): outcome record} from the separate outcomes file."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("pair") and rec.get("bar_ts") is not None:
                done[(rec["pair"], rec["bar_ts"])] = rec
    return done


def _series(symbol):
    """(timestamp -> index, rows) for one pair, or (None, None)."""
    p = os.path.join(paths.OHLCV_DIR, f"{symbol}.csv")
    if not os.path.exists(p):
        return None, None
    rows = read_csv(p)
    if not rows:
        return None, None
    return {r["timestamp"]: i for i, r in enumerate(rows)}, rows


# ---------------------------------------------------------------------------
# outcome: record what actually happened
# ---------------------------------------------------------------------------
def record_outcomes(verbose=True, path=PRED_LOG, out_path=OUTCOME_LOG):
    """Append an outcome for every prediction whose horizon has closed.

    Idempotent: a prediction already scored is skipped, so running this twice
    (or on every dashboard load) cannot double-count.

    A prediction whose bar is too near the end of the CSV is LEFT ALONE rather
    than scored as flat - there is simply no future bar to measure yet, and
    scoring it would invent a result.
    """
    preds = [r for r in read_records(path) if r.get("kind") == "prediction"]
    done = load_outcomes(out_path)

    cache = {}
    new = []
    pending = 0

    for r in preds:
        key = (r["pair"], r["bar_ts"])
        if key in done:
            continue
        sym = r["pair"]
        if sym not in cache:
            cache[sym] = _series(sym)
        idxmap, rows = cache[sym]
        if idxmap is None:
            continue
        i = idxmap.get(r["bar_ts"])
        if i is None:
            continue
        j = i + FWD_BARS
        if j >= len(rows):
            pending += 1
            continue
        c0, c1 = rows[i]["close"], rows[j]["close"]
        if not c0:
            continue
        fwd = (c1 - c0) / c0 * 100.0

        if fwd < -DEADZONE:
            actual = "down"
        elif fwd > DEADZONE:
            actual = "up"
        else:
            actual = "flat"

        worst = min(x["low"] for x in rows[i + 1:j + 1])
        best = max(x["high"] for x in rows[i + 1:j + 1])
        new.append({
            "kind": "outcome",
            "pair": sym,
            "bar_ts": r["bar_ts"],
            "horizon_ts": rows[j]["timestamp"],
            "p_down": r.get("p_down"),
            "veto": bool(r.get("veto")),
            "tier": r.get("tier"),
            "fwd_pct": round(fwd, 4),
            "max_up_pct": round((best - c0) / c0 * 100.0, 4),
            "max_down_pct": round((worst - c0) / c0 * 100.0, 4),
            "actual": actual,
            "scored_at": int(time.time()),
        })

    if new:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a") as fh:
            for rec in new:
                fh.write(json.dumps(rec) + "\n")

    if verbose:
        print(f"  scored {len(new):,} new outcomes "
              f"({len(done):,} already had one, {pending:,} still pending)")
    return len(new)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report(out_path=OUTCOME_LOG):
    """Aggregate the outcomes into the numbers that matter."""
    recs = list(load_outcomes(out_path).values())
    if not recs:
        return None

    n = len(recs)
    down_n = sum(1 for r in recs if r["actual"] == "down")
    base = down_n / n

    bands = {}
    for r in recs:
        p = r.get("p_down")
        if p is None:
            continue
        b = round(min(int(p * 10) / 10, 0.9), 1)
        d = bands.setdefault(b, {"n": 0, "down": 0})
        d["n"] += 1
        d["down"] += 1 if r["actual"] == "down" else 0

    vetoed = [r for r in recs if r.get("veto")]
    allowed = [r for r in recs if not r.get("veto")]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": n,
        "base_rate": base,
        "bands": dict(sorted(bands.items())),
        "vetoed": {
            "n": len(vetoed),
            "down_rate": (sum(1 for r in vetoed if r["actual"] == "down") / len(vetoed)
                          if vetoed else 0.0),
            "mean_fwd": mean([r["fwd_pct"] for r in vetoed]),
        },
        "allowed": {
            "n": len(allowed),
            "down_rate": (sum(1 for r in allowed if r["actual"] == "down") / len(allowed)
                          if allowed else 0.0),
            "mean_fwd": mean([r["fwd_pct"] for r in allowed]),
        },
    }


def render_report(rep):
    if not rep:
        return ("  No scored predictions yet.\n"
                "  Run:  hunter ml outcome   (once some predictions have a future bar)")

    L = []
    L.append("  LIVE PREDICTION SCORECARD")
    L.append("  " + "-" * 60)
    L.append(f"  scored predictions : {rep['n']:,}")
    L.append(f"  base rate (fell)   : {rep['base_rate'] * 100:.1f}%")
    L.append("")
    L.append("  Calibration — when the model says p_down, how often is it right?")
    L.append(f"  {'p_down':>8}  {'n':>7}  {'fell':>7}  {'vs base':>9}")
    for b, d in rep["bands"].items():
        if d["n"] < 20:
            continue
        rate = d["down"] / d["n"] * 100
        L.append(f"  {b:>8.1f}  {d['n']:>7,}  {rate:>6.1f}%  "
                 f"{rate - rep['base_rate'] * 100:>+8.1f}")
    L.append("")

    v, a = rep["vetoed"], rep["allowed"]
    L.append("  The veto")
    if v["n"]:
        L.append(f"    vetoed   : {v['n']:>6,}  fell {v['down_rate'] * 100:>5.1f}%  "
                 f"mean move {v['mean_fwd']:+.3f}%")
        L.append(f"    allowed  : {a['n']:>6,}  fell {a['down_rate'] * 100:>5.1f}%  "
                 f"mean move {a['mean_fwd']:+.3f}%")
        diff = (v["down_rate"] - a["down_rate"]) * 100
        L.append(f"    the vetoed bucket falls {diff:+.1f} points more often")
        verdict = ("the veto is helping" if v["mean_fwd"] < a["mean_fwd"]
                   else "the veto is HURTING (vetoed were better)")
        L.append(f"    verdict  : {verdict}")
    else:
        L.append("    has not fired yet")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    import shutil
    import tempfile

    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {str(detail)[:300]}")

    td = tempfile.mkdtemp()
    old_ohlcv = paths.OHLCV_DIR
    try:
        # three bars per pair: index 0 is the prediction, index 4 would be the
        # horizon - so with 5 bars the first prediction is scorable and a
        # prediction on the last bar is NOT.
        paths.OHLCV_DIR = os.path.join(td, "ohlcv")
        os.makedirs(paths.OHLCV_DIR)

        def bars(sym, closes):
            os.makedirs(paths.OHLCV_DIR, exist_ok=True)
            with open(os.path.join(paths.OHLCV_DIR, sym + ".csv"), "w", newline="") as f:
                f.write("timestamp,open,high,low,close,volume\r\n")
                for i, c in enumerate(closes):
                    f.write(f"{1000 + i * 900},{c},{c * 1.01},{c * 0.99},{c},1.0\r\n")

        # AAA: falls 1% by bar 4 -> down
        bars("AAAIDR", [100, 100, 100, 100, 99.0])
        # BBB: rises 1% -> up
        bars("BBBIDR", [100, 100, 100, 100, 101.0])
        # CCC: flat -> flat
        bars("CCCIDR", [100, 100, 100, 100, 100.0])
        # DDD: prediction on the LAST bar - no horizon yet, must stay pending
        bars("DDDIDR", [100, 100, 100, 100, 100.0])

        # bar_ts 1000 is index 0, so index 4 (the horizon) EXISTS for the first
        # three. DDDIDR's prediction is on its LAST bar (index 4, ts 4600): there
        # is no future bar yet, so it must stay unscored. Pointing DDDIDR at
        # ts 1000 instead would make it scorable and the check would be wrong.
        LAST_TS = 1000 + 4 * 900
        predlog = os.path.join(td, "pred.jsonl")
        outlog = os.path.join(td, "out.jsonl")
        with open(predlog, "w") as fh:
            for sym, p, veto, ts in (("AAAIDR", 0.70, True, 1000),
                                     ("BBBIDR", 0.70, True, 1000),
                                     ("CCCIDR", 0.40, False, 1000),
                                     ("DDDIDR", 0.40, False, LAST_TS)):
                fh.write(json.dumps({"logged_at": 1, "pair": sym, "bar_ts": ts,
                                     "p_down": p, "tier": 2, "veto": veto}) + "\n")

        n = record_outcomes(verbose=False, path=predlog, out_path=outlog)
        check("scored the predictions that had a future bar", n == 3, n)

        outs = load_outcomes(outlog)
        check("a fall is labelled down", outs[("AAAIDR", 1000)]["actual"] == "down")
        check("a rise is labelled up", outs[("BBBIDR", 1000)]["actual"] == "up")
        check("a small move is labelled flat", outs[("CCCIDR", 1000)]["actual"] == "flat")
        check("a prediction with no future bar is NOT scored",
              ("DDDIDR", 1000 + 4 * 900) not in outs)

        # idempotent
        n2 = record_outcomes(verbose=False, path=predlog, out_path=outlog)
        check("re-running scores nothing new", n2 == 0, n2)
        check("the outcome count did not double", len(load_outcomes(outlog)) == 3)

        # mfe/mae are recorded with the right sign
        a = outs[("AAAIDR", 1000)]
        check("max_down is negative for a fall", a["max_down_pct"] < 0, a["max_down_pct"])
        check("fwd_pct matches the label direction",
              (a["fwd_pct"] < 0) == (a["actual"] == "down"), a["fwd_pct"])

        # report shape
        rep = build_report(outlog)
        check("report counts every scored prediction", rep["n"] == 3, rep["n"])
        check("report separates vetoed from allowed",
              rep["vetoed"]["n"] == 2 and rep["allowed"]["n"] == 1, rep)

        text = render_report(rep)
        check("render_report produces readable output", "SCORECARD" in text)
        check("render_report handles an empty log",
              "No scored predictions yet" in render_report(None))

    finally:
        paths.OHLCV_DIR = old_ohlcv
        shutil.rmtree(td, ignore_errors=True)

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="ml_report", description=__doc__.split("\n")[1])
    ap.add_argument("action", nargs="?", default="report",
                    choices=["outcome", "report", "both"])
    a = ap.parse_args(argv)

    if a.action in ("outcome", "both"):
        record_outcomes()
    if a.action in ("report", "both"):
        print()
        print(render_report(build_report()))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        sys.exit(_self_check())
    sys.exit(main(sys.argv[1:]))
