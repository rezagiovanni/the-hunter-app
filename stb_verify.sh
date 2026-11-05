#!/bin/sh
# Verify the published app's predictor works on this machine.
#
# Everything happens under /tmp; nothing touches an existing install, no API key
# is needed (the Indodax history endpoint is public), and no data is deleted
# outside /tmp/hunter-verify.
#
# Checks, in the order that they can fail:
#   1. architecture and interpreter
#   2. the pinned lock installs WITHOUT compiling anything from source
#   3. real OHLCV can be fetched
#   4. predict.py produces predictions.json (the thing the bot's veto gate reads)
#   5. how long it took and how much memory it used
#
# Exit 0 only if the predictor produced a well-formed predictions file.
set -u

W=/tmp/hunter-verify
rm -rf "$W"

echo "================================================================"
echo "  1. environment"
echo "================================================================"
ARCH=$(uname -m)
echo "  arch    : $ARCH"
echo "  python  : $(python3 --version 2>&1)"
echo "  cores   : $(nproc 2>/dev/null || echo '?')"
free -h 2>/dev/null | head -2 | sed 's/^/  /'
echo

echo "================================================================"
echo "  2. fetch the published app"
echo "================================================================"
if ! command -v git >/dev/null 2>&1; then
    echo "  [SKIP] git is not installed; cannot clone."
    echo "         Install it with:  sudo apt install -y git"
    exit 1
fi
git clone --depth 1 -q https://github.com/rezagiovanni/the-hunter-app "$W/app" \
    || { echo "  [FAIL] clone failed"; exit 1; }
echo "  cloned: $(find "$W/app" -type f | wc -l) files"

# The model is a release ASSET, not a tracked file: it is ~1.1 MB of binary and
# would bloat every clone. The real installer downloads it too, so anything that
# clones instead of installing has to fetch it explicitly.
echo "  fetching model.joblib from the v0.1.0 release..."
if curl -fsSL -o "$W/app/model/model.joblib" \
     "https://github.com/rezagiovanni/the-hunter-app/releases/download/v0.1.0/model.joblib"; then
    SZ=$(wc -c < "$W/app/model/model.joblib")
    echo "  model: $SZ bytes"
    [ "$SZ" -gt 500000 ] || { echo "  [FAIL] model looks truncated"; exit 1; }
else
    echo "  [FAIL] could not download model.joblib"; exit 1
fi

if ! python3 -m venv "$W/venv" >/dev/null 2>&1; then
    echo "  [FAIL] python3 -m venv is unavailable."
    echo "         On Debian/Ubuntu/Armbian:  sudo apt install -y python3-venv"
    exit 1
fi
echo "  venv created"
echo

echo "================================================================"
echo "  3. install the pinned lock (must not compile from source)"
echo "================================================================"
"$W/venv/bin/pip" install --upgrade pip -q 2>&1 | tail -2 | sed 's/^/  /'
START=$(date +%s)
"$W/venv/bin/pip" install -r "$W/app/requirements.lock" > "$W/pip.log" 2>&1 || {
    echo "  [FAIL] install failed:"; tail -20 "$W/pip.log" | sed 's/^/    /'; exit 1; }
END=$(date +%s)
echo "  installed in $((END - START))s"

if grep -qE "Building wheel|Running setup\.py" "$W/pip.log"; then
    echo "  [WARN] something was built from source on this machine:"
    grep -E "Building wheel|Running setup\.py" "$W/pip.log" | head -5 | sed 's/^/    /'
    echo "         That is not fatal, but it means a C toolchain was needed."
else
    echo "  [PASS] no source builds - every package came as a prebuilt wheel"
fi

"$W/venv/bin/python" - <<'PY'
import sklearn, numpy, joblib, fastapi
print(f"  sklearn {sklearn.__version__}  numpy {numpy.__version__}  "
      f"joblib {joblib.__version__}  fastapi {fastapi.__version__}")
PY
echo

echo "================================================================"
echo "  4. fetch real market data (public endpoint, no key)"
echo "================================================================"
mkdir -p "$W/app/data/ohlcv" "$W/app/data"
head -c 0 /dev/null
"$W/venv/bin/python" - "$W" <<'PY'
import json, os, sys, time, csv, urllib.request
W = sys.argv[1]
out = os.path.join(W, "app", "data", "ohlcv")
now = int(time.time())
for sym in ("BTCIDR", "ETHIDR", "SOLIDR"):
    url = ("https://indodax.com/tradingview/history_v2?from=%d&to=%d&tf=15&symbol=%s"
           % (now - 90 * 86400, now, sym))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    except Exception as e:
        print(f"  {sym}: ERROR {e}")
        continue
    if not isinstance(data, list) or not data:
        print(f"  {sym}: no data ({str(data)[:80]})")
        continue
    p = os.path.join(out, sym + ".csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for r in data:
            w.writerow([r["Time"], r["Open"], r["High"], r["Low"], r["Close"], r["Volume"]])
    print(f"  {sym}: {len(data)} bars")
PY

# predict.py reads data/pairs.json (the eligible-pair list). It is produced by
# the market fetcher, which uses the same public, keyless endpoint.
echo "  building data/pairs.json via fetch_market.py..."
( cd "$W/app" && HUNTER_DATA_DIR="$W/app/data" timeout 180 "$W/venv/bin/python" \
    scripts/fetch_market.py > "$W/market.log" 2>&1 ) && echo "  pairs.json built" \
    || { echo "  [WARN] fetch_market failed; using a minimal pairs.json"; }

"$W/venv/bin/python" - "$W" <<'PY'
import json, os, sys
W = sys.argv[1]
pf = os.path.join(W, "app", "data", "pairs.json")
if not os.path.exists(pf):
    json.dump([{"symbol": s} for s in ("BTCIDR", "ETHIDR", "SOLIDR")],
              open(pf, "w"))
    print("  wrote a minimal pairs.json (3 pairs)")
else:
    d = json.load(open(pf))
    print(f"  pairs.json already present: {len(d)} pairs")
PY
echo

echo "================================================================"
echo "  5. run the predictor"
echo "================================================================"
cd "$W/app"
"$W/venv/bin/python" - "$W" <<'PY'
import resource, subprocess, sys, time, os, json
W = sys.argv[1]
app = os.path.join(W, "app")
env = dict(os.environ)
env["HUNTER_DATA_DIR"] = os.path.join(app, "data")
t = time.time()
r = subprocess.run([sys.executable, "model/predict.py"], capture_output=True,
                   text=True, cwd=app, env=env)
el = time.time() - t
out = (r.stdout or "") + (r.stderr or "")
print("  --- predictor output ---")
for line in out.strip().splitlines()[-14:]:
    print("  " + line)
print("  ------------------------")
print(f"  exit       : {r.returncode}")
print(f"  elapsed    : {el:.1f}s")
print(f"  peak RSS   : {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024:.0f} MiB")

pj = os.path.join(app, "model", "predictions.json")
if os.path.exists(pj):
    d = json.load(open(pj))
    pairs = d.get("pairs", d) if isinstance(d, dict) else d
    n = len(pairs) if hasattr(pairs, "__len__") else "?"
    print(f"  [PASS] predictions.json written, {n} pairs scored")
    sample = list(pairs.items())[:3] if isinstance(pairs, dict) else pairs[:3]
    for k, v in sample:
        print(f"         {k}: {v}")
    sys.exit(0 if r.returncode == 0 else 2)
print("  [FAIL] predictions.json was NOT written")
sys.exit(1)
PY
RC=$?

echo
echo "================================================================"
if [ "$RC" -eq 0 ]; then
    echo "  RESULT: PASS - the predictor runs on $ARCH"
    echo "  Copy the lines above back to LOKI so the result can be recorded."
else
    echo "  RESULT: FAIL (exit $RC) - copy the output above back to LOKI"
fi
echo "================================================================"
exit "$RC"
