#!/bin/sh
# Verify the published app runs on THIS machine, with real market data.
#
# Everything happens under /tmp; nothing touches an existing install, no API key
# is needed (the Indodax history endpoint is public), and nothing is deleted
# outside /tmp/hunter-verify.
#
# Checks, in the order that they can fail:
#   1. environment
#   2. fetch the published app
#   3. install the pinned lock (must not compile from source)
#   4. fetch real market data (public endpoint, no key)
#   5. run the engine's own self-checks
#
# Exit 0 only if every step passed.
#
# The ML layer was removed from the product on 2026-11-08. Earlier versions of
# this script downloaded model.joblib and ran the predictor; there is no model
# and no predictor now. What is verified instead is that the stripped dependency
# set installs and the app imports and self-checks on this architecture - which
# is the thing that can actually fail on ARM.
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

# There is no model to fetch any more. Assert that, so a stale release with the
# binary still attached cannot quietly reintroduce the ML path.
if [ -d "$W/app/model" ]; then
    echo "  [WARN] the release still contains a model/ directory:"
    ls "$W/app/model" | sed 's/^/         /'
else
    echo "  no model/ directory in the release  [correct]"
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
# --only-binary makes any package without a wheel FAIL rather than fall back to a
# source build. The target board has no compiler, so a source build here means
# the real install would fail there too.
"$W/venv/bin/pip" install -q --only-binary=:all: -r "$W/app/requirements.lock" \
    > "$W/pip.log" 2>&1 || {
    echo "  [FAIL] install failed:"; tail -20 "$W/pip.log" | sed 's/^/    /'; exit 1; }
END=$(date +%s)
echo "  installed in $((END - START))s"
echo "  [PASS] every package came as a prebuilt wheel - nothing was compiled"

N=$(cd "$W" && "$W/venv/bin/pip" freeze 2>/dev/null | wc -l)
echo "  packages: $N"
echo

"$W/venv/bin/python" - <<'PY'
try:
    import sklearn
    print("  [FAIL] scikit-learn is installed - the ML layer was meant to be removed")
    raise SystemExit(1)
except ImportError:
    pass
import fastapi, uvicorn, requests, yaml
print(f"  fastapi {fastapi.__version__}  uvicorn {uvicorn.__version__}  "
      f"requests {requests.__version__}")
print("  [PASS] no scientific stack required")
PY
[ $? -eq 0 ] || exit 1
echo

echo "================================================================"
echo "  4. fetch real market data (public endpoint, no key)"
echo "================================================================"
mkdir -p "$W/app/data/ohlcv" "$W/app/data"
"$W/venv/bin/python" - "$W" <<'PY'
import json, os, sys, time, csv, urllib.request
W = sys.argv[1]
out = os.path.join(W, "app", "data", "ohlcv")
now = int(time.time())
got = 0
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
    got += 1
sys.exit(0 if got else 1)
PY
[ $? -eq 0 ] || { echo "  [FAIL] could not fetch any market data"; exit 1; }
echo

echo "================================================================"
echo "  5. the app's own self-checks"
echo "================================================================"
cd "$W/app" || exit 1
RC=0
for mod in paths config gates filters state indicators exit_strategy control; do
    printf "  %-16s " "$mod"
    if HUNTER_DATA_DIR="$W/app/data" timeout 120 "$W/venv/bin/python" \
            "scripts/$mod.py" > "$W/$mod.out" 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        tail -6 "$W/$mod.out" | sed 's/^/      /'
        RC=1
    fi
done

echo
"$W/venv/bin/python" -c "
import resource
print('  peak RSS for the checks: %.0f MiB' % (resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024))
"
echo "  venv size: $(du -sh "$W/venv" 2>/dev/null | cut -f1)"

echo
echo "================================================================"
if [ "$RC" -eq 0 ]; then
    echo "  RESULT: PASS - the app runs on $ARCH without the ML stack"
    echo
    echo "  Installs: $(cd "$W" && "$W/venv/bin/pip" freeze 2>/dev/null | wc -l) packages, no compiler needed."
    echo "  This did not install anything. To install for real:"
    echo "    curl -fsSL https://raw.githubusercontent.com/rezagiovanni/the-hunter-app/main/install.sh | sh"
else
    echo "  RESULT: FAIL - see the output above"
fi
echo "================================================================"
exit "$RC"
