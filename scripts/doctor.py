#!/usr/bin/env python3
"""
doctor.py — diagnose a broken install and name the specific fault.

Design rule
-----------
A doctor that prints all-green is worse than no doctor, so every check reports
one of three outcomes and ALWAYS says what to do about a failure:

    [PASS]  working
    [WARN]  working but degraded, or worth knowing about
    [FAIL]  broken, with a Fix: line naming the exact command

Checks are independent and each is wrapped, so a crash inside one check reports
that check as failed rather than killing the run. A diagnostic that dies
mid-way is useless precisely when it is needed.

Nothing here repairs anything. `doctor` only looks.

What it checks (PRD UC-5, plus the things this codebase actually depends on)
--------------------------------------------------------------------------
  python, venv, dependencies
  install layout and permissions
  config presence, parseability and validity
  credentials
  crontab block integrity
  data freshness and coverage
  model and predictions
  the dashboard port and its health endpoint
  disk space
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
import term  # noqa: E402

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, label, detail="", fix=""):
        self.rows.append({"status": status, "label": label,
                          "detail": detail, "fix": fix})

    def ok(self, label, detail=""):
        self.add(PASS, label, detail)

    def warn(self, label, detail="", fix=""):
        self.add(WARN, label, detail, fix)

    def fail(self, label, detail="", fix=""):
        self.add(FAIL, label, detail, fix)

    def counts(self):
        c = {PASS: 0, WARN: 0, FAIL: 0}
        for r in self.rows:
            c[r["status"]] += 1
        return c

    def to_dict(self):
        return {"checks": self.rows, "counts": self.counts()}


# ---------------------------------------------------------------------------
# Individual checks. Each takes a Report and must not raise.
# ---------------------------------------------------------------------------
def check_python(rep):
    v = sys.version_info
    label = f"python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        rep.ok(label)
    else:
        rep.fail(label, "3.11 or newer is required",
                 "install a newer python3")


def check_venv(rep):
    venv = os.path.join(paths.APP_ROOT, "venv")
    venv_py = os.path.join(venv, "bin", "python")

    if os.path.exists(venv_py):
        rep.ok("virtual environment", venv)
        return

    # Development runs from the repo without an install venv; that is fine as
    # long as the current interpreter has what is needed, which the dependency
    # check below establishes.
    if os.path.abspath(paths.APP_ROOT) == os.path.abspath(paths.REPO_ROOT) and \
       not os.path.isdir(os.path.join(paths.REPO_ROOT, "venv")):
        rep.warn("no install venv", "running from the source tree",
                 "this is normal in development; an installed app has one")
    else:
        rep.fail("virtual environment missing", venv,
                 "reinstall: curl -fsSL <install-url> | bash")


def check_dependencies(rep):
    missing = []
    for mod in ("requests",):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        rep.fail("dependencies", f"missing: {', '.join(missing)}",
                 f"{sys.executable} -m pip install -r requirements.lock")
    else:
        rep.ok("dependencies", "requests")


def check_layout(rep):
    for name, path in (("app", paths.APP_ROOT), ("data", paths.DATA_DIR)):
        if os.path.isdir(path):
            if os.access(path, os.W_OK):
                rep.ok(f"{name} directory writable", path)
            else:
                rep.fail(f"{name} directory not writable", path,
                         f"chmod u+w {path}")
        else:
            if name == "data":
                rep.warn("data directory missing", path,
                         "it is created on first run")
            else:
                rep.fail(f"{name} directory missing", path,
                         "reinstall the app")


def check_config(rep):
    try:
        import config as cfgmod
    except Exception as exc:
        rep.fail("config module", str(exc), "the app tree may be incomplete")
        return

    path = cfgmod.config_path()
    if not os.path.exists(path):
        rep.warn("config.yaml missing", path,
                 "run: hunter setup")
        return

    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        rep.warn("config.yaml is readable by others", f"mode {oct(mode)}",
                 f"chmod 600 {path}")
    else:
        rep.ok("config.yaml permissions", oct(mode))

    cfg = cfgmod.load(path)
    problems = cfgmod.validate(cfg)
    if problems:
        rep.fail("config.yaml has invalid values", problems[0],
                 f"fix {path}, or re-run: hunter setup")
    else:
        rep.ok("config.yaml is valid")


def check_credentials(rep):
    try:
        import onboarding
    except Exception:
        return
    if onboarding.has_credentials():
        rep.ok("API credentials present")
    else:
        rep.warn("no API credentials", "not set up yet",
                 "run: hunter setup")


def check_crontab(rep):
    try:
        import crontab_manager as cm
    except Exception as exc:
        rep.warn("crontab check skipped", str(exc))
        return

    try:
        text = cm.read_crontab()
    except Exception as exc:
        # The most common real cause is that crontab is not installed at all.
        if "not installed" in str(exc):
            rep.fail("crontab is not installed", str(exc),
                     "sudo apt install cron")
        else:
            rep.fail("could not read the crontab", str(exc),
                     "check permissions: crontab -l")
        return

    before, block, after = cm.split_block(text)
    if not block:
        rep.fail("no the-hunter block in the crontab",
                 "scheduled jobs are not registered",
                 "hunter cron install")
        return

    states = cm.parse_jobs(block)
    expected = set(cm.JOB_ORDER)
    got = set(states)

    if got != expected:
        missing = expected - got
        extra = got - expected
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"unknown {', '.join(sorted(extra))}")
        rep.fail("crontab block is incomplete", "; ".join(detail),
                 "hunter cron install")
    else:
        rep.ok("crontab block intact", f"{len(states)} jobs")

    # An unterminated block means the markers were edited by hand.
    if cm.BEGIN_MARK in text and cm.END_MARK not in text:
        rep.fail("crontab block is unterminated",
                 "BEGIN marker present without END",
                 "hunter cron remove, then: hunter cron install")

    enabled = [n for n, on in states.items() if on]
    if enabled:
        rep.ok("scheduled jobs enabled", ", ".join(enabled))
    else:
        rep.warn("all scheduled jobs are disabled",
                 "nothing will fetch data or hunt",
                 "hunter cron enable market")


def check_data(rep):
    ohlcv = paths.OHLCV_DIR
    if not os.path.isdir(ohlcv):
        rep.warn("no market data", ohlcv, "run the fetch from Settings")
        return

    files = [f for f in os.listdir(ohlcv) if f.endswith(".csv")]
    if not files:
        rep.warn("market data directory is empty", ohlcv,
                 "run the fetch from Settings")
        return

    newest = 0
    for f in files:
        try:
            newest = max(newest, os.path.getmtime(os.path.join(ohlcv, f)))
        except OSError:
            pass

    age_h = (time.time() - newest) / 3600 if newest else 999
    detail = f"{len(files)} pairs, last write {age_h:.1f}h ago"

    if age_h > 24:
        rep.warn("market data is stale", detail,
                 "hunter cron enable ohlcv")
    else:
        rep.ok("market data present", detail)


def check_model(rep):
    model = os.path.join(paths.MODEL_DIR, "model.joblib")
    preds = os.path.join(paths.MODEL_DIR, "predictions.json")

    if not os.path.exists(model):
        rep.warn("model.joblib missing", "the ML veto layer is inactive",
                 "re-download it, or let the app run without a veto")
        return
    rep.ok("model.joblib present")

    if not os.path.exists(preds):
        rep.warn("no predictions yet", preds,
                 "the dashboard refreshes them automatically")
        return

    try:
        d = json.load(open(preds))
        age = time.time() - d.get("generated_at", 0)
        n = len(d.get("predictions", {}))
        if age > 3600:
            rep.warn("predictions are stale", f"{n} pairs, {age/60:.0f}m old",
                     "restart the dashboard, or check the predictor runs")
        else:
            rep.ok("predictions fresh", f"{n} pairs, {age/60:.0f}m old")
    except Exception as exc:
        rep.fail("predictions.json is unreadable", str(exc),
                 "delete it; the dashboard regenerates it")


def _dashsvc_diagnose():
    """dashsvc.diagnose(), never raising. {} if the module is unavailable."""
    try:
        import dashsvc
        return dashsvc.diagnose()
    except Exception:
        return {}


def dashboard_not_running_reason():
    d = _dashsvc_diagnose()
    return d.get("reason") or "the dashboard service is not running"


def dashboard_fix():
    """A command that will actually work in the current state.

    Three distinct cases, which the doctor used to collapse into one wrong
    answer (`systemctl --user restart hunter-dashboard`, which fails with
    "Unit not found" when no unit exists):
      no unit        -> hunter dashboard install
      no user manager -> sudo loginctl enable-linger $USER
      unit stopped   -> systemctl --user status hunter-dashboard
    """
    d = _dashsvc_diagnose()
    return d.get("fix") or "hunter dashboard status"


def check_port(rep, port=None):
    try:
        import config as cfgmod
        port = port or int(cfgmod.get(cfgmod.current(), "server.port", 8082))
    except Exception:
        port = port or 8082

    # Is something listening?
    s = socket.socket()
    s.settimeout(1.5)
    listening = False
    try:
        s.connect(("127.0.0.1", port))
        listening = True
    except OSError:
        pass
    finally:
        s.close()

    if not listening:
        d = _dashsvc_diagnose()
        if d.get("installed") and d.get("bus") is False:
            # A distinct, named fault: the unit exists and will never start until
            # linger is enabled. Worth its own line because the symptom (nothing
            # listening) looks identical to a stopped service.
            rep.fail(f"nothing listening on port {port}",
                     "the unit exists but this user has no systemd manager, so it cannot start",
                     "sudo loginctl enable-linger $USER")
        else:
            rep.warn(f"nothing listening on port {port}", dashboard_not_running_reason(),
                     dashboard_fix())
        return

    # Listening, but is it healthy?
    try:
        import urllib.request
        t0 = time.time()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/control",
                                   timeout=3) as r:
            r.read(1)
        ms = (time.time() - t0) * 1000
        rep.ok(f"dashboard responding on {port}", f"{ms:.0f}ms")
    except Exception as exc:
        rep.fail(f"port {port} is open but the dashboard is not healthy",
                 str(exc),
                 dashboard_fix())


def check_disk(rep):
    try:
        usage = shutil.disk_usage(paths.DATA_DIR if os.path.isdir(paths.DATA_DIR)
                                  else paths.APP_ROOT)
        free_gb = usage.free / 1e9
        if free_gb < 0.5:
            rep.fail("disk almost full", f"{free_gb:.1f} GB free",
                     "free space, or lower data.retention_years")
        elif free_gb < 2:
            rep.warn("disk space is low", f"{free_gb:.1f} GB free",
                     "consider pruning or freeing space")
        else:
            rep.ok("disk space", f"{free_gb:.1f} GB free")
    except Exception as exc:
        rep.warn("could not read disk usage", str(exc))


CHECKS = [
    ("python", check_python),
    ("venv", check_venv),
    ("dependencies", check_dependencies),
    ("layout", check_layout),
    ("config", check_config),
    ("credentials", check_credentials),
    ("crontab", check_crontab),
    ("data", check_data),
    ("model", check_model),
    ("port", check_port),
    ("disk", check_disk),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_checks(only=None):
    """Run every check. A check that raises is reported as a failed check, not
    as a crashed doctor."""
    rep = Report()
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        try:
            fn(rep)
        except Exception as exc:
            rep.fail(f"{name} check crashed", f"{type(exc).__name__}: {exc}",
                     "this is a bug in doctor; report it")
    return rep


def render(rep, as_json=False):
    if as_json:
        return json.dumps(rep.to_dict(), indent=2)

    L = [term.heading("The Hunter — diagnostics")]
    L.append("")

    tags = {
        PASS: term.green("[PASS]"),
        WARN: term.yellow("[WARN]"),
        FAIL: term.red("[FAIL]"),
    }

    # failures first: a diagnostic should lead with what is broken
    order = {FAIL: 0, WARN: 1, PASS: 2}
    for row in sorted(rep.rows, key=lambda r: order[r["status"]]):
        line = f"  {tags[row['status']]} {row['label']}"
        if row["detail"]:
            line += f"  {term.grey(row['detail'])}"
        L.append(line)
        if row["fix"]:
            L.append(f"         {term.grey('Fix: ' + row['fix'])}")

    c = rep.counts()
    L.append("")
    L.append(term.rule())
    summary = (f"  {c[PASS]} passed · {c[WARN]} warning · {c[FAIL]} failed")
    if c[FAIL]:
        summary = term.red(summary)
    elif c[WARN]:
        summary = term.yellow(summary)
    else:
        summary = term.green(summary)
    L.append(summary)
    L.append(term.rule())
    return "\n".join(L)


def main(argv=None, as_json=False):
    rep = run_checks()
    print(render(rep, as_json=as_json))
    # exit code doubles as a usable signal for scripts
    return 1 if rep.counts()[FAIL] else 0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {str(detail)[:250]}")

    print("=== doctor.py self-check ===")
    print()

    # ---- 1. every check is callable and reports something ----
    rep = run_checks()
    fired = {r["label"].split()[0].lower() for r in rep.rows}
    check("all checks ran and reported", len(rep.rows) >= len(CHECKS) - 1,
          f"{len(rep.rows)} rows from {len(CHECKS)} checks")

    # ---- 2. a check that raises does not kill the run ----
    def boom(rep_):
        raise RuntimeError("deliberate explosion")

    saved = CHECKS[:]
    try:
        CHECKS.append(("boom", boom))
        rep2 = run_checks()
        crashed = [r for r in rep2.rows if "crashed" in r["label"]]
        check("a raising check is reported, not fatal", len(crashed) == 1,
              [r["label"] for r in rep2.rows][-3:])
        check("the crash is attributed to the right check",
              crashed and "boom" in crashed[0]["label"], crashed)
    finally:
        CHECKS[:] = saved

    # ---- 3. every FAIL carries a fix ----
    rep3 = run_checks()
    fails = [r for r in rep3.rows if r["status"] == FAIL]
    check("every failure states a fix",
          all(r["fix"] for r in fails),
          [r["label"] for r in fails if not r["fix"]])

    # ---- 4. counts are consistent with the rows ----
    c = rep3.counts()
    check("counts add up to the number of checks",
          sum(c.values()) == len(rep3.rows),
          f"{c} vs {len(rep3.rows)}")

    # ---- 5. failures sort first in the rendered output ----
    r = Report()
    r.ok("all good")
    r.fail("something broken", "detail", "fix it")
    r.warn("something odd")
    text = render(r)
    pos_fail = text.find("something broken")
    pos_pass = text.find("all good")
    check("failures are listed before passes", 0 <= pos_fail < pos_pass,
          f"fail@{pos_fail} pass@{pos_pass}")

    # ---- 6. the fix hint appears on failure ----
    check("the fix line is rendered", "Fix: fix it" in text, text[-300:])

    # ---- 7. json output is valid and complete ----
    d = json.loads(render(r, as_json=True))
    check("--json is valid JSON", isinstance(d, dict))
    check("--json carries every check", len(d["checks"]) == 3, d["checks"])
    check("--json carries the counts", d["counts"][FAIL] == 1, d["counts"])

    # ---- 8. a healthy report says so ----
    healthy = Report()
    healthy.ok("a")
    healthy.ok("b")
    txt = render(healthy)
    check("a clean report has zero failures", "0 failed" in txt, txt[-200:])

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(_self_check())
    sys.exit(main(as_json="--json" in sys.argv))
