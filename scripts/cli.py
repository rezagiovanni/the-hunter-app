#!/usr/bin/env python3
"""
cli.py — the `hunter` command.

This is the app's front door and, once installed, the thing cron calls. It is
deliberately thin: it reads real state and reports it. It does not trade, does
not evaluate exits, and changes nothing except through the explicit control
subcommands (`start`, `stop`), which delegate to the engine's own control plane.

Design notes
------------
**Every command has `--json`.** The dashboard and any future script consume the
same data the human view shows, so there is one code path producing facts and
two renderings of them.

**Colour never carries meaning alone.** `term.py` supplies the vocabulary; every
status keeps its word or symbol. `NO_COLOR` or a pipe degrades to plain text.

**`_job <name>` is the scheduled entry point.** Cron calls one job per line; the
argument is validated against a fixed list so a typo cannot silently do nothing.

**Nothing here assumes a working install.** A missing state file, missing
or a stopped dashboard are all reported as facts, not crashes. This
is what `hunter status` is for: telling you what is wrong.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import term  # noqa: E402

try:
    import paths  # noqa: E402
    import config as cfgmod  # noqa: E402
except Exception as exc:                      # pragma: no cover - defensive
    print(f"hunter: cannot load its own modules: {exc}", file=sys.stderr)
    raise SystemExit(2)

VERSION = "0.1.0"

# Jobs cron may call. Kept here, not derived from the filesystem, so a job that
# is removed from the code cannot be invoked by a stale crontab line.
JOBS = {
    "market": {
        "script": "fetch_market.py",
        "desc": "fetch the top-20 pairs and their order books",
    },
    "ohlcv": {
        "script": "fetch_ohlcv.py",
        "desc": "append the latest 15m candles",
    },
    "backfill-stale": {
        "script": "backfill_stale.py",
        "desc": "fill gaps in recent history",
    },
    "backfill-deep": {
        "script": "backfill_deep.py",
        "desc": "fetch two years of history for new pairs",
    },
    "prune": {
        "script": "prune_data.py",
        "desc": "drop bars older than the retention floor",
    },
    "scanning": {
        "script": "run_hunter.py",
        "desc": "one hunting tick (this is the one that trades)",
    },
}

WIB_OFFSET = 7 * 3600


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def wib(ts):
    """Format a unix timestamp as WIB, which is what Reza reads."""
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts + WIB_OFFSET))


def ago(ts):
    """Human age of a timestamp."""
    if not ts:
        return "never"
    d = max(0, time.time() - ts)
    if d < 90:
        return f"{int(d)}s ago"
    if d < 5400:
        return f"{int(d / 60)}m ago"
    if d < 129600:
        return f"{d / 3600:.1f}h ago"
    return f"{d / 86400:.1f}d ago"


def rupiah(v):
    """IDR with thousands separators and an explicit sign when negative."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    sign = "-" if v < 0 else ""
    return f"{sign}IDR {abs(v):,.0f}"



def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# State collection — one place, used by both the human and JSON views
# ---------------------------------------------------------------------------

def collect_status():
    """Everything `hunter status` reports, as plain data."""
    st = read_json(paths.STATE_FILE, {}) or {}
    ctl = read_json(paths.CONTROL_FILE, {}) or {}

    trades = st.get("trade_log", []) or []
    # Tolerate malformed rows. A hand-edited or truncated state file should not
    # take down `hunter status` with an AttributeError - status is the command a
    # user runs to find out whether anything is wrong, so it must survive bad
    # input and report on it rather than crash. Non-dict entries are ignored.
    trades = [t for t in trades if isinstance(t, dict)]
    wins = sum(1 for t in trades if (t.get("net_pnl_idr") or 0) > 0)
    net = sum((t.get("net_pnl_idr") or 0) for t in trades)

    # data footprint
    pairs = read_json(paths.PAIRS_FILE, []) or []
    ohlcv_files = list(Path(paths.OHLCV_DIR).glob("*.csv")) if os.path.isdir(paths.OHLCV_DIR) else []
    ohlcv_bytes = 0
    for f in ohlcv_files:
        try:
            ohlcv_bytes += f.stat().st_size
        except OSError:
            pass
    newest_bar = None
    for f in ohlcv_files:
        try:
            newest_bar = max(newest_bar or 0, f.stat().st_mtime)
        except OSError:
            pass

    # dashboard
    dash = {"responding": False, "port": None, "latency_ms": None}
    try:
        import urllib.request
        port = int(cfgmod.get(cfgmod.current(), "server.port", 8082))
        dash["port"] = port
        t0 = time.time()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/control", timeout=1.5) as r:
            r.read(1)
        dash["responding"] = True
        dash["latency_ms"] = round((time.time() - t0) * 1000)
    except Exception:
        pass

    cfg = cfgmod.current()

    return {
        "version": VERSION,
        "mode": ctl.get("mode", "UNKNOWN"),
        "framework": ctl.get("framework", "?"),
        "phase": st.get("phase", "UNKNOWN"),
        "balance_idr": st.get("balance_idr"),
        "position": st.get("position"),
        "order": st.get("order"),
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "net_pnl_idr": round(net, 2),
        "hunt_counter": st.get("hunt_counter", 0),
        "data": {
            "pairs": len(pairs),
            "ohlcv_files": len(ohlcv_files),
            "ohlcv_bytes": ohlcv_bytes,
            "newest_bar_mtime": newest_bar,
        },
        "dashboard": dash,
        "dashboard_service": dashboard_service_status(),
        "config": {
            "sizing_fraction": cfgmod.get(cfg, "trading.sizing_fraction"),
            "cooldown_hours": cfgmod.get(cfg, "trading.cooldown_hours"),
            "framework_target": (cfgmod.framework_params(cfg, ctl.get("framework", "A"))
                                 .get("target_pct")),
            "framework_stop": (cfgmod.framework_params(cfg, ctl.get("framework", "A"))
                               .get("stop_pct")),
        },
        "paths": paths.describe(),
    }


def collect_where():
    d = paths.describe()
    return {
        "app": d["app_root"],
        "data": d["data_dir"],
        "market": d["ohlcv"],          # the OHLCV store, not its parent
        "config": cfgmod.config_path(),
        "logs": d["logs"],
        "model": d["model"],
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_where(data, as_json=False):
    if as_json:
        return json.dumps(data, indent=2)
    lines = [term.heading(f"THE HUNTER  {term.grey('paths')}")]
    labels = [("app", "App"), ("data", "Data"), ("market", "Market"),
              ("model", "Model"), ("config", "Config"), ("logs", "Logs")]
    for key, label in labels:
        v = data[key]
        exists = os.path.exists(v)
        lw = 8
        path = v if exists else term.yellow(v + "  (missing)")
        lines.append(f"  {term.pad(label, lw)} {path}")
    return "\n".join(lines)


def render_status(data, as_json=False):
    if as_json:
        return json.dumps(data, indent=2)

    L = []
    L.append(term.heading(
        f"THE HUNTER {term.grey('v' + data['version'])}"))
    L.append("")

    # --- control ---
    mode = data["mode"]
    mode_txt = term.badge(mode, "running" if mode == "RUNNING" else
                          ("stopped" if mode == "STOPPED" else "warn"))
    L.append(f"  {term.pad('Hunting', 18)}: {mode_txt}   "
             f"{term.grey('framework')} {term.bold(data['framework'])}")

    pos = data.get("position")
    if pos:
        L.append(f"  {term.pad('Position', 18)}: "
                 f"{term.bold(pos.get('symbol', '?'))}  "
                 f"{term.grey('entry')} {pos.get('entry_price')}")
    else:
        L.append(f"  {term.pad('Position', 18)}: {term.grey('none')}")

    L.append(f"  {term.pad('Phase', 18)}: {data['phase']}")

    bal = data.get("balance_idr")
    L.append(f"  {term.pad('Balance', 18)}: {term.bold(rupiah(bal)) if bal is not None else '-'}")

    t = data
    pnl = t["net_pnl_idr"]
    pnl_txt = term.green(f"+{rupiah(pnl)}") if pnl > 0 else (
        term.red(rupiah(pnl)) if pnl < 0 else term.grey(rupiah(pnl)))
    record = term.grey("({}W / {}L)".format(t["wins"], t["losses"]))
    L.append(f"  {term.pad('Trades', 18)}: {t['trades']}  {record}  net {pnl_txt}")

    cfg = data["config"]
    fw = data["framework"]
    target_txt = term.cyan("+{:.1f}%".format(cfg["framework_target"] * 100))
    stop_txt = term.cyan("-{:.1f}%".format(cfg["framework_stop"] * 100))
    fw_txt = term.grey("(framework {})".format(fw))
    L.append(f"  {term.pad('Exits', 18)}: target {target_txt}  stop {stop_txt}  {fw_txt}")

    # --- data ---
    L.append(term.section("Data"))
    d = data["data"]
    L.append(f"  {term.pad('Pairs', 18)}: {d['pairs']}")
    size_mb = d["ohlcv_bytes"] / 1e6
    L.append(f"  {term.pad('OHLCV', 18)}: {d['ohlcv_files']} files, {size_mb:,.1f} MB")
    L.append(f"  {term.pad('Last bar', 18)}: {ago(d['newest_bar_mtime'])}")

    # --- dashboard ---
    L.append(term.section("Dashboard"))
    dh = data["dashboard"]
    svc = data.get("dashboard_service") or {}
    if dh["responding"]:
        where = term.grey("127.0.0.1:{}  {}ms".format(dh["port"], dh["latency_ms"]))
        L.append(f"  {term.green('responding')}  {where}")
    else:
        port = dh["port"] or "?"
        L.append(f"  {term.yellow('[WARN]')} not responding on 127.0.0.1:{port}")

        # The fix depends on whether the service actually exists. Saying
        # "systemctl --user restart hunter-dashboard" unconditionally is what
        # produced "Unit hunter-dashboard.service not found" on a fresh install.
        if svc.get("installed"):
            L.append(f"         {term.grey('Fix: systemctl --user restart hunter-dashboard')}")
        else:
            L.append(f"         {term.grey('The service is not installed. To create and start it:')}")
            L.append(f"         {term.grey('  hunter dashboard install')}")

    L.append("")
    L.append(term.rule())
    return "\n".join(L)


def dashboard_service_status():
    """Whether the systemd unit exists and is running.

    Separate from `dashboard.responding` on purpose: a unit can exist but be
    stopped, or not exist at all, and the correct advice differs.
    """
    try:
        import dashsvc
        return dashsvc.status()
    except Exception:
        return {"installed": False, "active": False, "unit": "hunter-dashboard.service",
                "port": 8082, "path": ""}


def crontab_job_states():
    """(job, enabled) from the user's crontab, or None if crontab is unreadable.

    Reads the managed block only. A commented line is a disabled job — that is
    how the installer registers them, so "disabled" means the process never
    starts at all.
    """
    try:
        cr_cmd = os.environ.get("HUNTER_CRONTAB") or "crontab"
        r = subprocess.run([cr_cmd, "-l"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if r.returncode != 0:
        return [] if "no crontab" in (r.stderr or "").lower() else None

    inside = False
    seen = {}
    for line in r.stdout.splitlines():
        if line.strip().startswith("# BEGIN the-hunter"):
            inside = True
            continue
        if line.strip().startswith("# END the-hunter"):
            inside = False
            continue
        if not inside:
            continue
        s = line.strip()
        enabled = not s.startswith("#")
        body = s.lstrip("#").strip()
        if "_job" not in body:
            continue
        try:
            job = body.split("_job", 1)[1].split()[0]
        except IndexError:
            continue
        seen[job] = enabled
    return sorted(seen.items())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_where(args):
    data = collect_where()
    print(render_where(data, args.json))
    return 0


def cmd_status(args):
    data = collect_status()
    print(render_status(data, args.json))
    return 0


def cmd_start(args):
    import control
    control.set_mode(control.MODE_RUNNING, changed_by="cli")
    if args.json:
        print(json.dumps({"mode": control.get_mode()}, indent=2))
    else:
        print(f"  {term.green('[OK]')} hunting {term.bold('RUNNING')}")
        print(f"       {term.grey('the next tick will hunt within 60s')}")
    return 0


def cmd_stop(args):
    import control
    st = read_json(paths.STATE_FILE, {}) or {}
    pos = st.get("position")
    if pos and not args.yes:
        print(f"  {term.yellow('[WARN]')} a position is open: "
              f"{term.bold(pos.get('symbol'))} "
              f"({term.grey('entry')} {pos.get('entry_price')})")
        print(f"         {term.grey('STOP will market-sell it now.')}")
        print()
        try:
            answer = input("  Continue? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print(f"  {term.grey('cancelled')}")
            return 1
    if args.json:
        print(json.dumps({"mode": control.MODE_STOPPED}, indent=2))
    else:
        print(f"  {term.green('[OK]')} hunting {term.bold('STOPPED')}")
    return 0


def cmd_job(args):
    """The scheduled entry point. Cron calls `hunter _job <name>`.

    Validates against the fixed JOBS list, so a stale crontab line naming a
    removed job fails loudly instead of doing nothing.
    """
    name = args.name
    if name not in JOBS:
        print(f"hunter _job: unknown job '{name}'", file=sys.stderr)
        print(f"  known jobs: {', '.join(sorted(JOBS))}", file=sys.stderr)
        return 2

    script = HERE / JOBS[name]["script"]
    if not script.exists():
        print(f"hunter _job {name}: missing {script}", file=sys.stderr)
        return 2

    # Use the interpreter that can actually run the job, and run it unbuffered.
    #
    # Two bugs lived in the one line this replaces:
    #   [sys.executable, script]
    # 1. sys.executable is whatever python started the CLI. Under cron that is
    #    usually fine, but when `hunter _job` is invoked from the dashboard's
    #    venv it has no `requests`, and the job dies instantly. This is the
    #    THIRD place the same wrong-interpreter assumption appeared (dashboard's
    #    _resolve_engine_python and setup_state.engine_python were the others),
    #    so it now goes through the one shared resolver.
    # 2. No `-u`. Cron redirects stdout to a log file, and Python block-buffers
    #    a redirected stream, so a job that runs for 40 minutes wrote NOTHING to
    #    its log until it exited. A running job looked dead.
    import setup_state
    py = setup_state.engine_python()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run([py, "-u", str(script)], cwd=str(HERE.parent), env=env)
    return r.returncode


def cmd_cron(args):
    """Install, remove, list, enable and disable the scheduled jobs."""
    import crontab_manager as cm

    action = args.action

    if action == "status" or action == "list":
        states = cm.job_states()
        if args.json:
            print(json.dumps({"installed": bool(states), "jobs": states}, indent=2))
            return 0
        if not states:
            print(f"  {term.yellow('[WARN]')} no the-hunter block in the crontab")
            print(f"       {term.grey('Fix: hunter cron install')}")
            return 0
        L = [term.heading("SCHEDULED JOBS")]
        L.append("")
        for name, sched, desc in cm.JOBS:
            on = states.get(name, False)
            tag = term.badge("ON", "on") if on else term.badge("OFF", "off")
            # pad() measures display width, and the coloured tag carries escape
            # codes, so the tag must be padded through pad() too - an f-string
            # :<22 would count the escapes and shear the column.
            L.append(f"  {term.pad('thehunter_' + name, 26)}"
                     f"{term.pad(tag, 6)}"
                     f"{term.pad(term.grey(sched), 21)}"
                     f"{term.grey(desc)}")
        print("\n".join(L))
        return 0

    if action == "install":
        result = cm.install(dry_run=args.dry_run)
        if args.json:
            print(json.dumps({"changed": result["changed"],
                              "backup": result["backup"],
                              "jobs": cm.job_states()}, indent=2))
            return 0
        if not result["changed"]:
            print(f"  {term.grey('already installed, nothing changed')}")
            return 0
        print(f"  {term.green('[OK]')} installed 6 jobs, "
              f"{term.bold('all disabled')}")
        if result["backup"]:
            print(f"       {term.grey('previous crontab saved to ' + result['backup'])}")
        print(f"       {term.grey('enable one with: hunter cron enable market')}")
        return 0

    if action == "remove":
        result = cm.remove(dry_run=args.dry_run)
        if args.json:
            print(json.dumps({"changed": result["changed"]}, indent=2))
            return 0
        if not result["changed"]:
            print(f"  {term.grey('no block to remove')}")
            return 0
        print(f"  {term.green('[OK]')} the-hunter block removed; "
              f"{term.grey('your other crontab lines are untouched')}")
        return 0

    if action in ("enable", "disable"):
        if not args.job:
            print(f"  {term.red('[FAIL]')} which job? e.g. hunter cron {action} market",
                  file=sys.stderr)
            return 2
        try:
            states = cm.set_job(args.job, action == "enable", dry_run=args.dry_run)
        except cm.CrontabError as exc:
            print(f"  {term.red('[FAIL]')} {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps({"jobs": states}, indent=2))
            return 0
        word = "enabled" if action == "enable" else "disabled"
        if args.dry_run:
            # A dry run writes nothing, so it must not claim to have done so.
            print(f"  {term.grey('[DRY RUN]')} would have {word} "
                  f"thehunter_{args.job}; crontab unchanged")
            return 0
        print(f"  {term.green('[OK]')} thehunter_{args.job} {term.bold(word)}")
        return 0

    print(f"  unknown cron action '{action}'", file=sys.stderr)
    return 2


def cmd_version(args):
    if args.json:
        print(json.dumps({"version": VERSION}, indent=2))
    else:
        print(f"hunter {VERSION}")
    return 0


def cmd_setup(args):
    """First-time setup: collect and validate the API key."""
    import onboarding
    return onboarding.run(force=getattr(args, "force", False))


def cmd_doctor(args):
    """Diagnose the install and name any specific fault."""
    import doctor
    return doctor.main(as_json=args.json)


def cmd_upgrade(args):
    """Replace the app code; data, config and jobs are preserved."""
    import lifecycle
    import term as t

    res = lifecycle.upgrade(source=args.source, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1

    if args.dry_run:
        print(f"  {t.grey('[DRY RUN]')} nothing would be changed")
        print(f"       would replace: {', '.join(res['replaced']) or 'app/'}")
        print(f"       would preserve: {', '.join(res['preserved']) or 'nothing'}")
        return 0

    if res.get("source"):
        print(f"  {t.grey('source: ' + res['source'])}")

    if not res.get("ok"):
        print(f"  {t.red('[FAIL]')} upgrade did not complete", file=sys.stderr)
        return 1

    print(f"  {t.green('[ok]')} replaced {', '.join(res['replaced'])}")
    print(f"       preserved: {', '.join(res['preserved']) or 'nothing'}")
    if res.get("backup"):
        print(f"       previous app/ saved to {t.grey(res['backup'])}")
    if res.get("jobs_after"):
        on = [n for n, v in res["jobs_after"].items() if v]
        print(f"       jobs: {len(res['jobs_after'])} present, "
              f"{len(on)} enabled")

    # Say what happened to the dashboard. Its absence here is why an upgrade
    # could land correct code that the user never saw.
    _dr = res.get("dashboard_restarted")
    if _dr is True:
        print(f"       dashboard: restarted (now serving the new code)")
    elif _dr is False:
        print(f"       {t.yellow('[WARN]')} dashboard restart failed - it is "
              f"still serving the OLD code")
        if res.get("dashboard_restart_error"):
            print(f"       {t.grey(str(res['dashboard_restart_error'])[:140])}")
        print(f"       fix: {t.grey('hunter dashboard restart')}")
    elif _dr is None:
        print(f"       dashboard: not installed, nothing to restart")

    # Report the dependency outcome only when the lock actually changed, so the
    # normal case stays quiet. Silence here would hide a stale venv that will
    # fail at runtime with ModuleNotFoundError.
    if res.get("lock_changed"):
        if res.get("deps_updated"):
            print(f"       dependencies: updated from the new "
                  f"{t.grey('requirements.lock')}")
        else:
            print(f"       {t.yellow('[WARN]')} requirements.lock changed but "
                  f"the install failed")
            print(f"       {t.grey(str(res.get('deps_detail', ''))[:160])}")
            print(f"       fix: {t.grey('hunter doctor')}")
    return 0


def cmd_uninstall(args):
    """Remove the program. Trade history is kept unless --delete-data."""
    import lifecycle
    import term as t

    res = lifecycle.uninstall(dry_run=args.dry_run,
                              assume_yes=getattr(args, "yes", False),
                              delete_data=getattr(args, "delete_data", False))

    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1

    # An open position is never liquidated silently.
    if res.get("needs_confirmation"):
        pos = res["position"]
        print(f"  {t.yellow('[WARN]')} a position is open: "
              f"{t.bold(pos.get('symbol'))} "
              f"({t.grey('entry')} {pos.get('entry_price')})")
        print(f"         {t.grey('Run `hunter stop` first, or pass --yes to')}")
        print(f"         {t.grey('remove anyway (the position is left open).')}")
        return 1

    if args.dry_run:
        print(f"  {t.grey('[DRY RUN]')} nothing would be changed")
        for label, _ in res["steps"]:
            print(f"       would {label}")
        print(f"       would keep: {', '.join(res['kept']) or 'nothing'}")
        return 0

    for label, outcome in res.get("done", []):
        mark = t.green("[ok]") if outcome is True else t.yellow("[warn]")
        print(f"  {mark} {label}" + ("" if outcome is True else f"  {outcome}"))

    if res.get("kept"):
        print()
        print(f"  {t.bold('Kept')}: {', '.join(res['kept'])}")
        print(f"  {t.grey('Reinstalling later will pick them up.')}")
    if res.get("deleted_data"):
        print(f"  {t.grey('OHLCV CSVs and hunter_state.json were deleted.')}")
    else:
        print(f"  {t.grey('Trade history, OHLCV data and config.yaml were kept.')}")
        print(f"  {t.grey('Run with --delete-data to remove them as well.')}")
    return 0 if res.get("ok") else 1


def cmd_dashboard(args):
    """Install, start, stop or inspect the dashboard service."""
    import dashsvc

    action = getattr(args, "action", None) or "status"

    if action == "status":
        st = dashsvc.status()
        if args.json:
            print(json.dumps(st, indent=2))
            return 0

        if st["active"]:
            tag = term.green("running")
        elif not st["installed"]:
            tag = term.yellow("not installed")
        else:
            tag = term.yellow("installed, not running")

        print(f"  {term.bold('hunter-dashboard.service')}  {tag}")
        print(f"  {term.grey('unit   ' + st['path'])}")
        print(f"  {term.grey('port   ' + str(st['port']))}")
        if st["installed"]:
            print(f"  {term.grey('enabled ' + ('yes' if st.get('enabled') else 'no'))}")
            print(f"  {term.grey('user systemd manager ' + ('reachable' if st.get('bus') else 'NOT reachable'))}")

        # Say WHY, and give a command that works in this exact state.
        if not st["active"]:
            print()
            print(f"  {term.yellow('reason')}  {st.get('reason','')}")
            if st.get("fix"):
                print(f"  {term.grey('Fix: ' + st['fix'])}")
            if st.get("bus") and st["installed"]:
                tail = dashsvc.journal_tail(8)
                if tail:
                    print()
                    for line in tail:
                        print(f"  {term.grey(line[:110])}")
        return 0

    if action == "restart":
        res = dashsvc.restart()
        if args.json:
            print(json.dumps(res, indent=2))
            return 0 if res.get("ok") else 1
        if res.get("ok"):
            print(f"  {term.green('[OK]')} dashboard restarted")
            print(f"  {term.grey('it now runs the code on disk')}")
            return 0
        print(f"  {term.red('[FAIL]')} {res.get('message','')}")
        return 1

    if action == "install":
        # The app root is the directory containing this script's parent.
        app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = dashsvc.install(app, data_dir=os.path.join(os.path.dirname(app), "data"))
        if args.json:
            print(json.dumps(res, indent=2))
            return 0
        if res.get("error"):
            print(f"  {term.red('[FAIL]')} {res['error']}")
            return 1
        if res["written"] and res["started"]:
            print(f"  {term.green('[OK]')} dashboard service installed and started")
            print(f"  {term.grey('open  http://127.0.0.1:' + str(dashsvc.PORT))}")
            return 0

        # Written but not started: report the real cause, not the flags.
        st = dashsvc.status()
        print(f"  {term.yellow('[WARN]')} the unit was written but did not start")
        print(f"  {term.grey('reason  ' + st.get('reason',''))}")
        if st.get("fix"):
            print(f"  {term.grey('Fix: ' + st['fix'])}")
        if st.get("bus"):
            tail = dashsvc.journal_tail(8)
            if tail:
                print()
                for line in tail:
                    print(f"  {term.grey(line[:110])}")
        else:
            print(f"  {term.grey('Note: this user has no running systemd manager.')}")
            print(f"  {term.grey('      The dashboard can also run in the foreground:')}")
            print(f"  {term.grey('        hunter dashboard run')}")
        return 1

    if action == "run":
        # Foreground fallback for machines without a usable user systemd manager.
        app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        py = os.path.join(app, "venv", "bin", "python")
        if not os.path.exists(py):
            py = sys.executable
        script = os.path.join(app, "scripts", "dashboard.py")
        print(f"  {term.grey('starting the dashboard in the foreground; Ctrl-C to stop')}")
        print(f"  {term.grey('open  http://127.0.0.1:' + str(dashsvc.PORT))}")
        return subprocess.call([py, script])

    if action == "remove":
        res = dashsvc.remove()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"  {term.green('[OK]')} dashboard service removed")
        return 0

    print(f"  unknown action: {action}", file=sys.stderr)
    return 2


COMMANDS = {
    "where": (cmd_where, "show where the app, data and logs live"),
    "dashboard": (cmd_dashboard, "install, start, stop or inspect the dashboard service"),
    "status": (cmd_status, "live state: hunting mode, balance, data, jobs, dashboard"),
    "start": (cmd_start, "start hunting (enables the scanning job)"),
    "stop": (cmd_stop, "stop hunting (market-sells an open position first)"),
    "setup": (cmd_setup, "enter or replace your Indodax API key"),
    "doctor": (cmd_doctor, "diagnose the install and name any fault"),
    "upgrade": (cmd_upgrade, "replace the app code; data and settings are kept"),
    "uninstall": (cmd_uninstall, "remove the program; trade history is kept"),
    "cron": (cmd_cron, "install, list, enable or disable the scheduled jobs"),
    "version": (cmd_version, "print the version"),
}


def build_parser():
    p = argparse.ArgumentParser(
        prog="hunter",
        description="The Hunter — an Indodax trading bot with a local dashboard.",
        epilog="Run 'hunter <command> --help' for details on a command.",
    )
    p.add_argument("--version", action="store_true", help="print the version and exit")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    for name, (fn, desc) in COMMANDS.items():
        sp = sub.add_parser(name, help=desc, description=desc)
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        if name == "stop":
            sp.add_argument("-y", "--yes", action="store_true",
                            help="do not ask before selling an open position")
        if name == "upgrade":
            sp.add_argument("--source", default=None,
                            help="release tarball path or URL "
                                 "(default: the published release for this version)")
            sp.add_argument("--dry-run", action="store_true",
                            help="show what would change without changing it")
        if name == "uninstall":
            sp.add_argument("-y", "--yes", action="store_true",
                            help="do not stop for an open position")
            sp.add_argument("--delete-data", action="store_true",
                            help="also delete trade history and market data")
            sp.add_argument("--dry-run", action="store_true",
                            help="show what would be removed")
        if name == "setup":
            sp.add_argument("--force", action="store_true",
                            help="replace credentials that are already saved")
        if name == "dashboard":
            dsub = sp.add_subparsers(dest="action", metavar="<action>")
            dsub.add_parser("status", help="is it installed, running, and why not?")
            dsub.add_parser("install", help="create, enable and start the service")
            dsub.add_parser("restart", help="restart it so it serves the code on disk")
            dsub.add_parser("run", help="run it in the foreground (no systemd needed)")
            dsub.add_parser("remove", help="stop, disable and delete the service")

        if name == "cron":
            sp.add_argument("action",
                            choices=["install", "remove", "status", "list",
                                     "enable", "disable"],
                            help="what to do with the scheduled jobs")
            sp.add_argument("job", nargs="?", help="job name, for enable/disable")
            sp.add_argument("--dry-run", action="store_true",
                            help="show what would change without writing")

    # The scheduled entry point cron calls. It is deliberately absent from
    # --help: users do not run it by hand, and argparse renders help=SUPPRESS
    # as the literal string "==SUPPRESS==" in the subcommand list, so it is
    # removed from the display after construction instead of suppressed.
    jp = sub.add_parser("_job", help="internal: run one scheduled job")
    jp.add_argument("name")
    sub._choices_actions = [a for a in sub._choices_actions if a.dest != "_job"]

    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        # Bare `hunter` is the first thing a new user runs, so a genuinely
        # fresh install gets onboarding rather than a status page about a bot
        # that cannot trade. "Fresh" means no credentials AND no data: a machine
        # that is already trading must never be dragged into the key prompt.
        import onboarding
        if onboarding.is_fresh_install():
            return onboarding.run()
        print(render_status(collect_status()))
        print()
        print(f"  {term.grey('hunter --help  for all commands')}")
        return 0

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    if getattr(args, "version", False) and not args.command:
        return cmd_version(argparse.Namespace(json=False))

    if args.command == "_job":
        return cmd_job(args)

    if args.command in COMMANDS:
        # ensure --json exists even if the subparser did not add it
        if not hasattr(args, "json"):
            args.json = False
        if not hasattr(args, "yes"):
            args.yes = False
        if not hasattr(args, "dry_run"):
            args.dry_run = False
        if not hasattr(args, "job"):
            args.job = None
        if not hasattr(args, "force"):
            args.force = False
        for attr in ("source", "delete_data"):
            if not hasattr(args, attr):
                setattr(args, attr, None if attr == "source" else False)
        return COMMANDS[args.command][0](args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
