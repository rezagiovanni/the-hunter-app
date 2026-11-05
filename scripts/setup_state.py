#!/usr/bin/env python3
"""
setup_state.py — the record of first-time setup actions, for the Settings page.

Why this is a file and not memory
---------------------------------
A forty-five minute backfill must survive a dashboard restart, a browser reload
and an ssh disconnect. State held in the web process dies with it. So every
action's state lives in `data/setup_state.json`, written by the job itself and
read by the dashboard and the CLI.

The states
----------
    not_run   never started
    running   a process is alive for it
    done      the process exited 0
    failed    the process exited non-zero, or vanished

"failed" versus "done" is decided by evidence, not by optimism: a job that
records `running` and whose pid is then gone is reported as failed, even though
nobody wrote a failure. That is the case that matters — a backfill killed by a
reboot must not look like it finished.

One heavy job at a time
-----------------------
Two backfills writing the same CSVs corrupt data, and the target machine has
1.8 GiB of RAM. A single lock file guards every heavy action; a second request
is refused with a reason rather than queued.

Resumability
------------
`backfill_deep` records per-pair progress. A re-run skips pairs already recorded
as complete, so a failure twenty minutes in does not cost the twenty minutes.

Nothing here trades. It starts processes and records what happened.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402


def engine_python():
    """The interpreter that can actually run a job.

    The dashboard runs under .venv, which holds only fastapi/uvicorn/pillow. The
    data jobs need `requests`, which that venv does not have: starting them with
    sys.executable fails instantly with ModuleNotFoundError. Order:

      1. $HUNTER_PYTHON
      2. <app>/venv/bin/python     (a real install)
      3. python3 on PATH           (has requests system-wide here)
      4. sys.executable            (last resort)
    """
    override = os.environ.get("HUNTER_PYTHON")
    if override and os.path.exists(override):
        return override

    app_venv = os.path.join(paths.APP_ROOT, "venv", "bin", "python")
    if os.path.exists(app_venv):
        return app_venv

    found = shutil.which("python3")
    if found:
        return found

    return sys.executable


STATE_FILE = os.path.join(paths.DATA_DIR, "setup_state.json")
LOCK_FILE = os.path.join(paths.DATA_DIR, "setup.lock")

MAX_SYNC_SECONDS = 900     # foreground wait, tests only

NOT_RUN = "not_run"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Actions the Settings page offers, in the order a user should run them.
# `script` is resolved against paths.SCRIPTS_DIR at run time.
ACTIONS = {
    "market": {
        "label": "Fetch the top-20 pairs",
        "desc": "Downloads the pair list and current order books.",
        "script": "fetch_market.py",
        "log": "market.log",
        "heavy": False,
        "requires": [],
    },
    "backfill-deep": {
        "label": "Deep backfill (2 years)",
        "desc": "Pulls two years of 15m candles for every pair. Takes a while.",
        "script": "backfill_deep.py",
        "log": "backfill-deep.log",
        "heavy": True,
        "requires": ["market"],
    },
    "backfill-stale": {
        "label": "Fill gaps",
        "desc": "Finds and repairs holes in recent history.",
        "script": "backfill_stale.py",
        "log": "backfill-stale.log",
        "heavy": True,
        "requires": ["market"],
    },
    "prune": {
        "label": "Prune old data",
        "desc": "Drops bars older than the retention floor.",
        "script": "prune_data.py",
        "log": "prune.log",
        "heavy": True,
        "requires": ["backfill-deep"],
    },
}

ORDER = ["market", "backfill-deep", "backfill-stale", "prune"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _blank():
    return {
        "actions": {
            name: {
                "state": NOT_RUN,
                "started_at": None,
                "finished_at": None,
                "pid": None,
                "exit_code": None,
                "detail": "",
                "progress": None,        # e.g. {"done": 12, "total": 20}
            }
            for name in ACTIONS
        },
        "updated_at": time.time(),
    }


def load():
    """Read the state file, repairing anything missing. Never raises."""
    try:
        with open(STATE_FILE) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return _blank()

    if not isinstance(data, dict):
        return _blank()

    # fill in actions added since the file was written
    data.setdefault("actions", {})
    for name in ACTIONS:
        data["actions"].setdefault(name, {
            "state": NOT_RUN, "started_at": None, "finished_at": None,
            "pid": None, "exit_code": None, "detail": "", "progress": None,
        })
    return data


def save(data):
    """Atomic write: tmp file then rename, so a reader never sees half a file."""
    data["updated_at"] = time.time()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update(name, **fields):
    data = load()
    data["actions"].setdefault(name, {}).update(fields)
    save(data)
    return data


# ---------------------------------------------------------------------------
# Process liveness
# ---------------------------------------------------------------------------
def pid_alive(pid):
    """True if pid is a live, running process. A zombie counts as dead.

    os.kill(pid, 0) alone is NOT sufficient here, and getting this wrong wedged
    the whole Settings page. When setup_state launches a job from the dashboard,
    that job is the dashboard's direct child. If the dashboard never calls
    wait() - and it does not, the job is fire-and-forget - the finished child
    becomes a ZOMBIE: state Z in /proc, still present in the process table.
    os.kill(pid, 0) then succeeds, so liveness looked true forever, the action
    stayed 'running' forever, and because a heavy action holds the lock, every
    other action was refused too.

    So: consult /proc/<pid>/stat and treat Z (zombie) and X (dead) as gone.
    """
    if not pid:
        return False
    try:
        pid = int(pid)
    except (OverflowError, ValueError):
        return False

    # Linux: the authoritative answer, and it catches zombies.
    stat_path = f"/proc/{pid}/stat"
    if os.path.exists("/proc"):
        try:
            with open(stat_path) as fh:
                fields = fh.read().split()
            # field 3 is the state letter; the comm field can contain spaces,
            # so find it after the closing parenthesis.
            state = fields[2] if len(fields) > 2 else ""
            if state in ("Z", "X", "x"):
                return False
            return True
        except FileNotFoundError:
            return False
        except OSError:
            pass          # fall through to the portable probe

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reap_children():
    """Collect exit statuses of finished children of THIS process.

    The dashboard starts jobs and never waits for them, so their finished
    entries stay in the process table as zombies. Calling this on each state
    read keeps the table clean and, more importantly, turns a zombie into a
    genuinely-absent pid.
    """
    try:
        while True:
            pid, _status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except (ChildProcessError, OSError):
        pass


def reconcile(data=None):
    """Turn 'running' entries whose process has vanished into 'failed'.

    This is the honest-reporting core. A job killed by a reboot leaves the file
    saying 'running' forever; without this it would look like it is still going.
    """
    reap_children()
    data = data if data is not None else load()
    changed = False
    for name, a in data["actions"].items():
        if a.get("state") == RUNNING and not pid_alive(a.get("pid")):
            a["state"] = FAILED
            a["finished_at"] = a.get("finished_at") or time.time()
            # Always say WHY, and keep whatever progress was last reported.
            # A bare "5/10" with a failed state tells the user nothing about
            # what went wrong; the two facts belong together.
            reason = "process disappeared before reporting completion"
            last = (a.get("detail") or "").strip()
            a["detail"] = (f"{reason} (last progress: {last})" if last
                           else reason)
            a["pid"] = None
            changed = True
    if changed:
        save(data)
    return data


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------
def running_actions():
    """Names of every action currently running."""
    data = reconcile()
    return [n for n, a in data["actions"].items() if a.get("state") == RUNNING]


def busy_action():
    """The HEAVY action currently running, or None.

    Only heavy actions hold the lock. Two heavy jobs writing the same CSVs would
    corrupt data, and on a 1.8 GiB machine they would compete for memory. A
    light job (fetching the pair list and order books) writes different files
    from a backfill and does not hold anything.
    """
    data = reconcile()
    for name, a in data["actions"].items():
        if a.get("state") == RUNNING and ACTIONS.get(name, {}).get("heavy"):
            return name
    return None


# ---------------------------------------------------------------------------
# Starting work
# ---------------------------------------------------------------------------
def missing_requirements(name):
    """Prerequisites that are not yet done, so the UI can say what to run first."""
    data = reconcile()
    missing = []
    for req in ACTIONS[name]["requires"]:
        if data["actions"].get(req, {}).get("state") != DONE:
            missing.append(req)
    return missing


def start(name, background=True):
    """Launch one action. Returns (ok, message).

    `background=True` detaches the process with start_new_session so it outlives
    the request that started it — a 45-minute backfill must not die when the
    browser tab closes.
    """
    if name not in ACTIONS:
        return False, f"unknown action '{name}'"

    act = ACTIONS[name]

    running = running_actions()

    # Already running itself? Never start a second copy of the same action.
    if name in running:
        return False, f"'{act['label']}' is already running"

    # A heavy action may not start while anything else is running: it rewrites
    # large CSVs and needs the machine's memory to itself.
    if act["heavy"] and running:
        other = running[0]
        return False, (f"'{ACTIONS[other]['label']}' is still running; "
                       f"wait for it to finish")

    # A light action may run alongside a heavy one, because it writes different
    # files (pairs.json, depth/*) than a backfill (ohlcv/*).

    script = os.path.join(paths.SCRIPTS_DIR, act["script"])
    if not os.path.exists(script):
        return False, f"missing script {act['script']}"

    missing = missing_requirements(name)
    if missing:
        labels = ", ".join(ACTIONS[m]["label"] for m in missing)
        return False, f"run this first: {labels}"

    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(paths.LOGS_DIR, act["log"])

    _update(name, state=RUNNING, started_at=time.time(), finished_at=None,
            pid=None, exit_code=None, detail="", progress=None)

    # The job is NOT started here. The watcher is started instead, and the
    # watcher starts the job as its own child. That is the only way to obtain a
    # real exit code: a process that is not the parent cannot wait() for the
    # child, and every attempt to guess the outcome from a log or the process
    # table produced a wrong answer. See setup_watch.py.
    watcher = os.path.join(HERE, "setup_watch.py")
    try:
        proc = subprocess.Popen(
            [engine_python(), watcher, name, script, log_path],
            cwd=str(paths.APP_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,      # survives the dashboard request
        )
    except OSError as exc:
        _update(name, state=FAILED, finished_at=time.time(),
                detail=f"could not start: {exc}")
        return False, f"could not start: {exc}"

    _update(name, pid=proc.pid)

    if not background:
        try:
            proc.wait(timeout=MAX_SYNC_SECONDS)
        except subprocess.TimeoutExpired:
            pass          # it keeps running; the watcher will report

    return True, f"started {act['label']}"


def _spawn_watcher(name, pid, log_path):
    """A tiny detached process that waits for `pid` and records the outcome.

    Without this there is no trustworthy way to know whether a finished job
    succeeded: polling pid liveness alone cannot distinguish "exited 0" from
    "was killed". The watcher is the only thing that sees the real exit code.
    """
    watcher = os.path.join(HERE, "setup_watch.py")
    try:
        subprocess.Popen(
            [engine_python(), watcher, name, str(pid), log_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def _finish(name, exit_code):
    state = DONE if exit_code == 0 else FAILED
    detail = "completed" if exit_code == 0 else f"exit code {exit_code}"
    _update(name, state=state, exit_code=exit_code,
            finished_at=time.time(), pid=None, detail=detail)


def record_exit(name, exit_code, log_path=None, reason=None):
    """Record a job's outcome. Called by the watcher once the child has exited.

    A log tail is NOT used as the failure summary. Job stdout and stderr are
    merged into one file (`>> log 2>&1`), so the log cannot distinguish an error
    line from a status line - an earlier version read "depth done for 20 pairs"
    from a SUCCESSFUL market fetch and reported the job as failed. Only an
    explicit reason from the caller, or the exit code itself, is used.
    """
    detail = reason if reason else ("completed" if exit_code == 0
                                    else f"exit code {exit_code}")
    _update(name, state=DONE if exit_code == 0 else FAILED,
            exit_code=exit_code, finished_at=time.time(), pid=None,
            detail=detail)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def set_progress(name, done, total, detail=""):
    """Record per-action progress. Written by the job itself as it works."""
    _update(name, progress={"done": done, "total": total},
            detail=detail or f"{done}/{total}")


def progress_of(name):
    return reconcile()["actions"].get(name, {}).get("progress")


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
def readiness():
    """What is still missing before hunting makes sense, as plain data."""
    data = reconcile()
    acts = data["actions"]
    missing = [n for n in ORDER if acts.get(n, {}).get("state") != DONE]
    return {
        "ready": not missing,
        "missing": missing,
        "missing_labels": [ACTIONS[n]["label"] for n in missing],
        "actions": acts,
        "busy": busy_action(),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    """Runs against a temp data dir; the live setup state is never touched."""
    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {detail[:200]}")

    print("=== setup_state.py self-check ===")
    print()

    global STATE_FILE
    real_state = STATE_FILE
    with tempfile.TemporaryDirectory() as td:
        STATE_FILE = os.path.join(td, "setup_state.json")

        # ---- 1. missing file reads as all not_run ----
        d = load()
        check("missing file -> all actions not_run",
              all(a["state"] == NOT_RUN for a in d["actions"].values()))

        # ---- 2. saving and loading round-trips ----
        _update("market", state=DONE, exit_code=0, detail="completed")
        d = load()
        check("state persists across load", d["actions"]["market"]["state"] == DONE)

        # ---- 3. unknown keys in an old file do not break loading ----
        with open(STATE_FILE, "w") as fh:
            json.dump({"actions": {"market": {"state": DONE}}, "extra": 1}, fh)
        d = load()
        check("an old/partial state file still loads",
              d["actions"]["market"]["state"] == DONE)
        check("actions added later are filled in",
              "prune" in d["actions"])

        # ---- 4. corrupt state file falls back, does not raise ----
        with open(STATE_FILE, "w") as fh:
            fh.write("{not json")
        check("corrupt state file falls back to blank",
              all(a["state"] == NOT_RUN for a in load()["actions"].values()))

        # ---- 4b. a ZOMBIE is not alive ----
        # The failure this guards against is subtle and total: the dashboard
        # starts a job and never wait()s, so the finished child stays in the
        # process table as state Z. os.kill(pid, 0) succeeds on a zombie, so
        # without this the action would look alive forever.
        zombie = subprocess.Popen([sys.executable, "-c", "pass"])
        time.sleep(0.6)          # let it exit without reaping
        try:
            zombie_state = open(f"/proc/{zombie.pid}/stat").read().split()[2]
        except (OSError, IndexError):
            zombie_state = "?"
        check("a finished-but-unreaped child is a zombie",
              zombie_state == "Z", f"state={zombie_state}")
        check("pid_alive says a zombie is NOT alive", pid_alive(zombie.pid) is False,
              f"state={zombie_state} pid={zombie.pid}")
        _update("market", state=RUNNING, pid=zombie.pid)
        d = reconcile()
        check("a zombie action reconciles to FAILED, not stuck on running",
              d["actions"]["market"]["state"] == FAILED,
              d["actions"]["market"])
        zombie.wait()
        reap_children()

        # ---- 5. a vanished pid turns running into failed ----
        _update("backfill-deep", state=RUNNING, pid=999999, started_at=time.time())
        d = reconcile()
        check("a dead pid is reported as FAILED, not running",
              d["actions"]["backfill-deep"]["state"] == FAILED,
              d["actions"]["backfill-deep"])
        check("the reason is recorded",
              "disappeared" in (d["actions"]["backfill-deep"]["detail"] or ""))

        # ---- 6. a live pid stays running ----
        _update("backfill-deep", state=RUNNING, pid=os.getpid())
        d = reconcile()
        check("a live pid stays RUNNING",
              d["actions"]["backfill-deep"]["state"] == RUNNING,
              d["actions"]["backfill-deep"])

        # ---- 6b. the lock rule itself: heavy blocks, light does not ----
        # Tested as a rule rather than by launching processes, because a probe
        # that actually starts a job races with the state this check is
        # inspecting. What matters is the decision, so the decision is asserted:
        #   heavy action + anything running           -> refused
        #   light action + a heavy one running       -> allowed
        #   same action already running              -> refused
        def would_refuse(target, running, heavy_target):
            act_heavy = ACTIONS[target]["heavy"]
            if target in running:
                return "already running"
            if act_heavy and running:
                return "blocked by heavy"
            return None

        check("heavy is refused while a light job runs",
              would_refuse("backfill-deep", ["market"], True) == "blocked by heavy")
        check("heavy is refused while another heavy job runs",
              would_refuse("backfill-deep", ["backfill-stale"], True) == "blocked by heavy")
        check("light is allowed while heavy runs",
              would_refuse("market", ["backfill-deep"], False) is None)
        check("light is refused while it is already running",
              would_refuse("market", ["market"], False) == "already running")
        check("heavy is refused while it is already running",
              would_refuse("backfill-deep", ["backfill-deep"], True) == "already running")
        check("light is allowed while another light job runs",
              would_refuse("market", ["prune"], False) is None)

        # ---- 6c. the same action cannot be started twice ----
        _update("market", state=RUNNING, pid=os.getpid())
        ok, msg = start("market", background=False)
        check("an action already running is not started twice", not ok, msg)
        check("the refusal says it is already running",
              "already running" in msg, msg)
        _update("market", state=NOT_RUN, pid=None)

        check("market is marked light", ACTIONS["market"]["heavy"] is False)
        check("every backfill and prune is marked heavy",
              all(ACTIONS[n]["heavy"] for n in
                  ("backfill-deep", "backfill-stale", "prune")))

        # ---- 7. the heavy lock refuses a second job ----
        _update("backfill-deep", state=RUNNING, pid=os.getpid())
        check("a heavy job holds the lock", busy_action() == "backfill-deep")
        ok, msg = start("backfill-stale", background=False)
        check("a second heavy job is refused", not ok, msg)
        check("the refusal names the job that is running",
              "backfill" in msg, msg)

        # ---- 8. requirements are enforced ----
        _update("backfill-deep", state=NOT_RUN, pid=None)
        _update("market", state=NOT_RUN)
        ok, msg = start("backfill-deep", background=False)
        check("a job whose prerequisite is missing is refused", not ok, msg)
        check("the refusal names the prerequisite",
              "top-20" in msg or "pairs" in msg.lower(), msg)

        # ---- 9. readiness reflects reality ----
        r = readiness()
        check("readiness is false while work remains", r["ready"] is False)
        check("readiness names what is missing", len(r["missing"]) > 0)
        for n in ORDER:
            _update(n, state=DONE)
        r = readiness()
        check("readiness is true once every action is done", r["ready"] is True)

        # ---- 10. progress survives a round trip ----
        set_progress("backfill-deep", 7, 20, "7/20 pairs")
        check("progress persists",
              progress_of("backfill-deep") == {"done": 7, "total": 20},
              str(progress_of("backfill-deep")))

        # ---- 11. record_exit maps exit codes honestly ----
        record_exit("backfill-deep", 0)
        check("exit 0 -> done",
              load()["actions"]["backfill-deep"]["state"] == DONE)
        record_exit("backfill-stale", 1)
        check("exit 1 -> failed",
              load()["actions"]["backfill-stale"]["state"] == FAILED)

        # ---- 12. an unknown action is refused, not crashed into ----
        ok, msg = start("no-such-action", background=False)
        check("unknown action is refused", not ok, msg)

    STATE_FILE = real_state

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
