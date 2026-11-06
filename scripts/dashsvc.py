#!/usr/bin/env python3
"""
dashsvc.py — the dashboard's systemd user service.

Why this module exists
----------------------
The installer never created a systemd unit, but `hunter status` told the user to
run `systemctl --user restart hunter-dashboard` when the dashboard was down. On a
fresh install that command fails with "Unit not found", because there is no unit
to restart. Reporting a fix that cannot work is worse than reporting nothing.

This module is the single place that knows the unit's name, its contents, and how
to install, start, stop and inspect it. Everything goes through `systemctl`, so
it behaves identically whether systemd is present or not - but every call honours
$HUNTER_SYSTEMCTL, so a test can never touch the real service (the same reason
$HUNTER_CRONTAB exists).

Design notes
------------
* The unit is a USER unit (~/.config/systemd/user). No root is ever needed, which
  matches the rest of the install: everything lives under $HOME.
* `WantedBy=default.target` plus `Restart=always` makes it survive reboots and
  crashes - but NOT an explicit `stop`, which is deliberate: `hunter stop` must
  actually stop it.
* It is installed ENABLED and STARTED, because the install is otherwise inert
  (no crontab job runs), and the dashboard is how the user reaches Settings to
  fetch data. Starting a read-only web UI places no orders.
* `Environment=HUNTER_DATA_DIR=` is set explicitly so the service does not depend
  on guessing its layout from cwd.
"""

import os
import subprocess

UNIT_NAME = "hunter-dashboard.service"
PORT = 8082


def systemctl_cmd() -> str:
    """The systemctl binary, honouring $HUNTER_SYSTEMCTL.

    Mirrors crontab_manager.crontab_cmd(). Without this, a test that runs an
    install into a throwaway HOME still stops the developer's real dashboard.
    """
    return os.environ.get("HUNTER_SYSTEMCTL") or "systemctl"


def unit_dir(env=None) -> str:
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME")
    if base:
        return os.path.join(os.path.abspath(os.path.expanduser(base)), "systemd", "user")
    home = env.get("HOME") or os.path.expanduser("~")
    return os.path.join(home, ".config", "systemd", "user")


def unit_path(env=None) -> str:
    return os.path.join(unit_dir(env), UNIT_NAME)


def render_unit(app_root, data_dir=None, home=None) -> str:
    """The unit file. Pure function of its inputs, so it can be asserted on."""
    home = home or os.path.expanduser("~")
    data_dir = data_dir or os.path.join(home, ".local", "share", "the-hunter", "data")
    py = os.path.join(app_root, "venv", "bin", "python")
    script = os.path.join(app_root, "scripts", "dashboard.py")

    return f"""[Unit]
Description=The Hunter Dashboard (port {PORT})
Documentation=file://{os.path.join(app_root, 'README.md')}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={app_root}
Environment=PYTHONUNBUFFERED=1
Environment=HUNTER_DATA_DIR={data_dir}
ExecStart={py} {script}
Restart=always
RestartSec=5
# The dashboard reads CSVs and serves HTML. It needs no privileges.
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""


def _run(ctl, args, timeout=20):
    return subprocess.run([ctl] + args, capture_output=True, text=True, timeout=timeout)


def is_installed(env=None, ctl=None) -> bool:
    ctl = ctl or systemctl_cmd()
    try:
        r = _run(ctl, ["--user", "cat", UNIT_NAME])
        return r.returncode == 0
    except Exception:
        return False


def is_active(env=None, ctl=None) -> bool:
    ctl = ctl or systemctl_cmd()
    try:
        r = _run(ctl, ["--user", "is-active", UNIT_NAME])
        return r.stdout.strip() == "active"
    except Exception:
        return False


def install(app_root, data_dir=None, env=None, ctl=None, start=True) -> dict:
    """Write the unit, reload the daemon, enable and (optionally) start it.

    Returns a dict of what happened, including why, so the caller can report
    honestly rather than assuming success.
    """
    env = os.environ if env is None else env
    ctl = ctl or systemctl_cmd()
    res = {"written": False, "reloaded": False, "enabled": False,
           "started": False, "path": unit_path(env), "error": None}

    path = unit_path(env)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(render_unit(app_root, data_dir=data_dir, home=env.get("HOME")))
        os.chmod(path, 0o644)
        res["written"] = True
    except Exception as e:
        res["error"] = f"could not write {path}: {e}"
        return res

    try:
        r = _run(ctl, ["--user", "daemon-reload"])
        res["reloaded"] = r.returncode == 0
    except Exception as e:
        res["error"] = f"daemon-reload failed: {e}"
        return res

    try:
        r = _run(ctl, ["--user", "enable", UNIT_NAME])
        res["enabled"] = r.returncode == 0
    except Exception:
        pass

    if start:
        try:
            r = _run(ctl, ["--user", "restart", UNIT_NAME], timeout=40)
            res["started"] = r.returncode == 0
        except Exception:
            pass

    return res


def remove(env=None, ctl=None) -> dict:
    """Stop, disable and delete the unit. Safe when nothing is installed."""
    env = os.environ if env is None else env
    ctl = ctl or systemctl_cmd()
    res = {"stopped": False, "disabled": False, "deleted": False}

    try:
        res["stopped"] = _run(ctl, ["--user", "stop", UNIT_NAME]).returncode == 0
    except Exception:
        pass
    try:
        res["disabled"] = _run(ctl, ["--user", "disable", UNIT_NAME]).returncode == 0
    except Exception:
        pass

    path = unit_path(env)
    if os.path.exists(path):
        try:
            os.remove(path)
            res["deleted"] = True
        except Exception:
            pass

    try:
        _run(ctl, ["--user", "daemon-reload"])
    except Exception:
        pass
    return res


def user_bus_available(ctl=None) -> bool:
    """Can we actually talk to this user's systemd manager?

    This matters because `systemctl --user cat` can read the unit FILE while
    `start` / `is-active` fail, because those need a connection to the user
    manager's DBus. On a headless box reached over SSH, the manager often is not
    running at all unless linger is enabled - the unit looks installed and will
    never start. Distinguishing the two turns "not running" into a real reason.
    """
    ctl = ctl or systemctl_cmd()
    try:
        r = _run(ctl, ["--user", "is-system-running"], timeout=10)
        # "offline"/"unknown" mean we reached the client but not a manager.
        out = (r.stdout or "").strip()
        if r.returncode != 0 and out in ("", "offline", "unknown"):
            return False
        return r.returncode == 0
    except Exception:
        return False


def is_enabled(env=None, ctl=None) -> bool:
    ctl = ctl or systemctl_cmd()
    try:
        r = _run(ctl, ["--user", "is-enabled", UNIT_NAME], timeout=10)
        return r.stdout.strip() in ("enabled", "enabled-runtime")
    except Exception:
        return False


def journal_tail(lines: int = 12, ctl=None):
    """The last few log lines for the unit, or [] if unavailable."""
    ctl = ctl or systemctl_cmd()
    try:
        r = subprocess.run([ctl, "--user", "-u", UNIT_NAME, "-n", str(lines),
                            "--no-pager", "status"],
                           capture_output=True, text=True, timeout=15)
        return [l for l in (r.stdout or "").splitlines() if l.strip()][:lines]
    except Exception:
        return []


def diagnose(env=None, ctl=None) -> dict:
    """Why is the dashboard not running, and what should the user actually do?

    Returns {installed, active, enabled, bus, reason, fix}. The fix is a command
    that can work in the state described - never a restart of a unit that does
    not exist, and never a plain start when the user manager is unreachable.
    """
    env = os.environ if env is None else env
    ctl = ctl or systemctl_cmd()

    installed = is_installed(env, ctl)
    active = is_active(env, ctl)
    enabled = is_enabled(env, ctl) if installed else False
    bus = user_bus_available(ctl)

    if not installed:
        return {"installed": False, "active": False, "enabled": False, "bus": bus,
                "reason": "the dashboard service is not installed",
                "fix": "hunter dashboard install"}

    if active:
        return {"installed": True, "active": True, "enabled": enabled, "bus": bus,
                "reason": "running", "fix": None}

    if not bus:
        return {
            "installed": True, "active": False, "enabled": enabled, "bus": False,
            "reason": ("the unit exists, but this user's systemd manager is not "
                       "reachable, so it cannot be started"),
            "fix": "sudo loginctl enable-linger $USER",
        }

    return {
        "installed": True, "active": False, "enabled": enabled, "bus": True,
        "reason": "the unit exists but has stopped or failed to start",
        "fix": f"systemctl --user status {UNIT_NAME}",
    }


def status(env=None, ctl=None) -> dict:
    """What a caller needs to print an honest line about the dashboard service."""
    env = os.environ if env is None else env
    d = diagnose(env, ctl)
    d["unit"] = UNIT_NAME
    d["port"] = PORT
    d["path"] = unit_path(env)
    return d


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check() -> int:
    import tempfile

    failures = []

    def check(label, cond, detail=None):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail is not None:
                print(f"       {str(detail)[:300]}")

    # --- 1. the unit renders and names what systemd needs ---
    u = render_unit("/opt/app", data_dir="/opt/data", home="/home/u")
    check("unit has a [Unit] section", "[Unit]" in u)
    check("unit has a [Service] section", "[Service]" in u)
    check("unit has an [Install] section", "[Install]" in u)
    check("it runs the dashboard script", "dashboard.py" in u)
    check("it uses the app's venv python", "/opt/app/venv/bin/python" in u)
    check("it sets HUNTER_DATA_DIR explicitly",
          "Environment=HUNTER_DATA_DIR=/opt/data" in u, u)
    check("it restarts on crash", "Restart=always" in u)
    check("it is wanted by default.target", "WantedBy=default.target" in u)
    check("it works without working directory surprises",
          "WorkingDirectory=/opt/app" in u)
    check("the port in the description matches PORT",
          str(PORT) in u.splitlines()[1], u.splitlines()[1])

    # --- 2. paths follow XDG ---
    with tempfile.TemporaryDirectory() as td:
        p1 = unit_path(env={"HOME": td})
        check("unit path is under ~/.config/systemd/user",
              p1 == os.path.join(td, ".config", "systemd", "user", UNIT_NAME), p1)
        p2 = unit_path(env={"HOME": td, "XDG_CONFIG_HOME": os.path.join(td, "cfg")})
        check("XDG_CONFIG_HOME is honoured",
              p2 == os.path.join(td, "cfg", "systemd", "user", UNIT_NAME), p2)

    # --- 2b. diagnose() names the right cause in each state ---
    with tempfile.TemporaryDirectory() as td:
        def stub(name, script):
            path = os.path.join(td, name)
            with open(path, "w") as fh:
                fh.write("#!/bin/sh\n" + script)
            os.chmod(path, 0o755)
            return path

        # no user manager: is-system-running says offline
        no_bus = stub("ctl_nobus", 'case "$*" in *is-system-running*) echo offline; exit 1;; '
                                  '*cat*) exit 0;; *is-active*) echo inactive;; '
                                  '*is-enabled*) echo enabled;; esac; exit 0\n')
        d = diagnose(env={"HOME": td}, ctl=no_bus)
        check("diagnose: reports the manager unreachable", d["bus"] is False, d)
        check("diagnose: fix is enable-linger, not a restart",
              "enable-linger" in (d["fix"] or ""), d)

        # unit genuinely absent
        absent = stub("ctl_absent", 'case "$*" in *cat*) exit 1;; '
                                    '*is-active*) echo inactive;; esac; exit 0\n')
        d2 = diagnose(env={"HOME": td}, ctl=absent)
        check("diagnose: absent unit says not installed", d2["installed"] is False, d2)
        check("diagnose: absent unit fix is install",
              d2["fix"] == "hunter dashboard install", d2)

        # unit present, manager works, not running
        running_bus = stub("ctl_bus", 'case "$*" in *is-system-running*) echo running;; '
                                      '*cat*) exit 0;; *is-active*) echo inactive;; '
                                      '*is-enabled*) echo enabled;; esac; exit 0\n')
        d3 = diagnose(env={"HOME": td}, ctl=running_bus)
        check("diagnose: stopped unit names a status command",
              "systemctl --user status" in (d3["fix"] or ""), d3)
        check("diagnose: never advises restarting a missing unit",
              "restart" not in (d2["fix"] or ""), d2)

    # --- 3. systemctl is redirectable, so a test cannot touch the real unit ---
    import inspect
    src = inspect.getsource(systemctl_cmd)
    check("systemctl honours HUNTER_SYSTEMCTL", "HUNTER_SYSTEMCTL" in src)
    # Only the implementation matters, not this check's own source - the
    # assertion text would otherwise match itself and fail forever.
    impl = open(os.path.abspath(__file__)).read().split("def _self_check")[0]
    check("no subprocess call hardcodes systemctl",
          'subprocess.run(["systemctl"' not in impl
          and "subprocess.run(['systemctl'" not in impl)

    # --- 4. install/remove round-trip against a stub, never the real daemon ---
    with tempfile.TemporaryDirectory() as td:
        stub = os.path.join(td, "systemctl")
        log = os.path.join(td, "calls")
        with open(stub, "w") as fh:
            fh.write('#!/bin/sh\nprintf "%s\\n" "$*" >> "' + log + '"\n'
                     'case "$*" in *is-active*) echo inactive ;; esac\nexit 0\n')
        os.chmod(stub, 0o755)

        env = {"HOME": td, "HUNTER_SYSTEMCTL": stub}
        app = os.path.join(td, "app")
        os.makedirs(os.path.join(app, "scripts"))
        os.makedirs(os.path.join(app, "venv", "bin"))

        res = install(app, data_dir=os.path.join(td, "data"), env=env, ctl=stub)
        check("install wrote the unit", res["written"], res)
        check("install reloaded the daemon", res["reloaded"], res)
        check("install enabled it", res["enabled"], res)
        check("install started it", res["started"], res)
        check("the unit file exists on disk", os.path.exists(unit_path(env)))

        calls = open(log).read()
        check("it asked systemd to reload", "daemon-reload" in calls)
        check("it asked systemd to enable", "enable" in calls)
        check("it asked systemd to restart", "restart" in calls)

        st = status(env=env, ctl=stub)
        check("status reports it as installed", st["installed"], st)

        rem = remove(env=env, ctl=stub)
        check("remove stopped it", rem["stopped"], rem)
        check("remove disabled it", rem["disabled"], rem)
        check("remove deleted the unit file", rem["deleted"], rem)
        check("the unit file is gone", not os.path.exists(unit_path(env)))

        # removing twice must not raise
        rem2 = remove(env=env, ctl=stub)
        check("remove is safe when nothing is installed", rem2["deleted"] is False)

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


def _main(argv) -> int:
    """Small CLI so install.sh, doctor.py and the user share one implementation."""
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(prog="dashsvc", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="action")

    p_inst = sub.add_parser("install", help="write, enable and start the unit")
    p_inst.add_argument("--app", required=True, help="the installed app directory")
    p_inst.add_argument("--data", default=None, help="the data directory")
    p_inst.add_argument("--no-start", action="store_true")

    sub.add_parser("remove", help="stop, disable and delete the unit")
    sub.add_parser("status", help="report whether the unit is installed and active")
    sub.add_parser("unit", help="print the unit file that would be written")

    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.action == "install":
        res = install(a.app, data_dir=a.data, start=not a.no_start)
        if a.json:
            print(_json.dumps(res, indent=2))
        else:
            if res["written"]:
                print(f"  unit    {res['path']}")
            for key in ("written", "reloaded", "enabled", "started"):
                mark = "ok" if res[key] else "no"
                print(f"  {key:<9} {mark}")
            if res["error"]:
                print(f"  error   {res['error']}")
        return 0 if res["written"] and not res["error"] else 1

    if a.action == "remove":
        res = remove()
        print(_json.dumps(res, indent=2) if a.json else res)
        return 0

    if a.action == "status":
        st = status()
        print(_json.dumps(st, indent=2) if a.json else st)
        return 0

    if a.action == "unit":
        print(render_unit(os.environ.get("HUNTER_APP_ROOT", os.getcwd())))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(_main(sys.argv[1:]))
    sys.exit(_self_check())
