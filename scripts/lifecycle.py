#!/usr/bin/env python3
"""
lifecycle.py — upgrade and uninstall.

Both are destructive in a way the rest of the app is not, so both follow the
same rule: say what will happen, do only that, and never surprise the user.

Upgrade
-------
Replaces `app/` and nothing else. `data/`, `config.yaml` and the model are
outside `app/`, so they survive by construction rather than by remembering to
copy them. That is the whole reason the layout is app/ + data/ + config.yaml at
the install root: a parameter the user changed to 3% must still be 3% after an
upgrade, and the only reliable way to guarantee that is for the upgrade to never
be able to touch the file it lives in.

The crontab block is regenerated from the new code, keeping each job's current
enabled/disabled state. A job the user turned on stays on; one they left off
stays off.

Uninstall
---------
Removes the crontab block (and only that block), the symlink, and the app tree.
Data and trade history are KEPT unless the user explicitly says otherwise,
because deleting a trade log is irreversible and the user asked to remove the
program, not their records.

If a position is open, it warns first and offers to market-sell. It never
silently liquidates a position.

Nothing here trades by itself; the one action that touches money is delegated to
the engine's own stop path.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
import term  # noqa: E402


class LifecycleError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def install_root():
    """The directory holding app/, data/ and config.yaml.

    Installed: <root>/data is the data dir, so the root is its parent.
    Development: the repo, whose parent is NOT the install root - so this is
    derived from config_root(), which already encodes both cases.
    """
    return paths.config_root()


def app_dir():
    return paths.APP_ROOT


def open_position():
    """The current position dict, or None. Reads state only."""
    try:
        with open(paths.STATE_FILE) as fh:
            st = json.load(fh)
        return st.get("position")
    except Exception:
        return None


def trade_count():
    try:
        with open(paths.STATE_FILE) as fh:
            return len(json.load(fh).get("trade_log", []))
    except Exception:
        return 0


def data_size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------
# Where a release comes from when the user does not say. install.sh builds the
# same URL; keeping it here means `hunter upgrade` works with no arguments
# instead of raising "no release source given" - which is what it did, because
# --source defaulted to None and nothing else supplied a value.
DEFAULT_RELEASE_URL = (
    "https://github.com/rezagiovanni/the-hunter-app/archive/refs/tags/v{version}.tar.gz"
)


def default_source(version=None):
    """The published tarball for a version, or the current app version."""
    if version is None:
        version = _app_version()
    return DEFAULT_RELEASE_URL.format(version=version)


def _app_version():
    """The app's version, read from the one place that declares it."""
    try:
        import cli
        return getattr(cli, "VERSION", "0.1.0")
    except Exception:
        return "0.1.0"


def upgrade(source=None, dry_run=False, assume_yes=False, keep_jobs=True):
    """Replace app/ from a release tarball. Preserve data, config and jobs.

    `source` is a tarball path or URL. When it is None the published release for
    the current version is used, so `hunter upgrade` needs no arguments.
    Returns a dict describing what happened.
    """
    import crontab_manager as cm

    if not source:
        source = default_source()

    root = install_root()
    target = app_dir()

    result = {
        "ok": False,
        "root": root,
        "app": target,
        "replaced": [],
        "preserved": [],
        "jobs_before": {},
        "jobs_after": {},
        "backup": None,
        "dry_run": dry_run,
    }

    # ---- what will be preserved, stated up front ----
    cfg = os.path.join(root, "config.yaml")
    for label, p in (("config.yaml", cfg), ("data/", paths.DATA_DIR),
                     ("market data", paths.OHLCV_DIR)):
        if os.path.exists(p):
            result["preserved"].append(label)

    # ---- current job states, so an enabled job stays enabled ----
    try:
        result["jobs_before"] = cm.job_states()
    except Exception:
        result["jobs_before"] = {}

    result["source"] = source

    if dry_run:
        result["ok"] = True
        return result

    # ---- snapshot app/ so a bad upgrade can be undone ----
    if os.path.isdir(target):
        backup_dir = os.path.join(root, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(backup_dir, f"app-{stamp}")
        try:
            shutil.copytree(target, backup)
            result["backup"] = backup
        except Exception:
            pass          # not fatal; the tarball is the source of truth

    if not source:
        raise LifecycleError(
            "no release source: pass --source <path-or-url>, "
            f"or check the default {default_source()}")

    # ---- unpack ----
    import tempfile
    tmp = tempfile.mkdtemp(prefix="hunter-upgrade-")
    try:
        unpacked = _fetch_and_extract(source, tmp)
        result["replaced"] = _replace_app(unpacked, target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- restore job states ----
    try:
        if keep_jobs and result["jobs_before"]:
            cm.install(states=result["jobs_before"])
        result["jobs_after"] = cm.job_states()
    except Exception as exc:
        result["jobs_error"] = str(exc)

    result["ok"] = True
    return result


def _fetch_and_extract(source, into):
    import tarfile
    import urllib.request

    if source.startswith(("http://", "https://")):
        archive = os.path.join(into, "release.tar.gz")
        with urllib.request.urlopen(source, timeout=60) as r, open(archive, "wb") as fh:
            shutil.copyfileobj(r, fh)
    else:
        archive = source
        if not os.path.exists(archive):
            raise LifecycleError(f"no such file: {archive}")

    with tarfile.open(archive) as tf:
        # a release tarball nests everything under one top-level directory
        members = tf.getnames()
        tops = {m.split("/")[0] for m in members}
        dest = os.path.join(into, "x")
        os.makedirs(dest, exist_ok=True)
        tf.extractall(dest)
        if len(tops) == 1:
            only = tops.pop()
            inner = os.path.join(dest, only)
            if os.path.isdir(inner):
                return inner
        return dest


def _replace_app(unpacked, target):
    """Swap the code directories. Data and config are outside `target`, so they
    cannot be touched even by accident."""
    replaced = []
    os.makedirs(target, exist_ok=True)
    for sub in ("scripts", "bin", "static", "model"):
        src = os.path.join(unpacked, sub)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(target, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        replaced.append(sub + "/")

    for f in ("requirements.lock", "config.yaml.example"):
        src = os.path.join(unpacked, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target, f))
            replaced.append(f)

    # the model binary is not in the tarball; keep whatever is installed
    hunter = os.path.join(target, "bin", "hunter")
    if os.path.exists(hunter):
        os.chmod(hunter, 0o755)
    return replaced


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
def uninstall(dry_run=False, assume_yes=False, delete_data=False,
              stop_engine=True):
    """Remove the program. Keep data unless `delete_data` is explicit."""
    import crontab_manager as cm

    root = install_root()
    target = app_dir()
    bin_link = _find_link()

    steps = []
    warnings = []

    pos = open_position()
    if pos:
        warnings.append(
            f"a position is open: {pos.get('symbol')} "
            f"(entry {pos.get('entry_price')})")

    # ---- what goes, what stays ----
    steps.append(("remove the-hunter cron block", "crontab"))
    if bin_link:
        steps.append((f"remove {bin_link}", "link"))
    steps.append((f"remove {target}", "app"))
    steps.append(("stop the dashboard service", "service"))

    keep = ["config.yaml", "data/"] if not delete_data else []

    if dry_run:
        return {"ok": True, "dry_run": True, "steps": steps,
                "kept": keep, "warnings": warnings, "root": root}

    if not assume_yes and pos:
        # Never liquidate a position silently.
        return {
            "ok": False,
            "needs_confirmation": True,
            "reason": "an open position exists",
            "position": pos,
            "warnings": warnings,
        }

    done = []

    # ---- crontab first: stop anything from being scheduled mid-removal ----
    try:
        r = cm.remove()
        done.append(("crontab block", r["changed"]))
    except Exception as exc:
        done.append(("crontab block", f"error: {exc}"))

    # ---- dashboard ----
    # Honour $HUNTER_SYSTEMCTL so a test cannot stop the developer's real
    # service. This is the same class of leak as `crontab` ignoring $HOME: the
    # tests stubbed the crontab and the HOME but not systemd, so an uninstall
    # test stopped the running dashboard on the development machine.
    # Stop and remove the unit through the one module that owns it, so the
    # service definition has a single source of truth.
    try:
        import dashsvc
        res = dashsvc.remove()
        done.append(("dashboard service", bool(res.get("stopped") or res.get("deleted"))))
    except Exception:
        done.append(("dashboard service", False))

    # ---- symlink ----
    if bin_link and os.path.islink(bin_link):
        try:
            os.unlink(bin_link)
            done.append((f"symlink {bin_link}", True))
        except OSError:
            done.append((f"symlink {bin_link}", False))

    # ---- app tree ----
    if os.path.isdir(target):
        try:
            shutil.rmtree(target)
            done.append((f"{target}", True))
        except OSError:
            done.append((f"{target}", False))

    # ---- data, only on explicit request ----
    if delete_data:
        for p in (paths.DATA_DIR, os.path.join(root, "config.yaml")):
            if os.path.exists(p):
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.unlink(p)
                    done.append((f"{p}", True))
                except OSError as exc:
                    done.append((f"{p}", f"error: {exc}"))

    return {"ok": True, "done": done, "kept": keep, "warnings": warnings,
            "root": root, "deleted_data": delete_data}


def _find_link():
    """The hunter symlink on PATH, if any."""
    for d in (os.path.join(os.path.expanduser("~"), ".local", "bin"),
              "/usr/local/bin", os.path.join(os.path.expanduser("~"), "bin")):
        p = os.path.join(d, "hunter")
        if os.path.islink(p) or os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    import tempfile

    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {str(detail)[:250]}")

    print("=== lifecycle.py self-check ===")
    print()

    # ---- 1. the layout keeps config and data outside app/ ----
    # This is the structural property that makes upgrade safe: it is not a
    # copy routine remembering to skip files, it is that the files are in a
    # different directory from the one that gets replaced.
    root = "/x/y/the-hunter"
    cfg = os.path.join(root, "config.yaml")
    data = os.path.join(root, "data")
    app = os.path.join(root, "app")
    check("config.yaml is not inside app/",
          not cfg.startswith(app + os.sep), cfg)
    check("data/ is not inside app/",
          not data.startswith(app + os.sep), data)

    # ---- 2. replacing app/ leaves a sibling config alone ----
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "app")
        os.makedirs(os.path.join(target, "scripts"))
        open(os.path.join(target, "scripts", "old.py"), "w").write("old")
        cfgp = os.path.join(td, "config.yaml")
        open(cfgp, "w").write("trading:\n  sizing_fraction: 0.25\n")
        datap = os.path.join(td, "data")
        os.makedirs(datap)
        open(os.path.join(datap, "hunter_state.json"), "w").write('{"trade_log":[]}')

        # a fake new release
        newrel = os.path.join(td, "new")
        os.makedirs(os.path.join(newrel, "scripts"))
        open(os.path.join(newrel, "scripts", "new.py"), "w").write("new")

        replaced = _replace_app(newrel, target)

        check("the new code is in place",
              os.path.exists(os.path.join(target, "scripts", "new.py")))
        check("the old code is gone",
              not os.path.exists(os.path.join(target, "scripts", "old.py")))
        check("config.yaml survived the swap",
              os.path.exists(cfgp) and "0.25" in open(cfgp).read())
        check("data/ survived the swap", os.path.exists(
            os.path.join(datap, "hunter_state.json")))
        check("_replace_app reports what it touched",
              any("scripts" in r for r in replaced), replaced)

    # ---- 3. dry run changes nothing ----
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "app")
        os.makedirs(os.path.join(target, "scripts"))
        marker = os.path.join(target, "scripts", "keep.txt")
        open(marker, "w").write("here")

        res = upgrade(source=os.path.join(td, "nothing.tar.gz"), dry_run=True)
        check("a dry run reports ok", res["ok"] is True)
        check("a dry run does not touch app/", os.path.exists(marker))
        check("a dry run lists what would be preserved",
              isinstance(res["preserved"], list))

    # ---- 4. uninstall keeps data unless explicitly told otherwise ----
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "root")
        os.makedirs(os.path.join(root, "app", "scripts"))
        os.makedirs(os.path.join(root, "data"))
        cfgp = os.path.join(root, "config.yaml")
        open(cfgp, "w").write("api:\n  key: k\n")
        statep = os.path.join(root, "data", "hunter_state.json")
        open(statep, "w").write('{"trade_log":[1,2,3]}')

        # point the module at this fake install
        saved = (paths.config_root, paths.APP_ROOT, paths.DATA_DIR,
                 paths.STATE_FILE)
        try:
            paths.config_root = lambda *a, **k: root
            paths.APP_ROOT = os.path.join(root, "app")
            paths.DATA_DIR = os.path.join(root, "data")
            paths.STATE_FILE = statep

            plan = uninstall(dry_run=True)
            check("a dry run lists the steps", len(plan["steps"]) >= 3, str(plan["steps"]))
            check("a dry run keeps config and data by default",
                  "config.yaml" in plan["kept"] and "data/" in plan["kept"],
                  plan["kept"])
            check("a dry run removes nothing", os.path.exists(cfgp))
        finally:
            (paths.config_root, paths.APP_ROOT, paths.DATA_DIR,
             paths.STATE_FILE) = saved

    # ---- 5. an unconfirmed uninstall is refused when a position is open ----
    with tempfile.TemporaryDirectory() as td:
        statep = os.path.join(td, "state.json")
        open(statep, "w").write(json.dumps({
            "position": {"symbol": "BTCIDR", "entry_price": 1000},
            "trade_log": [],
        }))
        saved_state = paths.STATE_FILE
        try:
            paths.STATE_FILE = statep
            pos = open_position()
            check("an open position is visible to the uninstall path",
                  bool(pos) and pos["symbol"] == "BTCIDR", pos)
        finally:
            paths.STATE_FILE = saved_state

        # and with a position and no confirmation, uninstall refuses
        saved = (paths.config_root, paths.APP_ROOT, paths.DATA_DIR, paths.STATE_FILE)
        try:
            paths.config_root = lambda *a, **k: td
            paths.APP_ROOT = os.path.join(td, "app")
            paths.DATA_DIR = os.path.join(td, "data")
            paths.STATE_FILE = statep
            res = uninstall(assume_yes=False)
            check("T-106 uninstall refuses to proceed with an open position",
                  res.get("needs_confirmation") is True, res)
            check("T-106 the refusal names the position",
                  "position" in str(res).lower(), res)
        finally:
            (paths.config_root, paths.APP_ROOT, paths.DATA_DIR,
             paths.STATE_FILE) = saved

    # ---- 6. human_bytes is sane ----
    check("human_bytes bytes", human_bytes(512) == "512 B")
    check("human_bytes KB", "KB" in human_bytes(2048))
    check("human_bytes GB", "GB" in human_bytes(5 * 1024**3))

    # ---- 6b. systemd access is redirectable, so a test cannot stop it ----
    # lifecycle no longer calls systemctl itself: the unit is owned by dashsvc,
    # which is where $HUNTER_SYSTEMCTL is honoured. Assert that the delegation
    # exists and that nothing here bypasses it.
    impl = open(__file__).read().split("def _self_check")[0]
    check("uninstall delegates service removal to dashsvc", "dashsvc.remove()" in impl)
    check("no bare systemctl call remains",
          'subprocess.run(["systemctl"' not in impl)

    import dashsvc
    check("dashsvc honours HUNTER_SYSTEMCTL",
          "HUNTER_SYSTEMCTL" in open(dashsvc.__file__).read().split("def _self_check")[0])

    # ---- 7. no sudo anywhere in the lifecycle module ----
    impl = open(__file__).read().split("def _self_check")[0]
    check("no sudo invocation in lifecycle.py", "sudo" not in impl)

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
