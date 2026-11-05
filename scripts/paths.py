#!/usr/bin/env python3
"""
paths.py — single source of truth for every data path The Hunter uses.

Why this exists
---------------
Before this module, each script derived its own paths, and two of them hardcoded
the developer's absolute home directory:

    scripts/dashboard.py : BASE_DIR = Path("<dev-home>/Documents/projects/the-hunter")
    model/predict.py     : BASE     = Path("<dev-home>/Documents/projects/the-hunter")

That works on exactly one machine. An installable app must run from wherever it
is unpacked, so the location is resolved at runtime instead.

Resolution order for the data directory
---------------------------------------
1. `$HUNTER_DATA_DIR`                     — explicit override (tests, custom layouts)
2. `$XDG_DATA_HOME/the-hunter`             — if XDG_DATA_HOME is set
3. `~/.local/share/the-hunter`             — the installed-app default
4. `<repo>/data`                           — development fallback

Rule 4 is what keeps this change invisible on the dev machine: with no
environment variables set and no `~/.local/share/the-hunter` directory present,
every path resolves to exactly the file it resolved to before.

The repo root is always derived from this file's location, never from the
current working directory, so a script behaves the same however it is invoked.

No trading logic lives here. Paths only.
"""

import os

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
# The repo root is the directory containing scripts/ and data/. Derived from
# this file's own location so it cannot depend on the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
def _expand(path: str, env=None) -> str:
    """Expand a leading ~ using the supplied environment's HOME.

    os.path.expanduser() reads the real process environment, which makes it
    untestable and wrong when a caller passes a synthetic env. Expand by hand.
    """
    env = os.environ if env is None else env
    if path == "~":
        return env.get("HOME") or os.path.expanduser("~")
    if path.startswith("~/"):
        home = env.get("HOME") or os.path.expanduser("~")
        return os.path.join(home, path[2:])
    return path


def resolve_data_dir(env=None, repo_root=None) -> str:
    """Return the data directory, creating nothing and reading nothing.

    Pure function of the environment plus the repo layout, so it is trivially
    testable and has no side effects.

    Order: HUNTER_DATA_DIR > XDG_DATA_HOME/the-hunter > ~/.local/share/the-hunter
           > <repo>/data
    """
    env = os.environ if env is None else env
    repo_root = REPO_ROOT if repo_root is None else repo_root

    explicit = env.get("HUNTER_DATA_DIR")
    if explicit:
        return os.path.abspath(_expand(explicit, env))

    xdg = env.get("XDG_DATA_HOME")
    home = env.get("HOME") or os.path.expanduser("~")

    if xdg:
        base = os.path.join(os.path.abspath(os.path.expanduser(xdg)), "the-hunter")
    else:
        base = os.path.join(home, ".local", "share", "the-hunter")

    # The install root holds app/, data/, market/ and config.yaml. Data lives in
    # `data/` under it, NOT at the root: returning the root as the data dir made
    # it identical to the code tree's parent, so state files would land beside
    # the source and config.yaml would resolve outside the app entirely.
    installed = os.path.join(base, "data")

    # Use the installed-app location only when it already exists. On the dev
    # machine nothing has created it, so we fall through to the repo's own data
    # directory and behaviour is unchanged.
    if os.path.isdir(base):
        return installed

    return os.path.abspath(os.path.join(repo_root, "data"))


def config_root(env=None, repo_root=None) -> str:
    """The directory holding config.yaml.

    Installed: the install root (~/.local/share/the-hunter), i.e. the parent of
    data/, so wiping data/ does not destroy settings. Development: the repo, so
    a dev config.yaml sits beside the code and is gitignored.
    """
    env = os.environ if env is None else env
    repo_root = REPO_ROOT if repo_root is None else repo_root

    explicit = env.get("HUNTER_DATA_DIR")
    if explicit:
        return os.path.dirname(os.path.abspath(_expand(explicit, env)))

    xdg = env.get("XDG_DATA_HOME")
    home = env.get("HOME") or os.path.expanduser("~")
    base = (os.path.join(os.path.abspath(_expand(xdg, env)), "the-hunter")
            if xdg else os.path.join(home, ".local", "share", "the-hunter"))

    if os.path.isdir(base):
        return base

    return os.path.abspath(repo_root)


def resolve_app_root(env=None, repo_root=None) -> str:
    """Root of the application's install tree.

    When data lives under a separate HUNTER_DATA_DIR, the app tree is still the
    repo. Kept as a function so callers do not each invent their own idea.
    """
    env = os.environ if env is None else env
    return (REPO_ROOT if repo_root is None else repo_root)


# ---------------------------------------------------------------------------
# Resolved paths (module-level, computed once at import)
# ---------------------------------------------------------------------------
DATA_DIR = resolve_data_dir()
APP_ROOT = resolve_app_root()

OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
HUNT_TICKS_DIR = os.path.join(DATA_DIR, "hunt_ticks")
HUNT_EVENTS_FILE = os.path.join(DATA_DIR, "hunt_events.jsonl")

PAIRS_FILE = os.path.join(DATA_DIR, "pairs.json")
STATE_FILE = os.path.join(DATA_DIR, "hunter_state.json")
LOCK_FILE = os.path.join(DATA_DIR, "hunter.lock")
KILL_FILE = os.path.join(DATA_DIR, "hunter_stop")
CONTROL_FILE = os.path.join(DATA_DIR, "hunter_control.json")

STATIC_DIR = os.path.join(APP_ROOT, "static")
CONFIG_DIR = config_root()
MODEL_DIR = os.path.join(APP_ROOT, "model")
SCRIPTS_DIR = os.path.join(APP_ROOT, "scripts")


def describe() -> dict:
    """Human-readable summary, used by `hunter where` and self-checks."""
    return {
        "repo_root": REPO_ROOT,
        "config_dir": CONFIG_DIR,
        "app_root": APP_ROOT,
        "data_dir": DATA_DIR,
        "ohlcv": OHLCV_DIR,
        "depth": DEPTH_DIR,
        "reports": REPORTS_DIR,
        "logs": LOGS_DIR,
        "hunt_ticks": HUNT_TICKS_DIR,
        "hunt_events": HUNT_EVENTS_FILE,
        "pairs": PAIRS_FILE,
        "state": STATE_FILE,
        "control": CONTROL_FILE,
        "static": STATIC_DIR,
        "model": MODEL_DIR,
        "scripts": SCRIPTS_DIR,
    }


def _self_check():
    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {detail}")

    print("=== paths.py self-check ===")
    print()

    # --- 1. explicit override wins ---
    got = resolve_data_dir(env={"HUNTER_DATA_DIR": "/tmp/custom"}, repo_root="/repo")
    check("HUNTER_DATA_DIR override wins", got == "/tmp/custom")

    # --- 2. XDG when set and directory exists ---
    # resolution checks isdir(), so use a directory that really exists
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "the-hunter", "data"), exist_ok=True)
        got = resolve_data_dir(env={"XDG_DATA_HOME": td}, repo_root="/repo")
        check("XDG_DATA_HOME/the-hunter/data used when present",
              got == os.path.join(td, "the-hunter", "data"), got)

    # --- 3. dev fallback: no env, no installed dir -> <repo>/data ---
    # Use a HOME that cannot contain an installed dir
    with tempfile.TemporaryDirectory() as fakehome:
        got = resolve_data_dir(env={"HOME": fakehome}, repo_root="/repo")
        check("dev fallback is <repo>/data", got == "/repo/data")

    # --- 4. ~/.local/share/the-hunter wins over repo when it exists ---
    with tempfile.TemporaryDirectory() as fakehome:
        base = os.path.join(fakehome, ".local", "share", "the-hunter")
        os.makedirs(os.path.join(base, "data"), exist_ok=True)
        got = resolve_data_dir(env={"HOME": fakehome}, repo_root="/repo")
        check("installed location preferred once it exists",
              got == os.path.join(base, "data"), got)

    # --- 5. tilde expansion on the explicit override ---
    got = resolve_data_dir(env={"HUNTER_DATA_DIR": "~/hdata", "HOME": "/fake"}, repo_root="/repo")
    check("~ expanded in HUNTER_DATA_DIR", got == "/fake/hdata")

    # --- 6. live resolution on this machine is the repo's data dir ---
    # This is the property that makes the change invisible in development.
    live = resolve_data_dir(env={}, repo_root=REPO_ROOT)
    check("this machine resolves to <repo>/data", live == os.path.join(REPO_ROOT, "data"))

    # --- 7. every declared path sits under DATA_DIR or APP_ROOT ---
    d = describe()
    check("all data paths under data_dir",
          all(d[k].startswith(d["data_dir"]) for k in
              ("ohlcv", "depth", "reports", "logs", "hunt_ticks", "state", "control")))
    check("all app paths under app_root",
          all(d[k].startswith(d["app_root"]) for k in ("static", "model", "scripts")))

    # --- 7b. installed vs development layout ---
    # The installed app must place data INSIDE the install root, not at it, and
    # config.yaml at the root (so wiping data/ keeps settings). Getting this
    # wrong made the data dir identical to the code tree's parent, and put
    # config.yaml in ~/.local/share, outside the app entirely.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as fakehome:
        base = os.path.join(fakehome, ".local", "share", "the-hunter")
        os.makedirs(os.path.join(base, "data"), exist_ok=True)

        data = resolve_data_dir(env={"HOME": fakehome}, repo_root="/repo")
        check("installed data dir is <root>/data, not <root>",
              data == os.path.join(base, "data"), data)

        cfg_root = config_root(env={"HOME": fakehome}, repo_root="/repo")
        check("config root is the install root", cfg_root == base, cfg_root)

        check("config.yaml does not land in ~/.local/share",
              os.path.dirname(os.path.join(cfg_root, "config.yaml")) !=
              os.path.join(fakehome, ".local", "share"),
              cfg_root)

    check("dev data dir is still <repo>/data",
          resolve_data_dir(env={"HOME": "/nonexistent-home"},
                           repo_root=REPO_ROOT) ==
          os.path.join(REPO_ROOT, "data"))

    # --- 8. no hardcoded developer home anywhere in the *implementation* ---
    # Scan only the code above this self-check: the check's own source
    # necessarily contains the pattern it looks for, and testing the test would
    # always fail.
    needle = "/" + "home/"
    impl = open(__file__).read().split("def _self_check")[0]
    check("no hardcoded /home/<user> path in paths.py", needle not in impl)

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
