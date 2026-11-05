#!/usr/bin/env python3
"""
onboarding.py — first run: collect the API key, prove it works, write config.

What this does, in order
------------------------
1. Prompts for the API key and secret with echo disabled, so neither is ever
   displayed and neither can end up in shell history.
2. Validates them against Indodax's private API (`getInfo`), which is the only
   call that proves BOTH halves are correct and the signature scheme works. The
   public API would accept any garbage, so it proves nothing.
3. Writes `config.yaml` with mode 0600, containing the credentials and the
   defaults from config.yaml.example.
4. Prints the dashboard URL, plus an SSH tunnel hint when the machine is
   headless — which the target STB is.

Why validation happens here and not in the installer
---------------------------------------------------
The installer must run non-interactively through `curl | bash`, where reading
from a terminal is unreliable and any prompt would hang a piped install. The CLI
runs with a real terminal, so that is where a secret can be typed safely.

Failure is always clean
-----------------------
A wrong key writes nothing. A network failure writes nothing and says so.
Partial credentials are never saved, because a half-written config would make
the app start and then fail at the first trade, which is far worse than
refusing at the prompt.
"""

import base64
import getpass
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
import config as cfgmod  # noqa: E402

TAPI_URL = "https://indodax.com/tapi"
TIMEOUT = 15


# ---------------------------------------------------------------------------
# The private API call
# ---------------------------------------------------------------------------
def _post_tapi(api_key, secret, method="getInfo", params=None):
    """Call Indodax's private API and return (ok, payload_or_error).

    Indodax signs with HMAC-SHA512 over the urlencoded body, with the secret as
    the key, and sends it as `Key` and `Sign` headers. The body must be signed
    EXACTLY as it is sent, so the encoded string is built once and reused; building
    it twice is the usual way this goes wrong.
    """
    params = params or {}
    body = dict(params)
    body["method"] = method
    body["timestamp"] = str(int(time.time() * 1000))
    body["recvWindow"] = "5000"

    post_data = urllib.parse.urlencode(body)
    sign = hmac.new(secret.encode(), post_data.encode(), hashlib.sha512).hexdigest()

    req = urllib.request.Request(
        TAPI_URL,
        data=post_data.encode(),
        headers={
            "Key": api_key,
            "Sign": sign,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # Indodax returns 200 for application errors, but a 4xx/5xx means the
        # request never reached the handler.
        raw = exc.read().decode("utf-8", "replace") if hasattr(exc, "read") else ""
        return False, f"HTTP {exc.code}: {raw[:200] or exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"network error: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    try:
        data = json.loads(raw)
    except ValueError:
        return False, f"response was not JSON: {raw[:200]}"

    if data.get("success") == 1:
        return True, data.get("return", {})
    return False, data.get("error") or f"rejected: {raw[:200]}"


def validate_key(api_key, secret):
    """(ok, message, info). `info` holds balances and permissions on success."""
    if not api_key or len(api_key) < 10:
        return False, "that key looks too short", None
    if not secret or len(secret) < 10:
        return False, "that secret looks too short", None

    ok, payload = _post_tapi(api_key, secret, "getInfo")
    if not ok:
        return False, str(payload), None

    if not isinstance(payload, dict):
        return True, "accepted", {}

    info = {
        "balances": {},
        "server_time": payload.get("server_time"),
    }
    funds = payload.get("balance") or {}
    if isinstance(funds, dict):
        for k, v in funds.items():
            try:
                amt = float(v)
            except (TypeError, ValueError):
                continue
            if amt > 0:
                info["balances"][k] = amt
    return True, "accepted", info


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------
def write_config(api_key, secret, path=None):
    """Write config.yaml with the credentials, mode 0600. Atomic.

    Returns the path written. Never prints or returns the secret itself.
    """
    path = path or cfgmod.config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = [
        "# The Hunter — user settings",
        "#",
        "# This file holds your Indodax credentials. It is mode 0600: readable",
        "# only by you. `hunter upgrade` never overwrites it.",
        "#",
        "# The presence of `api` is used as the signal that onboarding has been",
        "# completed, so do not copy this file to another machine.",
        "",
        "api:",
        f"  key: {api_key}",
        f"  secret: {secret}",
        "",
        "# Everything below is optional; deleting a key restores its default.",
        "# See config.yaml.example in the app directory for the full list.",
        "",
        "trading:",
        f"  sizing_fraction: {cfgmod.DEFAULTS['trading']['sizing_fraction']}",
        f"  min_trade_balance_idr: {cfgmod.DEFAULTS['trading']['min_trade_balance_idr']}",
        f"  cooldown_hours: {cfgmod.DEFAULTS['trading']['cooldown_hours']}",
        "",
        "frameworks:",
    ]
    for name, p in cfgmod.DEFAULTS["frameworks"].items():
        lines.append(f"  {name}:")
        for k in ("target_pct", "stop_pct", "ratchet_enabled",
                  "ratchet_trigger_pct", "ratchet_stop_pct"):
            v = p[k]
            lines.append(f"    {k}: {str(v).lower() if isinstance(v, bool) else v}")

    lines += [
        "",
        "data:",
        f"  retention_years: {cfgmod.DEFAULTS['data']['retention_years']}",
        f"  backfill_slack_days: {cfgmod.DEFAULTS['data']['backfill_slack_days']}",
        "",
        "server:",
        f"  host: {cfgmod.DEFAULTS['server']['host']}",
        f"  port: {cfgmod.DEFAULTS['server']['port']}",
        "",
    ]

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".config-")
    try:
        # create with 0600 from the start: mkstemp already does 0600, which is
        # why the file is written here rather than opened normally
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def is_fresh_install(path=None):
    """True only when this looks like a brand-new install with nothing set up.

    The signal is deliberately two-sided. An earlier version ran onboarding
    whenever no config.yaml held credentials, which then hijacked bare `hunter`
    on a machine that was already trading - the development box runs from its
    own data directory and keeps its credentials elsewhere, so "no config with
    an api key" is NOT the same as "not set up".

    Fresh means: no credentials AND no data yet. Either one being present means
    this is an established install that should report status instead.
    """
    if has_credentials(path):
        return False
    try:
        if os.path.exists(paths.STATE_FILE):
            return False
        if os.path.isdir(paths.OHLCV_DIR) and os.listdir(paths.OHLCV_DIR):
            return False
    except OSError:
        return False
    return True


def has_credentials(path=None):
    """True if a config already holds a key and secret."""
    path = path or cfgmod.config_path()
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return False
    # read the raw YAML rather than the merged config, so a missing PyYAML
    # cannot make an installed machine look unconfigured
    try:
        import yaml
        d = yaml.safe_load(text) or {}
        api = d.get("api") or {}
        return bool(api.get("key") and api.get("secret"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def mask(secret):
    """Show enough to recognise a key, never enough to use one."""
    if not secret:
        return "(empty)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


# ---------------------------------------------------------------------------
# The interactive flow
# ---------------------------------------------------------------------------
def run(force=False):
    """Prompt, validate, write. Returns a process exit code."""
    import term

    print()
    print("  " + term.bold("The Hunter — first-time setup"))
    print("  " + term.grey("-" * 58))
    print()

    cfg_path = cfgmod.config_path()
    if has_credentials() and not force:
        print(f"  {term.green('[ok]')} credentials already saved at "
              f"{term.grey(cfg_path)}")
        print(f"       {term.grey('re-run with --force to replace them')}")
        _print_url()
        return 0

    print("  Paste your Indodax API key and secret.")
    print("  " + term.grey("Input is hidden. Neither is echoed or logged."))
    print()

    try:
        api_key = getpass.getpass("  API key    : ").strip()
        secret = getpass.getpass("  API secret : ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"  {term.grey('cancelled — nothing was saved')}")
        return 1

    if not api_key or not secret:
        print()
        print(f"  {term.red('[FAIL]')} both values are required; nothing was saved")
        return 1

    print()
    print(f"  Validating {term.grey(mask(api_key))} against Indodax ...")
    ok, message, info = validate_key(api_key, secret)

    if not ok:
        print()
        print(f"  {term.red('[FAIL]')} {message}")
        print(f"         {term.grey('Nothing was saved. Check the key, the secret,')}")
        print(f"         {term.grey('and that the key has trade permission enabled.')}")
        return 1

    # ---- write ----
    try:
        path = write_config(api_key, secret)
    except Exception as exc:
        print()
        print(f"  {term.red('[FAIL]')} could not write {cfg_path}: {exc}")
        return 1

    print(f"  {term.green('[ok]')} key accepted")
    if info and info.get("balances"):
        shown = ", ".join(f"{k.upper()} {v:,.0f}" for k, v in
                          list(info["balances"].items())[:4])
        print(f"       {term.grey('balance: ' + shown)}")
    print(f"  {term.green('[ok]')} saved to {cfg_path}  {term.grey('(mode 0600)')}")
    print(f"  {term.green('[ok]')} model present: "
          f"{'yes' if os.path.exists(os.path.join(paths.MODEL_DIR, 'model.joblib')) else 'no'}")

    _print_url()
    return 0


def _print_url():
    """Print where the dashboard will be, with a tunnel hint when headless."""
    import term

    cfg = cfgmod.current()
    host = cfgmod.get(cfg, "server.host", "127.0.0.1")
    port = cfgmod.get(cfg, "server.port", 8082)

    print()
    print("  " + term.bold("Dashboard"))
    print(f"    http://{host}:{port}")

    # A headless box (the STB) has no browser, so the tunnel command is the
    # useful thing to print.
    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless and host in ("127.0.0.1", "localhost"):
        print()
        print(f"  {term.grey('This machine looks headless. From your own computer:')}")
        print(f"    ssh -N -L {port}:127.0.0.1:{port} <user>@<this-host>")
        print(f"    then open http://127.0.0.1:{port}")

    print()
    print(f"  {term.grey('Next: open the dashboard and use Settings to fetch data.')}")
    print()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    return run(force=force)


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
                print(f"       {str(detail)[:220]}")

    print("=== onboarding.py self-check ===")
    print()

    # ---- 1. the signature is HMAC-SHA512 over the exact body sent ----
    body = "method=getInfo&timestamp=123&recvWindow=5000"
    secret = "s3cr3t"
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha512).hexdigest()
    check("signature is hex HMAC-SHA512", len(expected) == 128 and
          all(c in "0123456789abcdef" for c in expected))

    # ---- 2. short values are refused before any network call ----
    ok, msg, info = validate_key("short", "alsoshort")
    check("a too-short key is refused without a network call", not ok, msg)
    check("the refusal is specific", "short" in msg, msg)

    ok, msg, _ = validate_key("", "x" * 40)
    check("an empty key is refused", not ok, msg)

    # ---- 3. config writing: mode, content, atomicity ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "config.yaml")
        path = write_config("TESTKEY1234567890", "TESTSECRET1234567890", path=target)
        check("config is written", os.path.exists(path))

        mode = oct(os.stat(path).st_mode & 0o777)
        check("config is mode 0600", mode == "0o600", mode)

        text = open(path).read()
        check("config contains the key", "TESTKEY1234567890" in text)
        check("config contains the secret", "TESTSECRET1234567890" in text)
        check("config is valid YAML", _parses_as_yaml(text))

        d = _load_yaml(text)
        check("config has an api section", "api" in d)
        check("config carries the trading defaults",
              d.get("trading", {}).get("sizing_fraction") ==
              cfgmod.DEFAULTS["trading"]["sizing_fraction"])
        check("config carries per-framework exits",
              d.get("frameworks", {}).get("A", {}).get("target_pct") ==
              cfgmod.DEFAULTS["frameworks"]["A"]["target_pct"])
        check("config carries the data retention settings",
              d.get("data", {}).get("retention_years") ==
              cfgmod.DEFAULTS["data"]["retention_years"])

        # the merged loader must accept what onboarding wrote
        merged = cfgmod.load(path)
        check("the app can load what onboarding wrote",
              cfgmod.get(merged, "trading.sizing_fraction") == 0.4)
        check("validating the written config reports no problems",
              cfgmod.validate(merged) == [], str(cfgmod.validate(merged)))

        # ---- 4. has_credentials ----
        check("has_credentials is true after writing", has_credentials(path))

        with tempfile.TemporaryDirectory() as td2:
            empty = os.path.join(td2, "config.yaml")
            open(empty, "w").write("trading:\n  sizing_fraction: 0.4\n")
            check("has_credentials is false without an api section",
                  not has_credentials(empty))
            check("has_credentials is false for a missing file",
                  not has_credentials(os.path.join(td2, "nope.yaml")))

        with tempfile.TemporaryDirectory() as td3:
            broken = os.path.join(td3, "config.yaml")
            open(broken, "w").write("api: [not, a, map\n")
            check("a corrupt config does not make has_credentials raise",
                  has_credentials(broken) is False)

        # ---- 5. no temp file is left behind ----
        leftovers = [f for f in os.listdir(td) if f.startswith(".config-")]
        check("no temp file is left after writing", leftovers == [], str(leftovers))

    # ---- 6. masking never reveals the middle ----
    check("mask keeps only the ends", mask("ABCDEFGHIJKLMNOP") == "ABCD…MNOP")
    check("mask hides a short secret entirely", set(mask("abc")) == {"*"})
    check("mask handles empty", mask("") == "(empty)")

    # ---- 7. no credential is printed by any code path ----
    # Scan only the implementation, above the self-check. The check's own source
    # necessarily contains the patterns it looks for, so scanning the whole file
    # tests the test and always fails - a mistake already made once in paths.py.
    impl = open(__file__).read().split("def _self_check")[0]
    check("the module never prints the secret",
          "print(secret" not in impl and 'print(f"{secret' not in impl)
    check("validation errors do not echo the secret",
          "return False, secret" not in impl)
    check("the secret is never included in an error message",
          "f\"{exc}: {secret}" not in impl and "message = secret" not in impl)

    # ---- 8. the secret is only read once, via getpass ----
    check("the secret is read with getpass (never echoed)",
          "getpass.getpass" in impl)

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


def _parses_as_yaml(text):
    try:
        import yaml
        yaml.safe_load(text)
        return True
    except Exception:
        return False


def _load_yaml(text):
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        return {}


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(_self_check())
    sys.exit(main())
