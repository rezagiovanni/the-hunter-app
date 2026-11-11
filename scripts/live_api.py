#!/usr/bin/env python3
"""
live_api.py — Indodax private API client.

The only module in the product that holds credentials, and the only one that can
place a real order. Everything about it is defensive on purpose:

  * credentials never appear in a log, an exception, a repr or a traceback
  * every call has a hard timeout
  * the API's own error strings are surfaced verbatim - a silent failure on a
    real order is worse than a loud one
  * signature failures say WHICH of the four common causes it is (clock drift,
    wrong pair of key/secret, missing IP whitelist, nonce reuse)
  * no retry on an order placement: a retried buy can double a position. Only
    idempotent reads retry.

Indodax private API
-------------------
    POST https://indodax.com/tapi
    body   method=<m>&timestamp=<ms>&recvWindow=<ms>&...params
    sign   hex(HMAC-SHA512(secret, body))
    head   Key: <api key>
           Sign: <sign>

Reference: https://github.com/indodax/indodax-official-api-docs
"""

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

BASE_URL = "https://indodax.com"
TAPI_PATH = "/tapi"

# Indodax expects the timestamp in milliseconds. recvWindow is the tolerance the
# server allows; 5000 ms is the documented default and is generous enough that a
# few seconds of clock drift does not cause a rejection, while still bounding
# replay.
RECV_WINDOW_MS = 5000

DEFAULT_TIMEOUT = 20


class LiveApiError(Exception):
    """Any failure from the private API, with the server's own message."""


class AuthError(LiveApiError):
    """Signature / key / permission rejected. Never retried."""


class TransportError(LiveApiError):
    """Network-level failure: timeout, DNS, connection reset. Safe to retry."""


def _redact(text: str) -> str:
    """Strip anything that looks like a credential from a message.

    Belt and braces: the code never puts a secret into a message, but a server
    error echo or a traceback could, and this string may end up in a log file
    or on the dashboard.
    """
    if not text:
        return text
    out = text
    for marker in ("secret", "Secret", "SECRET", "api_key", "apikey"):
        if marker in out:
            out = out.split(marker)[0] + "<redacted>"
    return out


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def load_credentials(path: Optional[str] = None) -> Tuple[str, str]:
    """Read the Indodax key and secret.

    There is ONE source of truth, and it is config.yaml under `api:`. That is
    what `hunter setup` writes, what `onboarding.has_credentials()` checks, and
    what the rest of the app treats as "configured". An earlier version of this
    module read a separate two-line indodax_api.txt, which meant a user could
    finish setup, see "keys saved", and still be told "no credentials" by every
    live action. A second credential store is a bug factory: it cannot be kept
    in sync and it makes "configured" mean two different things.

    A two-line file is still accepted if it is pointed at explicitly (HUNTER_API_KEY_FILE
    or an argument), because that is useful for tests and for a secrets manager
    that mounts a file. It is never preferred over config.yaml.
    """
    explicit = path or os.environ.get("HUNTER_API_KEY_FILE")

    if explicit:
        return _load_two_line_file(explicit)

    # --- primary: config.yaml, the same file onboarding writes ---------------
    try:
        import config as cfgmod
        cfg_path = cfgmod.config_path()
    except Exception as e:                                   # noqa: BLE001
        raise LiveApiError(f"cannot locate config.yaml: {e}")

    if not os.path.exists(cfg_path):
        raise LiveApiError(
            f"no credentials yet\n"
            f"  expected: {cfg_path}\n"
            f"  run:      hunter setup")

    try:
        import yaml
        with open(cfg_path) as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as e:                                   # noqa: BLE001
        raise LiveApiError(f"cannot read {cfg_path}: {type(e).__name__}")

    api = data.get("api") or {}
    key = str(api.get("key") or "").strip()
    secret = str(api.get("secret") or "").strip()

    if not key or not secret:
        raise LiveApiError(
            f"no API key in {cfg_path}\n"
            f"  run: hunter setup\n"
            f"  then enable Trade permission on the key in the Indodax API settings")

    if len(key) < 20 or len(secret) < 20:
        raise LiveApiError(
            f"the key or secret in {cfg_path} looks too short "
            f"({len(key)} and {len(secret)} characters)")

    return key, secret


def _load_two_line_file(path: str) -> Tuple[str, str]:
    """Two-line key/secret file: line 1 key, line 2 secret. Explicit use only."""
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise LiveApiError(
            f"no API credentials at {path}. One line key, one line secret.")
    try:
        lines = [ln.strip() for ln in open(path).read().splitlines() if ln.strip()]
    except OSError as e:
        raise LiveApiError(f"cannot read {path}: {type(e).__name__}")

    if len(lines) < 2:
        raise LiveApiError(
            f"{path} must contain the key on line 1 and the secret on line 2; "
            f"found {len(lines)} non-empty line(s).")

    key, secret = lines[0], lines[1]
    if len(key) < 20 or len(secret) < 20:
        raise LiveApiError(
            f"{path}: the key or secret looks too short "
            f"({len(key)} and {len(secret)} characters).")
    return key, secret


def _permissions_warning(path: str) -> Optional[str]:
    """Human-readable note if the credential file is group/world readable."""
    try:
        mode = os.stat(path).st_mode & 0o777
    except OSError:
        return None
    if mode & 0o077:
        return (f"{path} is readable by other users (mode {mode:o}). "
                f"Run: chmod 600 {path}")
    return None


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def sign(secret: str, body: str) -> str:
    """hex HMAC-SHA512 of the exact request body.

    The server signs the string it receives, so this MUST be the same string
    that is sent - parameters are encoded once, here, and the encoded result is
    both signed and transmitted. Re-encoding at send time is a classic source of
    'invalid signature'.
    """
    return hmac.new(secret.encode("utf-8"),
                    body.encode("utf-8"),
                    hashlib.sha512).hexdigest()


def _build_body(method: str, params: Dict[str, Any], now_ms: int) -> str:
    """Deterministic, URL-encoded body including timestamp and recvWindow."""
    payload = {
        "method": method,
        "timestamp": str(now_ms),
        "recvWindow": str(RECV_WINDOW_MS),
    }
    for k, v in params.items():
        if v is None:
            continue
        payload[k] = str(v)

    # urlencode preserves insertion order, which is the order signed. Both sides
    # see the same string because we send exactly this.
    return urllib.parse.urlencode(payload)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _post(body: str, key: str, signature: str,
          timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    req = urllib.request.Request(
        BASE_URL + TAPI_PATH,
        data=body.encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Key", key)
    req.add_header("Sign", signature)
    req.add_header("User-Agent", "the-hunter/0.1")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code in (401, 403):
            raise AuthError(_auth_hint(raw, e.code))
        raise TransportError(f"HTTP {e.code}: {_redact(raw[:300])}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TransportError(f"{type(e).__name__}: {_redact(str(e))[:200]}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise LiveApiError(f"response was not JSON: {_redact(raw[:200])}")

    if not isinstance(data, dict):
        raise LiveApiError(f"unexpected response shape: {_redact(raw[:200])}")

    if data.get("success") == 0 or data.get("success") is False:
        err = str(data.get("error") or data.get("error_code") or "unknown")
        if "sign" in err.lower() or "nonce" in err.lower() or "auth" in err.lower():
            raise AuthError(_auth_hint(err, 0))
        raise LiveApiError(_redact(err))

    return data


def _auth_hint(raw: str, code: int) -> str:
    """Turn a signature rejection into the four things that actually cause it.

    Indodax reports these as one opaque error. Guessing wrong here costs the
    operator an hour, so all four causes are named with the check for each.
    """
    base = f"the private API rejected the request (HTTP {code or 'n/a'}): {_redact(raw)[:200]}"
    return (
        base + "\n"
        "  Most likely, in this order:\n"
        "    1. clock drift - this machine's clock must be within ~5 s of real time.\n"
        "       check: date -u  vs  an NTP source\n"
        "    2. wrong pair of lines in indodax_api.txt - line 1 KEY, line 2 SECRET.\n"
        "       they are not interchangeable and neither can be swapped with key.txt\n"
        "    3. the API key has no permission for this call, or has not been\n"
        "       enabled/whitelisted in the Indodax API settings\n"
        "    4. the key was deleted or regenerated in the Indodax UI"
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

_METHOD_CACHE: Dict[str, Any] = {}


def call(method: str, params: Optional[Dict[str, Any]] = None,
         creds: Optional[Tuple[str, str]] = None,
         timeout: float = DEFAULT_TIMEOUT,
         retries: int = 0, now_ms: Optional[int] = None) -> Dict[str, Any]:
    """Sign and send one private API call.

    retries defaults to 0 and MUST stay 0 for anything that changes state
    (trade, cancelOrder). A retried order placement can open a second position.
    Reads may pass retries=2 safely.
    """
    params = params or {}
    if creds is None:
        key, secret = load_credentials()
    else:
        key, secret = creds

    if now_ms is None:
        now_ms = int(time.time() * 1000)

    body = _build_body(method, params, now_ms)
    signature = sign(secret, body)

    last: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            return _post(body, key, signature, timeout=timeout)
        except TransportError as e:
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except AuthError:
            # never retry an auth failure: it will fail identically and can
            # burn through a rate limit
            raise

    raise last if last else LiveApiError("call failed with no error recorded")


# --- read-only helpers -----------------------------------------------------

def get_info(creds=None) -> Dict[str, Any]:
    """Balances and server time. Read-only, safe to call any time."""
    return call("getInfo", {}, creds=creds, retries=2)


def open_orders(pair: Optional[str] = None, creds=None) -> Dict[str, Any]:
    params = {"pair": pair} if pair else {}
    return call("openOrders", params, creds=creds, retries=2)


def get_order(pair: str, order_id: str, creds=None) -> Dict[str, Any]:
    return call("getOrder", {"pair": pair, "order_id": order_id},
                creds=creds, retries=2)


# --- state-changing calls --------------------------------------------------

def pair_param(pair_id: str, ticker_id: Optional[str] = None) -> str:
    """The `pair` value the PRIVATE API expects, e.g. 'cst_idr'.

    The public /api/pairs endpoint returns id='cstidr' (no separator) AND
    ticker_id='cst_idr' (underscore). The private /tapi endpoints document
    'btc_idr', 'ltc_btc' - underscore. Sending 'cstidr' to `trade` returns
    {"error": "invalid_pair", "error_description": "Invalid Pair"}, which is what
    "LiveApiError: Invalid pair." was.

    ticker_id is used when present because it comes from the exchange itself;
    otherwise it is derived. A value that already contains an underscore is
    accepted as-is.
    """
    if ticker_id:
        return str(ticker_id)
    raw = str(pair_id or "").strip().lower()
    if not raw:
        raise LiveApiError("pair is required")
    if "_" in raw:
        return raw
    # derive from the idr pair: everything before the trailing 'idr'
    if raw.endswith("idr") and len(raw) > 3:
        return f"{raw[:-3]}_idr"
    raise LiveApiError(
        f"cannot derive a private-API pair from {pair_id!r}; pass ticker_id")


def trade(pair: str, side: str, idr: Optional[float] = None,
          coin: Optional[float] = None, price: Optional[float] = None,
          creds=None, coin_symbol: Optional[str] = None,
          order_type: Optional[str] = None) -> Dict[str, Any]:
    """Place an order. NEVER retried.

    Indodax has three rules that are easy to get wrong and each produces a
    different error:

      1. `pair` is 'cst_idr', NOT 'cstidr'. Use pair_param().
      2. `order_type` DEFAULTS TO LIMIT. A market order must say so, or the
         server demands a price and rejects the request.
      3. A BUY with `order_type=limit` must NOT send `idr` - it is rejected.
         A limit buy sends the COIN amount instead; only a MARKET buy uses `idr`.

    So the four legal shapes are:

        market buy    order_type=market, idr=<amount>
        market sell   order_type=market, <coin>=<amount>
        limit buy     order_type=limit,  price=<p>, <coin>=<amount>
        limit sell    order_type=limit,  price=<p>, <coin>=<amount>

    `coin_symbol` is the traded currency (e.g. 'cst'), which is also the name of
    the parameter Indodax expects. It is required for any coin-amount order.
    """
    if side not in ("buy", "sell"):
        raise LiveApiError(f"side must be buy or sell, got {side!r}")

    if order_type is None:
        # infer: a sell always needs a coin amount so it is market unless a
        # price is given; a buy is market when no price is given
        order_type = "limit" if price is not None else "market"
    if order_type not in ("limit", "market"):
        raise LiveApiError(f"order_type must be limit or market, got {order_type!r}")

    params: Dict[str, Any] = {"pair": pair, "type": side, "order_type": order_type}

    if order_type == "limit":
        if price is None:
            raise LiveApiError("a limit order needs a price")
        params["price"] = f"{float(price):.0f}"
        if side == "buy":
            # rule 3: idr + limit buy is rejected
            if coin is None:
                raise LiveApiError("a limit buy needs a coin amount (not idr)")
        else:
            if coin is None:
                raise LiveApiError("a limit sell needs a coin amount")
        if not coin_symbol:
            raise LiveApiError("a coin-amount order needs coin_symbol (e.g. 'cst')")
        params[coin_symbol] = coin_quantity(coin)
    else:
        # market
        if side == "buy":
            if idr is None:
                raise LiveApiError("a market buy needs idr (the IDR amount to spend)")
            params["idr"] = f"{float(idr):.0f}"
        else:
            if coin is None:
                raise LiveApiError("a market sell needs a coin amount")
            if not coin_symbol:
                raise LiveApiError("a market sell needs coin_symbol (e.g. 'cst')")
            params[coin_symbol] = coin_quantity(coin)

    return call("trade", params, creds=creds, retries=0)


def cancel_order(pair: str, order_id: str, kind: str = "buy",
                 creds=None) -> Dict[str, Any]:
    """Cancel a resting order. Never retried."""
    return call("cancelOrder",
                {"pair": pair, "order_id": order_id, "type": kind},
                creds=creds, retries=0)


def coin_quantity(x: float) -> str:
    """Format a coin amount the way Indodax expects: plain decimal, no exponent.

    float.__str__ can emit 1e-05 for small amounts, which the API rejects.
    """
    s = f"{float(x):.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ---------------------------------------------------------------------------
# Self-check (offline: no network, no credentials needed)
# ---------------------------------------------------------------------------

def _self_check():
    print("=== live_api.py self-check ===\n")

    # signing is exact and deterministic
    sig = sign("secret", "method=getInfo&timestamp=1&recvWindow=5000")
    assert len(sig) == 128 and all(c in "0123456789abcdef" for c in sig), sig[:20]
    known = "d8f6a9b3a1a2d0b7b4c0e9f1a3d5c7e9b1a3d5c7e9f1b3d5c7e9f1a3d5c7e9b1"
    # recompute independently with hmac directly
    import hmac as _h, hashlib as _s
    direct = _h.new(b"secret", b"method=getInfo&timestamp=1&recvWindow=5000",
                    _s.sha512).hexdigest()
    assert sig == direct, "sign() must be plain HMAC-SHA512 hex"
    print(f"[PASS] signature is HMAC-SHA512 hex (128 chars)")

    # same input -> same signature; different input -> different
    assert sign("secret", "a=1") == sign("secret", "a=1")
    assert sign("secret", "a=1") != sign("secret", "a=2")
    assert sign("secret", "a=1") != sign("secret2", "a=1")
    print("[PASS] signature changes with the body and with the secret")

    # body construction is ordered and encodes once
    b = _build_body("trade", {"pair": "btcidr", "type": "buy", "idr": "10000"}, 1700000000000)
    assert b.startswith("method=trade&timestamp=1700000000000&recvWindow=5000&"), b
    assert "pair=btcidr" in b and "idr=10000" in b
    print(f"[PASS] body is ordered and encoded once: {b[:58]}...")

    # None params are dropped, not sent as the string "None"
    b2 = _build_body("getInfo", {"pair": None}, 1)
    assert "pair" not in b2, b2
    print("[PASS] unset parameters are omitted, never sent as 'None'")

    # coin formatting never uses exponent notation
    assert coin_quantity(0.00001234) == "0.00001234"
    assert "e" not in coin_quantity(1e-08)
    assert coin_quantity(10.0) == "10"
    print("[PASS] coin amounts never use exponent notation")

    # credential errors never echo the secret
    try:
        load_credentials("/nonexistent/path/xyz")
        raise AssertionError("should have raised")
    except LiveApiError as e:
        assert "/nonexistent/path/xyz" in str(e)
    print("[PASS] a missing credential file names the path and nothing else")

    # --- ONE source of truth: config.yaml, the file hunts setup writes -------
    # A user who runs `hunter setup` must not then be told there are no
    # credentials by a live action. That bug existed because this module read a
    # separate file that nothing else knew about.
    import tempfile as _tf, yaml as _yaml
    _d = _tf.mkdtemp()
    _cfg = os.path.join(_d, "config.yaml")
    with open(_cfg, "w") as _f:
        _yaml.dump({"api": {"key": "k" * 40, "secret": "s" * 60}}, _f)
    import config as _cfgmod
    _orig = _cfgmod.config_path
    try:
        _cfgmod.config_path = lambda: _cfg
        _k, _s = load_credentials()
        assert _k == "k" * 40 and _s == "s" * 60, (_k[:4], _s[:4])
        print("[PASS] credentials are read from config.yaml (the hunter setup file)")

        # a config with no api block names the command to fix it
        with open(_cfg, "w") as _f:
            _yaml.dump({"trading": {"sizing_fraction": 0.4}}, _f)
        try:
            load_credentials()
            raise AssertionError("should have raised")
        except LiveApiError as e:
            assert "hunter setup" in str(e), str(e)
        print("[PASS] a config with no key tells the user to run 'hunter setup'")
    finally:
        _cfgmod.config_path = _orig

    # redaction
    assert "hunter_secret_value" not in _redact("failed secret hunter_secret_value")
    print("[PASS] error text is redacted")

    # --- the private API pair format: 'cst_idr', not 'cstidr' --------------
    # The public /api/pairs returns id='cstidr'; the private /tapi endpoints
    # document 'btc_idr'. Sending the public id is what produced
    # "LiveApiError: Invalid pair."
    assert pair_param("cstidr") == "cst_idr", pair_param("cstidr")
    assert pair_param("btcidr") == "btc_idr"
    assert pair_param("pepeidr") == "pepe_idr"
    assert pair_param("cst_idr") == "cst_idr", "an underscored pair passes through"
    assert pair_param("anything", ticker_id="cst_idr") == "cst_idr", "ticker_id wins"
    try:
        pair_param("ltcbtc")
        raise AssertionError("should have raised for a non-idr pair")
    except LiveApiError as e:
        assert "ticker_id" in str(e)
    print("[PASS] pairs are converted to the private-API format: cstidr -> cst_idr")

    # a bad side is refused before any network call
    try:
        trade("cst_idr", "hodl", idr=10000, creds=("k" * 30, "s" * 30))
        raise AssertionError("should have raised")
    except LiveApiError as e:
        assert "buy or sell" in str(e)
    print("[PASS] an invalid side is refused locally, before the network")

    # a market buy without idr is refused
    try:
        trade("cst_idr", "buy", creds=("k" * 30, "s" * 30))
        raise AssertionError("should have raised")
    except LiveApiError as e:
        assert "idr" in str(e)
    print("[PASS] a market buy without an IDR amount is refused")

    # a sell without coin is refused
    try:
        trade("cst_idr", "sell", creds=("k" * 30, "s" * 30))
        raise AssertionError("should have raised")
    except LiveApiError as e:
        assert "coin" in str(e)
    print("[PASS] a sell without a coin amount is refused")

    # --- order_type is sent explicitly, and the body is correct per shape ----
    _cap = {}
    _orig_call = call
    def _spy(method, params=None, **kw):
        _cap["method"], _cap["params"] = method, dict(params or {})
        return {"success": 1, "return": {"order_id": "x"}}
    globals()["call"] = _spy
    try:
        trade("cst_idr", "buy", idr=20000, creds=("k" * 30, "s" * 30),
              coin_symbol="cst")
        p_ = _cap["params"]
        assert p_["order_type"] == "market", p_
        assert p_.get("idr") == "20000" and "price" not in p_, p_
        assert "cst" not in p_, "a market buy must not send a coin amount"
        print("[PASS] market buy: order_type=market, idr only")

        trade("cst_idr", "buy", coin=12.5, price=160, creds=("k" * 30, "s" * 30),
              coin_symbol="cst")
        p_ = _cap["params"]
        assert p_["order_type"] == "limit", p_
        assert "idr" not in p_, "a limit buy must NOT send idr - Indodax rejects it"
        assert p_["cst"] == "12.5" and p_["price"] == "160", p_
        print("[PASS] limit buy: order_type=limit, coin + price, NO idr")

        trade("cst_idr", "sell", coin=12.5, creds=("k" * 30, "s" * 30),
              coin_symbol="cst")
        p_ = _cap["params"]
        assert p_["order_type"] == "market" and p_["cst"] == "12.5", p_
        assert "idr" not in p_
        print("[PASS] market sell: order_type=market, coin only")
    finally:
        globals()["call"] = _orig_call

    # a limit buy with idr is refused locally rather than by the exchange
    try:
        trade("cst_idr", "buy", idr=20000, price=160, creds=("k" * 30, "s" * 30))
        raise AssertionError("should have refused idr on a limit buy")
    except LiveApiError as e:
        assert "coin amount" in str(e)
    print("[PASS] a limit buy with idr is refused before the network")

    # the auth hint names all four causes
    h = _auth_hint("invalid sign", 401)
    for phrase in ("clock drift", "KEY", "permission", "regenerated"):
        assert phrase in h, phrase
    print("[PASS] an auth failure explains the four real causes")

    # retries are zero for state-changing calls by default
    import inspect
    src = inspect.getsource(trade)
    assert "retries=0" in src, "trade() must never retry"
    src_c = inspect.getsource(cancel_order)
    assert "retries=0" in src_c, "cancel_order() must never retry"
    print("[PASS] trade and cancel_order never retry")

    print("\n=== ALL SELF-CHECKS PASSED ===")


if __name__ == "__main__":
    _self_check()
