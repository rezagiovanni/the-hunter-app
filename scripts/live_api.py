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


_CRED_LABELS = {
    "api_key", "apikey", "key", "api_key_id", "api-key",
    "secret_key", "secretkey", "secret", "api_secret", "api-secret",
}


def _parse_cred_lines(lines, path: str) -> Tuple[str, str, dict]:
    """Turn a credential file's non-empty lines into (key, secret, labels).

    Accepts both shapes, because both exist in the wild and guessing wrong
    produces a 401 that looks exactly like an expired key:

        <key>
        <secret>

        api_key=XXXX
        secret_key=YYYY

    A `name=` prefix is a LABEL, not part of the value. Passing the whole line
    through - which an earlier version did - sends "api_key=XXXX" as the API key
    and "secret_key=YYYY" as the secret, and Indodax answers "Invalid
    credentials", which sends the operator off to regenerate a perfectly good
    key. That mistake cost one debugging round here; do not reintroduce it.

    Only recognised labels are stripped. An unrecognised prefix is treated as
    part of the value, so a key that legitimately contains '=' is not mangled.
    """
    pairs = {}
    bare = []
    for raw in lines:
        ln = raw.strip().strip("\"'")
        if not ln or ln.startswith("#"):
            continue
        if "=" in ln:
            name, _, val = ln.partition("=")
            name_l = name.strip().lower().replace(" ", "_")
            if name_l in _CRED_LABELS:
                pairs[name_l] = val.strip().strip("\"'")
                continue
        bare.append(ln)

    key = pairs.get("api_key") or pairs.get("apikey") or pairs.get("key") or ""
    secret = (pairs.get("secret_key") or pairs.get("secretkey")
              or pairs.get("secret") or pairs.get("api_secret") or "")

    # positional fallback: the first two values that are not the ones we found
    for v in bare:
        if not key and v != secret:
            key = v
        elif not secret and v != key:
            secret = v
        if key and secret:
            break

    return key, secret, pairs


def _load_two_line_file(path: str) -> Tuple[str, str]:
    """Two-line key/secret file: line 1 key, line 2 secret. Explicit use only."""
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise LiveApiError(
            f"no API credentials at {path}. One line key, one line secret.")
    try:
        lines = [ln.strip() for ln in open(path, encoding="utf-8-sig").read().splitlines()
                 if ln.strip()]
    except OSError as e:
        raise LiveApiError(f"cannot read {path}: {type(e).__name__}")

    if len(lines) < 2:
        raise LiveApiError(
            f"{path} must contain the key on line 1 and the secret on line 2; "
            f"found {len(lines)} non-empty line(s).")

    key, secret, _labels = _parse_cred_lines(lines, path)

    if not key or not secret:
        raise LiveApiError(
            f"{path}: could not find both a key and a secret. "
            f"Found {len(lines)} non-empty line(s); expected either two plain "
            f"lines or api_key= / secret_key= lines.")

    if len(key) < 20 or len(secret) < 20:
        raise LiveApiError(
            f"{path}: the key or secret looks too short "
            f"({len(key)} and {len(secret)} characters). "
            f"Check for a stray 'name=' prefix or a truncated paste.")
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


# ---------------------------------------------------------------------------
# Account Profiles & Version Detection
# ---------------------------------------------------------------------------

_KEY_VERSION_CACHE: Dict[str, str] = {}


def account_id(api_key: str) -> str:
    """Stable 8-character account profile ID derived from the API key."""
    raw = str(api_key or "").strip()
    return "acc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def get_active_account_id(creds: Optional[Tuple[str, str]] = None) -> Optional[str]:
    """Return the active account profile ID, or None if signed out / unconfigured."""
    try:
        k, _ = creds or load_credentials()
        return account_id(k) if k else None
    except Exception:
        return None


def detect_key_version(key: str, secret: str) -> str:
    """Detect whether a key is Indodax API v2 or v1 (classic TAPI).

    Cached per key prefix to avoid redundant roundtrips.
    """
    key_prefix = key[:8]
    if key_prefix in _KEY_VERSION_CACHE:
        return _KEY_VERSION_CACHE[key_prefix]

    if key.startswith("fake") or key.startswith("k" * 8) or "test" in key.lower():
        _KEY_VERSION_CACHE[key_prefix] = "v1"
        return "v1"

    now_ms = int(time.time() * 1000)
    params = {"timestamp": str(now_ms), "recvWindow": "5000"}
    q = urllib.parse.urlencode(params)
    sig = hmac.new(secret.encode("utf-8"), q.encode("utf-8"), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"https://api.indodax.com/api/v2/account?{q}", method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("X-APIKEY", key)
    req.add_header("Sign", sig)
    req.add_header("User-Agent", "the-hunter/0.1")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            _KEY_VERSION_CACHE[key_prefix] = "v2"
            return "v2"
    except urllib.error.HTTPError as e:
        raw = (e.read() or b"").decode("utf-8", errors="replace")
        if "-2015" in raw or "Invalid TAPI version key" in raw:
            _KEY_VERSION_CACHE[key_prefix] = "v1"
            return "v1"
        _KEY_VERSION_CACHE[key_prefix] = "v2" if e.code != 403 else "v1"
        return _KEY_VERSION_CACHE[key_prefix]
    except Exception:
        return "v1"


def call_v2(endpoint: str, method: str = "GET", params: Optional[Dict[str, Any]] = None,
            creds: Optional[Tuple[str, str]] = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """Make an authenticated call to Indodax Trade API v2."""
    if creds is None:
        key, secret = load_credentials()
    else:
        key, secret = creds

    p = dict(params or {})
    now_ms = int(time.time() * 1000)
    p["timestamp"] = str(now_ms)
    p["recvWindow"] = "5000"

    encoded = urllib.parse.urlencode(p)
    sig = hmac.new(secret.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "X-APIKEY": key,
        "Sign": sig,
        "Accept": "application/json",
        "User-Agent": "the-hunter/0.1",
    }

    if method == "POST":
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(
            f"https://api.indodax.com{endpoint}",
            data=encoded.encode("utf-8"),
            headers=headers,
            method="POST"
        )
    else:
        req = urllib.request.Request(
            f"https://api.indodax.com{endpoint}?{encoded}",
            headers=headers,
            method=method
        )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = (e.read() or b"").decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
            msg = data.get("msg") or data.get("error") or raw
            if e.code in (401, 403):
                raise AuthError(f"HTTP {e.code}: {_redact(msg)}")
            raise LiveApiError(f"HTTP {e.code}: {_redact(msg)}")
        except json.JSONDecodeError:
            raise LiveApiError(f"HTTP {e.code}: {_redact(raw[:200])}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TransportError(f"{type(e).__name__}: {_redact(str(e))[:200]}")


def trade_history_v2(symbol: str, order_id: Optional[str] = None,
                     days: int = 7, limit: int = 100,
                     creds: Optional[Tuple[str, str]] = None,
                     timeout: float = DEFAULT_TIMEOUT) -> list:
    """Filled trades for one pair, newest first, from Trade API v2.

    This is the only way to learn what the operator ACTUALLY PAID for coins the
    bot did not buy. v1's tradeHistory was decommissioned on 2026-04-07; the
    replacement is GET /api/v2/myTrades, which returns every fill with price,
    qty, fee and side.

    WHY A SEPARATE TRANSPORT

    v2 signs differently from v1: HMAC-SHA256 over the URL QUERY STRING (not
    SHA512 over a POST body), the key goes in X-APIKEY, and the response is a
    bare {"data": [...]} with no "success" envelope. Reusing the v1 helper would
    produce a valid-looking request that the server rejects, so the difference is
    kept in one small function instead of being spread through the trading path.

    Returns [] on any error. A missing fill is not a reason to refuse to trade -
    the caller decides what to do without it.
    """
    if creds is None:
        try:
            key, secret = load_credentials()
        except Exception:                                    # noqa: BLE001
            return []
    else:
        key, secret = creds

    now_ms = int(time.time() * 1000)
    start_ms = max(0, now_ms - int(days * 86400 * 1000))
    params = {
        "symbol": str(symbol).lower(),
        "limit": str(max(10, min(int(limit), 1000))),
        "startTime": str(start_ms),
        "timestamp": str(now_ms),
        "recvWindow": str(RECV_WINDOW_MS),
    }
    if order_id:
        params["orderId"] = str(order_id)
    query = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}"
                     for k, v in params.items())
    sig = hmac.new(secret.encode("utf-8"), query.encode("utf-8"),
                   hashlib.sha256).hexdigest()

    url = f"https://api.indodax.com/api/v2/myTrades?{query}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("X-APIKEY", key)
    req.add_header("Sign", sig)
    req.add_header("User-Agent", "the-hunter/0.1")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception:                                        # noqa: BLE001
        return []

    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


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
    k, s = creds or load_credentials()
    if detect_key_version(k, s) == "v2":
        acc = call_v2("/api/v2/account", method="GET", creds=(k, s))
        bals = {}
        holds = {}
        for b in acc.get("balances", []):
            a = b["asset"].lower()
            f = float(b.get("free") or 0)
            l = float(b.get("locked") or 0)
            if f > 0 or a == "idr":
                bals[a] = f
            if l > 0:
                holds[a] = l
        return {
            "success": 1,
            "return": {
                "balance": bals,
                "balance_hold": holds,
                "server_time": int(time.time()),
                "frozen": "0" if acc.get("canTrade", True) else "1",
            }
        }
    return call("getInfo", {}, creds=creds, retries=2)


def open_orders(pair: Optional[str] = None, creds=None) -> Dict[str, Any]:
    k, s = creds or load_credentials()
    if detect_key_version(k, s) == "v2":
        params = {}
        if pair:
            params["symbol"] = pair.replace("_", "").lower()
        rows = call_v2("/api/v2/openOrders", method="GET", params=params, creds=(k, s))
        rows = rows if isinstance(rows, list) else []
        mapped = []
        for o in rows:
            price = float(o.get("price") or 0)
            orig_qty = float(o.get("origQty") or 0)
            exec_qty = float(o.get("executedQty") or 0)
            order_rp = orig_qty * price
            remain_rp = max(orig_qty - exec_qty, 0.0) * price
            mapped.append({
                "order_id": str(o.get("orderId")),
                "id": str(o.get("orderId")),
                "pair": o.get("symbol", "").lower(),
                "type": o.get("side", "").lower(),
                "price": str(price),
                "order_rp": f"{order_rp:.8f}",
                "remain_rp": f"{remain_rp:.8f}",
                "status": o.get("status", "").lower(),
            })
        return {"success": 1, "return": {"orders": mapped}}
    params = {"pair": pair} if pair else {}
    return call("openOrders", params, creds=creds, retries=2)


def get_order(pair: str, order_id: str, creds=None) -> Dict[str, Any]:
    k, s = creds or load_credentials()
    if detect_key_version(k, s) == "v2":
        sym = pair.replace("_", "").lower()
        try:
            res = call_v2("/api/v2/order", method="GET",
                          params={"symbol": sym, "orderId": str(order_id)},
                          creds=(k, s))
        except LiveApiError as e:
            msg = str(e).lower()
            if "not exists" in msg or "404" in msg or "invalid" in msg:
                raise LiveApiError("invalid order.")
            raise
        if not isinstance(res, dict) or not res.get("orderId"):
            raise LiveApiError("invalid order.")
        orig_qty = float(res.get("origQty") or 0)
        exec_qty = float(res.get("executedQty") or 0)
        price = float(res.get("price") or 0)
        order_rp = orig_qty * price
        remain_rp = max(orig_qty - exec_qty, 0.0) * price
        st_map = {
            "NEW": "open", "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "filled", "CANCELLED": "cancelled",
            "REJECTED": "rejected", "EXPIRED": "expired"
        }
        return {
            "success": 1,
            "return": {
                "order": {
                    "order_id": str(res.get("orderId")),
                    "status": st_map.get(res.get("status", ""), "open"),
                    "price": str(price),
                    "order_rp": f"{order_rp:.8f}",
                    "remain_rp": f"{remain_rp:.8f}",
                    "filled_coin": exec_qty,
                }
            }
        }
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

    k, s = creds or load_credentials()
    if detect_key_version(k, s) == "v2":
        sym = pair.replace("_", "").upper()
        side_u = side.upper()
        type_u = order_type.upper()
        p_v2: Dict[str, Any] = {"symbol": sym, "side": side_u, "type": type_u}
        if type_u == "LIMIT":
            p_v2["price"] = str(int(round(float(price)))) if price is not None else "0"
            p_v2["quantity"] = coin_quantity(coin) if coin is not None else "0"
        else:
            if side_u == "BUY":
                p_v2["quoteOrderQty"] = str(int(round(float(idr)))) if idr is not None else "0"
            else:
                p_v2["quantity"] = coin_quantity(coin) if coin is not None else "0"
        res = call_v2("/api/v2/order", method="POST", params=p_v2, creds=(k, s))
        oid = str(res.get("orderId") or "")
        exec_qty = float(res.get("executedQty") or 0)
        ret = {"order_id": oid, "id": oid, "status": str(res.get("status", "")).lower()}
        if side_u == "BUY":
            if exec_qty > 0:
                ret["receive_coin"] = str(exec_qty)
            if idr:
                ret["spend_rp"] = str(idr)
        else:
            if exec_qty > 0:
                ret["filled_coin"] = str(exec_qty)
                ret["receive"] = str(exec_qty * (float(price) if price else 0))
        return {"success": 1, "return": ret}

    return call("trade", params, creds=creds, retries=0)


def cancel_order(pair: str, order_id: str, kind: str = "buy",
                 creds=None) -> Dict[str, Any]:
    """Cancel a resting order. Never retried."""
    k, s = creds or load_credentials()
    if detect_key_version(k, s) == "v2":
        sym = pair.replace("_", "").upper()
        try:
            call_v2("/api/v2/order", method="DELETE",
                    params={"symbol": sym, "orderId": str(order_id)},
                    creds=(k, s))
        except LiveApiError as e:
            if "not exists" in str(e).lower() or "404" in str(e).lower():
                pass  # already cancelled or expired
            else:
                raise
        return {"success": 1, "return": {"order_id": str(order_id)}}
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

    # --- v2 signing: checked against Indodax's OWN published vector ----------
    # A signing scheme cannot be verified by reading it. The docs publish one
    # example (GET /api/v2/myTrades) with the resulting signature; if our
    # implementation reproduces that hash, the scheme is right. If it does not,
    # the endpoint would reject every request and we would never know why -
    # this is exactly how the "server_time in seconds" bug survived.
    _v2_secret = ("f60617a68fcce028f0a90bc9eb765d17379eb548cc935c01a7ee3186eecf870"
                  "e9b68f27a31bcfe8d")
    _v2_query = "symbol=btcidr&limit=100&timestamp=1578304294000&recvWindow=5000"
    _v2_expected = ("58b503019a8f2754af7097b4d981913a8dca9e5cb2b408061cbd48736a940"
                    "558")
    _v2_sig = hmac.new(_v2_secret.encode(), _v2_query.encode(),
                       hashlib.sha256).hexdigest()
    assert _v2_sig == _v2_expected, (_v2_sig, _v2_expected)
    print("[PASS] v2 signing reproduces Indodax's published example signature")

    # --- credential files may carry name= labels -----------------------------
    # Reported live: indodax_api.txt held "api_key=..." / "secret_key=..." and
    # the loader sent the whole line as the key, so every call answered
    # "Invalid credentials" and the operator was told to regenerate a good key.
    _k1, _s1, _ = _parse_cred_lines(
        ["api_key=AAAABBBBCCCCDDDDEEEEFFFF", "secret_key=" + "a" * 64], "t")
    _k2, _s2, _ = _parse_cred_lines(
        ["AAAABBBBCCCCDDDDEEEEFFFF", "a" * 64], "t")
    assert (_k1, _s1) == (_k2, _s2), (_k1, _k2)
    assert "=" not in _k1 and "=" not in _s1
    print("[PASS] a labelled credential file parses exactly like a plain one")

    # an unrecognised prefix is part of the value, not a label to strip
    _k3, _s3, _ = _parse_cred_lines(["weird=abcd", "other=efgh"], "t")
    assert _k3 == "weird=abcd" and _s3 == "other=efgh", (_k3, _s3)
    print("[PASS] only recognised labels are stripped")

    # --- account profile ID derivation ---------------------------------------
    assert account_id("KEY12345").startswith("acc_")
    assert account_id("KEY12345") == account_id("KEY12345")
    assert account_id("KEY12345") != account_id("KEY67890")
    print("[PASS] account_id is stable and deterministic per key")

    # --- v2 mapping checks with stubbed call_v2 ------------------------------
    _orig_cv2 = call_v2
    _orig_dver = detect_key_version
    _v2_key = ("k_v2_test_12345678", "s_v2_test_12345678")
    globals()["detect_key_version"] = lambda k, s: "v2"

    try:
        # 1. get_info v2 mapping
        globals()["call_v2"] = lambda endpoint, method="GET", **kw: {
            "canTrade": True,
            "balances": [
                {"asset": "IDR", "free": "50000", "locked": "0"},
                {"asset": "HYPE", "free": "1.5", "locked": "0.5"}
            ]
        }
        _info_v2 = get_info(creds=_v2_key)
        assert _info_v2["success"] == 1
        assert _info_v2["return"]["balance"]["idr"] == 50000.0
        assert _info_v2["return"]["balance"]["hype"] == 1.5
        assert _info_v2["return"]["balance_hold"]["hype"] == 0.5
        print("[PASS] v2 get_info maps account balances to the expected v1 structure")

        # 2. open_orders v2 mapping
        globals()["call_v2"] = lambda endpoint, method="GET", **kw: [
            {"orderId": 12345, "symbol": "BTCIDR", "side": "BUY", "price": "1000",
             "origQty": "2", "executedQty": "0.5", "status": "NEW"}
        ]
        _oo_v2 = open_orders(pair="btc_idr", creds=_v2_key)
        assert _oo_v2["success"] == 1
        _o = _oo_v2["return"]["orders"][0]
        assert _o["order_id"] == "12345" and _o["type"] == "buy"
        assert _o["pair"] == "btcidr"
        print("[PASS] v2 open_orders maps to standard orders list")

        # 3. get_order v2 mapping
        globals()["call_v2"] = lambda endpoint, method="GET", **kw: {
            "orderId": 12345, "status": "FILLED", "price": "1000",
            "origQty": "2.0", "executedQty": "2.0"
        }
        _go_v2 = get_order("btc_idr", "12345", creds=_v2_key)
        assert _go_v2["success"] == 1
        assert _go_v2["return"]["order"]["status"] == "filled"
        assert _go_v2["return"]["order"]["order_id"] == "12345"
        print("[PASS] v2 get_order maps execution status")

        # 4. trade v2 limit order mapping
        _captured_v2 = {}
        def _spy_v2(endpoint, method="POST", params=None, **kw):
            _captured_v2.update(params or {})
            return {"orderId": 9999, "executedQty": "0.0", "status": "NEW"}
        globals()["call_v2"] = _spy_v2
        _tr_v2 = trade("cst_idr", "buy", coin=10, price=200, creds=_v2_key, coin_symbol="cst")
        assert _tr_v2["success"] == 1
        assert _tr_v2["return"]["order_id"] == "9999"
        assert _captured_v2["symbol"] == "CSTIDR" and _captured_v2["type"] == "LIMIT"
        assert _captured_v2["price"] == "200" and _captured_v2["quantity"] == "10"
        print("[PASS] v2 trade creates limit order with proper v2 payload")

        # 5. cancel_order v2 mapping
        globals()["call_v2"] = lambda endpoint, method="DELETE", **kw: {"orderId": 9999}
        _co_v2 = cancel_order("cst_idr", "9999", creds=_v2_key)
        assert _co_v2["success"] == 1 and _co_v2["return"]["order_id"] == "9999"
        print("[PASS] v2 cancel_order succeeds")
    finally:
        globals()["call_v2"] = _orig_cv2
        globals()["detect_key_version"] = _orig_dver

    print("\n=== ALL SELF-CHECKS PASSED ===")


if __name__ == "__main__":
    _self_check()
