#!/usr/bin/env python3
"""
term.py — colour and layout for the CLI.

Two rules, both deliberate:

1. **Colour is never the only signal.** Every status keeps its word and symbol
   ("[PASS]", "⚠", "[FAIL]"). Colour reinforces the meaning; it does not carry
   it. This survives colour blindness, a screenshot, a pasted log, and a
   terminal that ignored the escapes.

2. **Colour switches itself off.** When `NO_COLOR` is set (any value), or stdout
   is not a TTY, or `TERM` is "dumb", every function returns plain text. Without
   this, `hunter status > report.txt` fills with escape codes and the file is
   unusable.

Layout is fixed-width by display column, with wide characters counted as two so
box drawing does not shear when a symbol or a Chinese pair name appears.
"""

import os
import sys
import unicodedata

# ---------------------------------------------------------------------------
# Is colour on?
# ---------------------------------------------------------------------------
def supports_color(stream=None) -> bool:
    """Whether ANSI colour should be emitted to `stream` (default: stdout)."""
    if os.environ.get("NO_COLOR") is not None:
        return False                      # https://no-color.org — presence is enough
    if os.environ.get("TERM", "") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True                       # explicit override, used by the tests
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


_COLOR = None


def enabled() -> bool:
    """Cached answer, so a pipe check happens once."""
    global _COLOR
    if _COLOR is None:
        _COLOR = supports_color()
    return _COLOR


def set_enabled(value: bool) -> None:
    """Force colour on or off. Tests and `--color/--no-color` use this."""
    global _COLOR
    _COLOR = value


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_GREY = "\033[90m"


def _wrap(code: str, text: str) -> str:
    if not enabled():
        return text
    return f"{code}{text}{RESET}"


def bold(t):    return _wrap(_BOLD, t)
def dim(t):     return _wrap(_DIM, t)
def red(t):     return _wrap(_RED, t)
def green(t):   return _wrap(_GREEN, t)
def yellow(t):  return _wrap(_YELLOW, t)
def blue(t):    return _wrap(_BLUE, t)
def magenta(t): return _wrap(_MAGENTA, t)
def cyan(t):    return _wrap(_CYAN, t)
def grey(t):    return _wrap(_GREY, t)


# ---------------------------------------------------------------------------
# Semantic helpers — the vocabulary the CLI actually uses
# ---------------------------------------------------------------------------
OK = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

SYMBOL_OK = "✓"
SYMBOL_WARN = "⚠"
SYMBOL_FAIL = "✗"
SYMBOL_DOT = "•"
SYMBOL_ARROW = "→"


def status_ok(text: str) -> str:
    return green(text)


def status_warn(text: str) -> str:
    return yellow(text)


def status_fail(text: str) -> str:
    return red(text)


def badge(text: str, kind: str = "ok") -> str:
    """A short status word, coloured but always readable as words."""
    texts = {
        "ok": green, "on": green, "running": green, "fresh": green,
        "warn": yellow, "stale": yellow, "off": yellow,
        "fail": red, "error": red, "stopped": red,
        "info": cyan, "muted": grey,
    }
    fn = texts.get(kind, cyan)
    return fn(text)


def mark(ok: bool) -> str:
    """A tick or cross, plus the colour. Symbol alone is never the only cue —
    callers pair it with text."""
    return green(SYMBOL_OK) if ok else red(SYMBOL_FAIL)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def width(s: str) -> int:
    """Display width, counting wide characters as two."""
    total = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _strip_ansi(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def visible_width(s: str) -> int:
    return width(_strip_ansi(s))


def pad(s: str, target: int, align: str = "left") -> str:
    """Pad to `target` display columns, ignoring escape codes."""
    gap = max(0, target - visible_width(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def row(label: str, value: str, label_width: int = 18, colon: bool = True) -> str:
    """`label   : value`, aligned. The shape most of `status` uses."""
    sep = ":" if colon else " "
    return f"  {pad(label, label_width)}{sep} {value}"


def kv_block(pairs, label_width: int = 18) -> list:
    return [row(k, v, label_width) for k, v in pairs]


def rule(char: str = "─", columns: int = 62) -> str:
    return grey(char * columns)


def heading(text: str, columns: int = 62) -> str:
    return f"{bold(text)}  {grey('─' * max(0, columns - len(text) - 2))}"


def section(text: str) -> str:
    return f"\n  {bold(cyan(text))}"


def table(headers, rows, aligns=None, indent=2):
    """A simple left-aligned table. Columns are sized to content."""
    if not rows:
        return []
    aligns = aligns or ["left"] * len(headers)
    cols = len(headers)
    widths = [visible_width(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], visible_width(str(r[i])))

    out = []
    head = " | ".join(pad(bold(str(headers[i])), widths[i], aligns[i])
                      for i in range(cols))
    out.append(" " * indent + head)
    out.append(" " * indent + grey("-+-".join("-" * w for w in widths)))
    for r in rows:
        out.append(" " * indent + " | ".join(
            pad(str(r[i]), widths[i], aligns[i]) for i in range(cols)))
    return out


def bar(fraction: float, columns: int = 20, full: str = "█", empty: str = "░") -> str:
    """A tiny progress bar. Clamped to 0..1."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * columns))
    return full * filled + empty * (columns - filled)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    failures = []

    def check(label, cond):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)

    print("=== term.py self-check ===")
    print()

    # --- NO_COLOR disables ---
    old = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    check("NO_COLOR set -> no colour", supports_color(sys.stdout) is False)
    if old is None:
        del os.environ["NO_COLOR"]
    else:
        os.environ["NO_COLOR"] = old

    # --- TERM=dumb disables ---
    old_term = os.environ.get("TERM")
    os.environ.pop("NO_COLOR", None)
    os.environ["TERM"] = "dumb"
    check("TERM=dumb -> no colour", supports_color(sys.stdout) is False)
    if old_term is None:
        del os.environ["TERM"]
    else:
        os.environ["TERM"] = old_term

    # --- not a TTY disables ---
    class Pipe:
        def isatty(self):
            return False

    os.environ.pop("NO_COLOR", None)
    os.environ.pop("TERM", None) if old_term is None else os.environ.__setitem__("TERM", old_term)
    check("piped output -> no colour", supports_color(Pipe()) is False)

    class Tty:
        def isatty(self):
            return True

    check("a real TTY -> colour", supports_color(Tty()) is True)

    # --- FORCE_COLOR overrides ---
    os.environ["FORCE_COLOR"] = "1"
    check("FORCE_COLOR overrides everything", supports_color(Pipe()) is True)
    del os.environ["FORCE_COLOR"]

    # --- with colour off, the functions are identity ---
    set_enabled(False)
    check("green() is identity when colour is off", green("x") == "x")
    check("bold() is identity when colour is off", bold("x") == "x")
    check("badge() returns the bare word when colour is off", badge("ON", "on") == "ON")

    # --- with colour on, escapes appear AND the word survives ---
    set_enabled(True)
    g = green("x")
    check("green() emits an escape when colour is on", "\033[" in g)
    check("the text is still present inside the escape", "x" in g)
    check("badge keeps its word readable", "ON" in badge("ON", "on"))

    # --- width handles wide characters ---
    check("ascii width", width("abc") == 3)
    check("wide char counts as two", width("中") == 2)
    check("visible_width ignores escapes", visible_width(green("abc")) == 3)

    # --- pad aligns by display width, not by byte length ---
    set_enabled(True)
    coloured = green("abc")
    check("pad on a coloured string reaches the right width",
          visible_width(pad(coloured, 10)) == 10)
    check("pad on wide characters reaches the right width",
          visible_width(pad("中中", 6)) == 6)

    # --- rule never overflows ---
    set_enabled(False)
    check("rule width matches", visible_width(rule()) == 62)
    check("heading width is sane", visible_width(heading("TEST")) <= 62)

    # --- status words survive with colour off, which is the point ---
    set_enabled(False)
    line = f"{mark(True)} {badge('PASS', 'ok')}"
    check("status symbols render without colour", "✓" in line and "PASS" in line)

    # --- table aligns columns ---
    t = table(["A", "BB"], [["x", "y"], ["longer", "z"]])
    check("table returns one line per row plus header and rule", len(t) == 4)
    check("table columns line up",
          visible_width(t[0]) == visible_width(t[3]))

    # --- bar clamps ---
    check("bar clamps below 0", bar(-1.0) == "░" * 20)
    check("bar clamps above 1", bar(2.0) == "█" * 20)
    check("bar half", bar(0.5) == "█" * 10 + "░" * 10)

    set_enabled(None if False else enabled())

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
