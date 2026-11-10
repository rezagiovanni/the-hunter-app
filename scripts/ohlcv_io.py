#!/usr/bin/env python3
"""
ohlcv_io.py — reading and writing OHLCV candle files.

Why this module exists
----------------------
`write_csv` was defined FIVE times, once in each of:

    backfill_deep.py, backfill_stale.py, fetch_ohlcv.py,
    backfill_from_date.py, prune_data.py

with `CSV_COLUMNS` declared four times alongside them. Four of the five wrote
straight over the live file:

    with open(path, "w", newline="") as f:
        ...

That is unsafe for exactly the files this bot depends on. Rewriting one pair
means writing its ENTIRE history - up to ~73,000 bars, several MB - and a
backfill of all 20 pairs runs for tens of minutes. A kill, a power loss or a
full disk part-way through `open(path, "w")` leaves that pair truncated, with no
error anywhere: the bars are simply gone, and the next reader sees a shorter
file that still parses.

Only prune_data.py did it correctly, with a temp file and os.replace(). The
others were one interrupted run away from silent data loss.

Every write now goes through write_csv() here, which:
  * writes to a temp file in the SAME directory (same filesystem, so the rename
    is atomic)
  * fsyncs before renaming, so a power cut cannot leave a renamed-but-empty file
  * os.replace()s over the original, so a reader sees the old file or the
    complete new one, never a partial one
  * preserves CRLF line endings, which every existing CSV in this project uses
  * sorts rows by timestamp, because every caller needs ascending order
"""

import csv
import os
import tempfile

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def read_csv(path):
    """Rows from a candle CSV as a list of dicts with int/float values.

    Returns [] if the file is missing, empty or unreadable - callers treat a
    missing file as "no history yet", not as an error.
    """
    if not os.path.exists(path):
        return []

    rows = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({
                        "timestamp": int(r["timestamp"]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r["volume"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue        # skip a malformed row rather than lose the file
    except OSError:
        return []

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def write_csv(path, rows, crlf=True):
    """Write candle rows to `path` atomically.

    The file is either the previous one or the complete new one. A crash mid-write
    leaves a stray .tmp file, never a truncated CSV.

    crlf=True keeps the existing format: every CSV in this project uses CRLF, and
    a naive text-mode rewrite elsewhere would silently convert all 300+ MB of
    them to LF.
    """
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    terminator = "\r\n" if crlf else "\n"
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                    lineterminator=terminator)
            writer.writeheader()
            writer.writerows(rows_sorted)
            f.flush()
            os.fsync(f.fileno())        # survive a power cut, not just a crash
        os.replace(tmp, path)           # atomic on POSIX, same filesystem
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    import ast
    import glob
    import shutil
    import sys

    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {str(detail)[:300]}")

    td = tempfile.mkdtemp()
    try:
        p = os.path.join(td, "BTCIDR.csv")

        # --- 1. round trip ---
        rows = [
            {"timestamp": 200, "open": 2.0, "high": 3.0, "low": 1.0, "close": 2.5, "volume": 10.0},
            {"timestamp": 100, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 5.0},
        ]
        write_csv(p, rows)
        got = read_csv(p)
        check("round trip preserves every row", len(got) == 2, got)
        check("rows come back sorted ascending",
              [r["timestamp"] for r in got] == [100, 200], got)
        check("values survive as numbers", got[0]["close"] == 1.5, got[0])

        # --- 2. CRLF is the default, matching every existing CSV ---
        raw = open(p, "rb").read()
        check("writes CRLF line endings", b"\r\n" in raw, raw[:60])
        check("header is exactly CSV_COLUMNS",
              raw.split(b"\r\n")[0].decode() == ",".join(CSV_COLUMNS), raw[:80])

        # --- 3. atomic: a failure mid-write must not touch the original ---
        before = open(p, "rb").read()

        class Boom:
            def __init__(self, rows):
                self.rows = rows

            def __iter__(self):
                yield self.rows[0]
                raise RuntimeError("disk on fire")

        try:
            write_csv(p, Boom([{"timestamp": 1, "open": 1, "high": 1, "low": 1,
                                "close": 1, "volume": 1}]))
        except Exception:
            pass
        check("a failed write leaves the original intact",
              open(p, "rb").read() == before)
        check("a failed write leaves no .tmp behind",
              not glob.glob(os.path.join(td, "*.tmp")), glob.glob(os.path.join(td, "*.tmp")))

        # --- 4. an existing file is replaced, not appended to ---
        write_csv(p, rows[:1])
        check("rewriting replaces rather than appends", len(read_csv(p)) == 1)

        # --- 5. missing / malformed input is handled, not raised ---
        check("missing file reads as empty", read_csv(os.path.join(td, "nope.csv")) == [])
        bad = os.path.join(td, "bad.csv")
        with open(bad, "w", newline="") as f:
            f.write("timestamp,open,high,low,close,volume\r\n1,2,3,4,5,6\r\nbroken\r\n")
        check("a malformed row is skipped, the rest survive", len(read_csv(bad)) == 1)

        # --- 6. this module is the ONLY production writer ---
        # Only TOP-LEVEL definitions count. prune_data.py has an indented
        # write_csv inside its self-check - a test fixture for its own logic,
        # not a production writer - and flagging that would be a false positive.
        here = os.path.dirname(os.path.abspath(__file__))
        offenders = []
        for f in glob.glob(os.path.join(here, "*.py")):
            if os.path.basename(f) == "ohlcv_io.py":
                continue
            try:
                tree = ast.parse(open(f).read())
            except SyntaxError:
                continue
            for node in tree.body:                   # top level only
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in ("write_csv", "read_csv"):
                        offenders.append(f"{os.path.basename(f)}:{node.name}")
        check("no other module defines a top-level write_csv/read_csv",
              not offenders, offenders)

        # --- 7. and the production CSV writers actually import it ---
        missing = []
        for name in ("backfill_deep.py", "backfill_stale.py", "fetch_ohlcv.py"):
            fp = os.path.join(here, name)
            if os.path.exists(fp) and "ohlcv_io" not in open(fp).read():
                missing.append(name)
        check("the backfills and fetcher import the shared writer", not missing, missing)

    finally:
        shutil.rmtree(td, ignore_errors=True)

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
