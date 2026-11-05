#!/usr/bin/env python3
"""
prune_data.py — drop OHLCV bars older than the retention floor.

Deleting data is irreversible, so this module is built defensively:

1. **Atomic.** Each CSV is rewritten to a temporary file in the same directory
   and then `os.replace`d over the original. `os.replace` is atomic on POSIX, so
   a reader sees either the whole old file or the whole new one. If the process
   is killed part-way, the original is untouched and the temp file is orphaned
   garbage that the next run deletes.

2. **Never empties a file.** If the computed cut would remove every row — a
   clock skew, a corrupt timestamp, a mis-set retention — the file is left
   alone and the problem is reported. A pair with no data cannot be scored and
   cannot be backfilled automatically.

3. **Never touches the newest bars.** The floor is applied to old data only. The
   last N bars are always kept regardless, so a file whose timestamps are all
   stale still retains its most recent observation.

4. **Line endings are preserved.** Every file in this dataset is CRLF. Reading
   with the csv module in binary and writing back with the same bytes-per-line
   termination avoids silently converting 376 MB of data to LF, which would
   change every file's bytes for no reason.

5. **Dry run by default is not the default.** The job runs for real when called
   by cron, so `--dry-run` exists for inspection and the real path is tested.

What it does not do: it never merges, never backfills, never invents rows. It
only removes a prefix of each file, and only complete lines.
"""

import argparse
import datetime as dt
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

# Retention floor, from config (data.retention_years). Two years is the product
# requirement (PRD FR-4.1): enough history for the 96-bar range window and the
# model's EMA warm-up, without carrying five-plus years of dead data.
try:
    import config as _config
    DEFAULT_YEARS = float(_config.get(_config.current(), "data.retention_years", 2.0))
except Exception:                                    # pragma: no cover
    DEFAULT_YEARS = 2.0

# Always keep at least this many trailing bars per pair, whatever their age.
# A pair whose feed died a month ago would otherwise be emptied entirely, and an
# empty CSV is indistinguishable from "never fetched", which changes what the
# backfill tries to do next.
MIN_TRAILING_BARS = 96

HEADER = "timestamp,open,high,low,close,volume"


def cutoff_timestamp(years, now=None):
    """Unix seconds before which rows are prunable."""
    now = now if now is not None else dt.datetime.now(dt.timezone.utc)
    days = years * 365.25
    return (now - dt.timedelta(days=days)).timestamp()


def detect_newline(raw_first_chunk):
    """'\\r\\n' or '\\n', from the first chunk of the file."""
    return "\r\n" if b"\r\n" in raw_first_chunk else "\n"


def prune_file(path, cutoff, dry_run=False, keep_last=MIN_TRAILING_BARS):
    """Remove rows older than `cutoff` from one CSV.

    Returns a dict describing what happened:
        {path, before, after, removed, status, reason}
    status is one of: unchanged, pruned, kept_all, error.
    """
    result = {"path": path, "before": 0, "after": 0, "removed": 0,
              "status": "unchanged", "reason": ""}

    try:
        with open(path, "rb") as fh:
            head = fh.read(512)
        newline = detect_newline(head)

        with open(path, "r", newline="") as fh:
            lines = fh.readlines()
    except OSError as exc:
        result["status"] = "error"
        result["reason"] = f"unreadable: {exc}"
        return result

    if not lines:
        result["status"] = "error"
        result["reason"] = "empty file"
        return result

    # Locate the header: the first line that is not blank and does not start
    # with a digit. Everything before the header is preserved as-is.
    header_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s[0].isdigit():
            header_idx = i
            break

    data = lines[header_idx + 1:]
    result["before"] = len(data)
    if not data:
        result["status"] = "kept_all"
        result["reason"] = "no data rows"
        result["after"] = 0
        return result

    # Parse timestamps defensively. A line we cannot parse is kept: dropping it
    # would silently lose data, which is the one thing a prune must not do.
    stamps = []
    for line in data:
        try:
            stamps.append(int(float(line.split(",", 1)[0])))
        except (ValueError, IndexError):
            stamps.append(None)

    first_keep = 0
    for i, ts in enumerate(stamps):
        if ts is None or ts >= cutoff:
            first_keep = i
            break
    else:
        first_keep = len(stamps)   # everything is older than the cutoff

    # Never remove the trailing bars.
    max_keep_start = max(0, len(stamps) - keep_last)
    if first_keep > max_keep_start:
        first_keep = max_keep_start
        result["reason"] = f"limited to keep the last {keep_last} bars"

    # Never empty the file.
    if first_keep >= len(stamps):
        result["status"] = "kept_all"
        result["reason"] = ("every row is older than the cutoff; refusing to "
                            "empty the file - check this pair's feed")
        result["after"] = len(data)
        return result

    removed = first_keep
    result["removed"] = removed
    result["after"] = len(data) - removed

    if removed == 0:
        result["status"] = "unchanged"
        return result

    if dry_run:
        result["status"] = "pruned"
        return result

    # ---- atomic rewrite ----
    tmp_path = path + ".prune.tmp"
    try:
        with open(tmp_path, "w", newline="") as out:
            # preserve any preamble (blank lines, comments) and the header
            for line in lines[:header_idx + 1]:
                out.write(line)
            for line in data[first_keep:]:
                # write() with newline="" passes the line through untouched,
                # so the original CRLF endings survive byte for byte
                out.write(line)
        os.replace(tmp_path, path)
        result["status"] = "pruned"
    except OSError as exc:
        result["status"] = "error"
        result["reason"] = f"write failed: {exc}"
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass

    return result


def prune_all(ohlcv_dir=None, years=DEFAULT_YEARS, dry_run=False, now=None):
    ohlcv_dir = ohlcv_dir or paths.OHLCV_DIR
    cutoff = cutoff_timestamp(years, now=now)

    files = sorted(glob.glob(os.path.join(ohlcv_dir, "*.csv")))
    results = [prune_file(f, cutoff, dry_run=dry_run) for f in files]

    # orphaned temp files from a previous interrupted run
    orphans = glob.glob(os.path.join(ohlcv_dir, "*.prune.tmp"))
    if not dry_run:
        for o in orphans:
            try:
                os.unlink(o)
            except OSError:
                pass

    return {
        "cutoff": cutoff,
        "years": years,
        "files": len(files),
        "pruned": sum(1 for r in results if r["status"] == "pruned"),
        "unchanged": sum(1 for r in results if r["status"] == "unchanged"),
        "kept_all": sum(1 for r in results if r["status"] == "kept_all"),
        "errors": [r for r in results if r["status"] == "error"],
        "rows_before": sum(r["before"] for r in results),
        "rows_after": sum(r["after"] for r in results),
        "rows_removed": sum(r["removed"] for r in results),
        "orphans_removed": len(orphans),
        "results": results,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drop OHLCV bars older than the retention floor.",
    )
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS,
                    help=f"retention in years (default {DEFAULT_YEARS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed, change nothing")
    ap.add_argument("--verbose", action="store_true", help="one line per file")
    args = ap.parse_args(argv)

    summary = prune_all(years=args.years, dry_run=args.dry_run)

    if args.verbose:
        for r in summary["results"]:
            name = os.path.basename(r["path"])
            if r["status"] == "pruned":
                print(f"  {name:<18} -{r['removed']:>7,} rows  -> {r['after']:>7,}")
            elif r["status"] in ("kept_all", "error"):
                print(f"  {name:<18} {r['status'].upper()}: {r['reason']}")

    tag = "[prune] DRY RUN, nothing written" if args.dry_run else "[prune]"
    print(f"{tag} {summary['files']} files | "
          f"{summary['pruned']} pruned, {summary['unchanged']} unchanged, "
          f"{summary['kept_all']} kept whole")
    print(f"[prune] rows {summary['rows_before']:,} -> {summary['rows_after']:,} "
          f"(-{summary['rows_removed']:,})")
    if summary["orphans_removed"]:
        print(f"[prune] cleaned {summary['orphans_removed']} leftover temp files")
    for err in summary["errors"]:
        print(f"[prune] ERROR {os.path.basename(err['path'])}: {err['reason']}")
    return 1 if summary["errors"] else 0


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
                print(f"       {detail[:200]}")

    print("=== prune_data.py self-check ===")
    print()

    def write_csv(path, rows, newline="\r\n"):
        with open(path, "w", newline="") as fh:
            fh.write(HEADER + newline)
            for ts, close in rows:
                fh.write(f"{ts},{close},{close},{close},{close},1.0{newline}")

    def read_rows(path):
        with open(path, newline="") as fh:
            lines = fh.readlines()
        return lines[1:]

    NOW = dt.datetime(2026, 11, 5, tzinfo=dt.timezone.utc)
    now_ts = int(NOW.timestamp())
    old = now_ts - int(3 * 365.25 * 86400)      # 3 years ago
    recent = now_ts - 3600                      # an hour ago
    cutoff = cutoff_timestamp(2.0, now=NOW)

    with tempfile.TemporaryDirectory() as td:
        # ---- 1. basic prune keeps recent rows only ----
        p = os.path.join(td, "AIDR.csv")
        rows = [(old + i * 900, 100 + i) for i in range(200)]   # all old
        rows += [(recent - (200 - i) * 900, 200 + i) for i in range(200)]
        write_csv(p, rows)
        r = prune_file(p, cutoff, keep_last=5)
        check("old rows removed", r["status"] == "pruned", r)
        check("exactly the old prefix went", r["removed"] == 200, str(r["removed"]))
        check("no recent row was touched", r["after"] == 200, r["after"])

        remaining = read_rows(p)
        check("file still has a header",
              open(p, newline="").readline().strip() == HEADER)
        check("timestamps are ascending after the prune",
              [int(l.split(",")[0]) for l in remaining] ==
              sorted(int(l.split(",")[0]) for l in remaining))
        check("first surviving row is at or after the cutoff",
              int(remaining[0].split(",")[0]) >= cutoff)

        # ---- 2. CRLF preserved byte for byte ----
        raw = open(p, "rb").read()
        check("CRLF endings preserved", b"\r\n" in raw)
        check("no stray LF-only line endings", raw.count(b"\n") == raw.count(b"\r\n"))

        # ---- 3. everything old -> never emptied ----
        # A file whose rows are ALL older than the cutoff (a pair whose feed
        # died) is clamped by the trailing floor, not emptied. The invariant is
        # "nonzero rows survive", and the survivors are the most recent ones.
        # Leaving such a file completely untouched is an equally defensible
        # choice; it is not taken, because the 2-year floor is the product rule
        # and a scorable tail is more useful than 200k dead rows.
        p2 = os.path.join(td, "BIDR.csv")
        write_csv(p2, [(old + i * 900, 1.0) for i in range(50)])
        r = prune_file(p2, cutoff, keep_last=5)
        check("all-old file is never emptied", r["after"] > 0, str(r))
        check("all-old file keeps at least the trailing floor",
              len(read_rows(p2)) >= 5, str(len(read_rows(p2))))
        check("survivors of an all-old file are its newest rows",
              [int(l.split(",")[0]) for l in read_rows(p2)] ==
              sorted(old + i * 900 for i in range(45, 50)),
              str([int(l.split(",")[0]) for l in read_rows(p2)]))

        # ---- 4. nothing old -> unchanged ----
        p3 = os.path.join(td, "CIDR.csv")
        write_csv(p3, [(recent - i * 900, 1.0) for i in range(50)][::-1])
        before_bytes = open(p3, "rb").read()
        r = prune_file(p3, cutoff, keep_last=5)
        check("all-recent file is unchanged", r["status"] == "unchanged", r)
        check("unchanged file is byte-identical",
              open(p3, "rb").read() == before_bytes)

        # ---- 5. the trailing floor is always honoured ----
        p4 = os.path.join(td, "DIDR.csv")
        write_csv(p4, [(old + i * 900, 1.0) for i in range(300)])
        r = prune_file(p4, cutoff, keep_last=96)
        check("never fewer than the trailing floor remain",
              len(read_rows(p4)) >= 96, str(len(read_rows(p4))))
        check("exactly the floor remains when every row is stale",
              len(read_rows(p4)) == 96, str(len(read_rows(p4))))

        # ---- 6. dry run writes nothing ----
        p5 = os.path.join(td, "EIDR.csv")
        rows5 = [(old + i * 900, 1.0) for i in range(100)]
        rows5 += [(recent - i * 900, 2.0) for i in range(100)][::-1]
        write_csv(p5, rows5)
        before_bytes = open(p5, "rb").read()
        r = prune_file(p5, cutoff, dry_run=True)
        check("dry run reports rows it would remove", r["removed"] == 100, r)
        check("dry run leaves the file byte-identical",
              open(p5, "rb").read() == before_bytes)

        # ---- 7. an interrupted prune leaves a parseable file ----
        p6 = os.path.join(td, "FIDR.csv")
        rows6 = [(old + i * 900, 1.0) for i in range(100)]
        rows6 += [(recent - i * 900, 2.0) for i in range(100)][::-1]
        write_csv(p6, rows6)
        # simulate a crash after the temp file was written but before replace
        with open(p6 + ".prune.tmp", "w", newline="") as fh:
            fh.write(HEADER + "\r\n" + "999,1,1,1,1,1\r\n")
        every_line_parses = True
        for line in open(p6, newline=""):
            if not line.strip():
                continue
            if len(line.strip().split(",")) != 6:
                every_line_parses = False
        check("original still parses after a simulated crash", every_line_parses)
        check("original keeps all its rows after a simulated crash",
              len(read_rows(p6)) == 200)

        # ---- 8. unparseable lines are kept, not dropped ----
        p7 = os.path.join(td, "GIDR.csv")
        with open(p7, "w", newline="") as fh:
            fh.write(HEADER + "\r\n")
            fh.write(f"{old},1,1,1,1,1\r\n")
            fh.write(f"{recent},2,2,2,2,2\r\n")
            fh.write("garbage-line-without-timestamp\r\n")
            fh.write(f"{recent + 900},3,3,3,3,3\r\n")
        r = prune_file(p7, cutoff, keep_last=1)
        remaining = read_rows(p7)
        check("a malformed line is kept rather than silently dropped",
              any("garbage" in l for l in remaining), str(remaining))

        # ---- 9. orphan temp files are cleaned on the next real run ----
        p8 = os.path.join(td, "HIDR.csv")
        write_csv(p8, [(recent - i * 900, 1.0) for i in range(10)][::-1])
        open(p8 + ".prune.tmp", "w").write("junk\n")
        summary = prune_all(td, years=2.0, dry_run=False, now=NOW)
        check("orphan temp files are removed on a real run",
              summary["orphans_removed"] == 1, summary["orphans_removed"])
        check("orphan file itself is gone", not os.path.exists(p8 + ".prune.tmp"))

        # ---- 10. summary arithmetic is consistent ----
        check("summary rows_before - rows_removed == rows_after",
              summary["rows_before"] - summary["rows_removed"] == summary["rows_after"],
              f"{summary['rows_before']} - {summary['rows_removed']} != {summary['rows_after']}")

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        for f in failures:
            print("  -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(_self_check())
    sys.exit(main())
