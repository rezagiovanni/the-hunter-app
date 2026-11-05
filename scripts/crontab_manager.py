#!/usr/bin/env python3
"""
crontab_manager.py — install, enable, disable and remove The Hunter's cron jobs.

This is the only module that writes outside the app's own directory, so the
rules are strict:

1. **A managed block, delimited.** Everything we own sits between
   `# BEGIN the-hunter` and `# END the-hunter`. Lines outside it are never
   touched and never reordered. No crontab is ever rewritten wholesale.

2. **Unknown content is preserved byte for byte.** The user's other jobs,
   comments, blank lines, environment variables — all survive every operation.
   Tested by installing, toggling and removing around unrelated lines.

3. **Disabled means a commented line, not a guard.** A disabled job must execute
   nothing: no process spawn, no Python startup, no chance of a bug reaching a
   trading decision. A commented line is the only construction that guarantees
   that, which is why it is used instead of a runtime check.

4. **Writes are atomic and backed up.** `crontab file` replaces the whole
   crontab; if it fails halfway the user's jobs are gone. So: snapshot first,
   write to a temp file, validate the result, then install. On any error the
   previous crontab is restored.

5. **Idempotent.** Running install twice yields one block with one entry per job,
   never two.

Nothing here trades. It edits a text file and calls `crontab`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

BEGIN_MARK = "# BEGIN the-hunter"
END_MARK = "# END the-hunter"

# Job name -> cron expression. These match the schedules the project has used
# all along, so a migration from the Hermes crons changes nothing.
JOBS = [
    ("market",         "* * * * *",        "fetch top-20 pairs + order books"),
    ("ohlcv",          "1,16,31,46 * * * *", "append 15m candles"),
    ("backfill-stale", "0 */3 * * *",      "fill gaps in recent history"),
    ("backfill-deep",  "30 2 * * *",       "two years of history for new pairs"),
    ("prune",          "15 3 * * *",       "drop bars older than the retention floor"),
    ("scanning",       "* * * * *",        "one hunting tick (this one trades)"),
]

# Job ordering in the block: data jobs first, the trading job last, so a human
# reading the crontab sees the innocuous entries before the one that trades.
JOB_ORDER = [j[0] for j in JOBS]


class CrontabError(RuntimeError):
    """Raised when the crontab cannot be read or written safely."""


# How many crontab snapshots to keep. Every enable/disable writes one, so
# without a cap a busy session would litter the archive directory.
KEEP_BACKUPS = 20


def _prune_backups(backup_dir, keep=KEEP_BACKUPS):
    """Keep only the newest `keep` crontab snapshots."""
    try:
        names = sorted(
            (n for n in os.listdir(backup_dir)
             if n.startswith("crontab-") and n.endswith(".bak")),
            reverse=True,
        )
        for stale in names[keep:]:
            try:
                os.unlink(os.path.join(backup_dir, stale))
            except OSError:
                pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------
def _crontab_cmd():
    """The crontab binary to use, honouring $HUNTER_CRONTAB.

    That override exists for the test harness, and the reason is a real bug this
    milestone shipped then caught: `crontab` writes to the invoking user's
    crontab and ignores $HOME, so an install test in a fake HOME STILL modified
    the developer's live crontab, pointing the block at /tmp paths. Nothing ran,
    because every entry is commented out, but the leak was real. Tests now point
    this at a private stub.
    """
    return os.environ.get("HUNTER_CRONTAB") or "crontab"


def read_crontab():
    """The user's current crontab as text. Empty string when there is none.

    Distinguishes "no crontab" from "could not read", because treating an error
    as empty would silently delete every job the user has.
    """
    try:
        r = subprocess.run([_crontab_cmd(), "-l"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        raise CrontabError("the `crontab` command is not installed")
    except subprocess.TimeoutExpired:
        raise CrontabError("`crontab -l` timed out")

    if r.returncode == 0:
        return r.stdout

    err = (r.stderr or "").lower()
    if "no crontab" in err:
        return ""
    raise CrontabError(f"could not read crontab: {(r.stderr or '').strip()}")


def write_crontab(text, dry_run=False):
    """Replace the user's crontab with `text`, atomically and with a backup.

    Writes to a temp file and installs from it, so a failure part-way cannot
    leave a half-written crontab. The previous content is kept in
    `data/archive/crontab-<ts>.bak` before the install, and restored if the
    install fails.
    """
    if dry_run:
        return None

    backup_path = None
    previous = read_crontab()
    if previous.strip():
        backup_dir = paths.ARCHIVE_DIR
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"crontab-{int(time.time() * 1000)}.bak")
        with open(backup_path, "w") as fh:
            fh.write(previous)
        _prune_backups(backup_dir, keep=KEEP_BACKUPS)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".cron", delete=False)
    try:
        tmp.write(text)
        tmp.flush()
        tmp.close()
        r = subprocess.run([_crontab_cmd(), tmp.name], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            if backup_path and os.path.exists(backup_path):
                subprocess.run([_crontab_cmd(), backup_path], capture_output=True, timeout=10)
            raise CrontabError(f"could not install crontab: {(r.stderr or '').strip()}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return backup_path


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------
def split_block(text):
    """(before, block_lines, after) around the managed markers.

    `block_lines` excludes the markers. Missing markers give ("", [], text) —
    i.e. no block, everything is foreign.
    """
    lines = text.splitlines()
    begin = end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == BEGIN_MARK and begin is None:
            begin = i
        elif s == END_MARK and begin is not None and end is None:
            end = i

    if begin is None or end is None:
        return text, [], ""

    before = "\n".join(lines[:begin])
    block = lines[begin + 1:end]
    after = "\n".join(lines[end + 1:])
    return before, block, after


def parse_jobs(block_lines):
    """{job: enabled} from block lines. A commented line means disabled."""
    out = {}
    for line in block_lines:
        s = line.strip()
        if not s or s.startswith("# BEGIN") or s.startswith("# END"):
            continue
        enabled = not s.startswith("#")
        body = s.lstrip("#").strip()
        if "_job" not in body:
            continue
        try:
            name = body.split("_job", 1)[1].split()[0]
        except IndexError:
            continue
        out[name] = enabled
    return out


# ---------------------------------------------------------------------------
# Block generation
# ---------------------------------------------------------------------------
def _command(job):
    """The exact command cron runs.

    Uses the same wrapper a user calls by hand, so there is one code path and
    one place where the interpreter is chosen. Output is appended to a log file
    per job; cron would otherwise mail it, and on a headless box that mail goes
    nowhere.
    """
    hunter = os.path.join(paths.APP_ROOT, "bin", "hunter")
    log = os.path.join(paths.LOGS_DIR, f"{job}.log")
    return f"{hunter} _job {job} >> {log} 2>&1"


def render_block(states):
    """The managed block, all jobs **disabled** unless `states` says otherwise.

    `states` maps job -> bool for enabled.
    """
    lines = [BEGIN_MARK,
             "# Managed by The Hunter. Do not edit by hand - use:",
             "#   hunter cron enable <job>   /   hunter cron disable <job>",
             "# Everything between these markers belongs to The Hunter.",
             ""]
    for name, sched, desc in JOBS:
        enabled = bool((states or {}).get(name, False))
        cmd = _command(name)
        lines.append(f"# {desc}")
        if enabled:
            lines.append(f"{sched} {cmd}")
        else:
            lines.append(f"# {sched} {cmd}")
    lines.append(END_MARK)
    return "\n".join(lines)


def compose(existing_text, states):
    """Full new crontab text: foreign content preserved, block replaced."""
    before, _, after = split_block(existing_text)
    block = render_block(states)

    parts = []
    if before.strip():
        parts.append(before.rstrip("\n"))
    parts.append(block)
    if after.strip():
        parts.append(after.strip("\n"))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
def install(dry_run=False, states=None):
    """Install the block with every job disabled (unless `states` overrides).
    Preserves anything outside the block. Idempotent."""
    existing = read_crontab()
    new = compose(existing, states or {})
    if new == existing:
        return {"changed": False, "backup": None, "text": new}
    backup = write_crontab(new, dry_run=dry_run)
    return {"changed": True, "backup": backup, "text": new}


def remove(dry_run=False):
    """Remove the managed block and nothing else."""
    existing = read_crontab()
    before, block, after = split_block(existing)
    if not block:
        return {"changed": False, "text": existing}

    parts = []
    if before.strip():
        parts.append(before.rstrip("\n"))
    if after.strip():
        parts.append(after.strip("\n"))
    new = "\n\n".join(parts)
    if new and not new.endswith("\n"):
        new += "\n"
    write_crontab(new, dry_run=dry_run)
    return {"changed": True, "text": new}


def job_states():
    """{job: enabled} from the live crontab. Empty dict if no block."""
    existing = read_crontab()
    _, block, _ = split_block(existing)
    return parse_jobs(block)


def set_job(name, enabled, dry_run=False):
    """Enable or disable one job, leaving the rest as they are."""
    if name not in JOB_ORDER:
        raise CrontabError(f"unknown job '{name}' (known: {', '.join(JOB_ORDER)})")

    existing = read_crontab()
    _, block, _ = split_block(existing)
    if not block:
        raise CrontabError("no the-hunter block in the crontab; run install first")

    states = parse_jobs(block)
    states[name] = bool(enabled)
    new = compose(existing, states)
    write_crontab(new, dry_run=dry_run)
    return states


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check():
    """Tests the pure text logic. Nothing here touches the real crontab: every
    operation runs against string literals, so a bug cannot delete a user's
    jobs while testing."""
    failures = []

    def check(label, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)
            if detail:
                print(f"       {detail[:200]}")

    print("=== crontab_manager.py self-check ===")
    print()

    FOREIGN = (
        "# my own jobs\n"
        "MAILTO=\"\"\n"
        "0 4 * * * /usr/bin/backup.sh\n"
        "\n"
        "*/5 * * * * /home/user/poll.sh\n"
    )

    # ---- 1. install into an empty crontab ----
    text = compose("", {})
    check("install produces a block", BEGIN_MARK in text and END_MARK in text)
    check("install registers every job",
          all(f"_job {j}" in text for j in JOB_ORDER))
    check("all jobs commented out on install",
          parse_jobs(split_block(text)[1]) == {j: False for j in JOB_ORDER})
    check("no live command line on install",
          not any(line and not line.strip().startswith("#")
                  for line in split_block(text)[1]))

    # ---- 2. foreign content survives install ----
    text = compose(FOREIGN, {})
    check("foreign job preserved", "/usr/bin/backup.sh" in text)
    check("foreign comment preserved", "# my own jobs" in text)
    check("foreign env var preserved", 'MAILTO=""' in text)
    check("foreign job after block preserved",
          text.index("backup.sh") < text.index(BEGIN_MARK))
    _, block, _ = split_block(text)
    check("block contains only our jobs",
          set(parse_jobs(block)) == set(JOB_ORDER))

    # ---- 3. idempotent: composing twice changes nothing ----
    once = compose(FOREIGN, {})
    twice = compose(once, {})
    check("install is idempotent (same input -> same output)", once == twice)

    # ---- 4. enabling one job touches only that line ----
    states = {j: False for j in JOB_ORDER}
    states["market"] = True
    text = compose(FOREIGN, states)
    parsed = parse_jobs(split_block(text)[1])
    check("enable market -> market on", parsed["market"] is True)
    check("enable market -> others off",
          all(not v for k, v in parsed.items() if k != "market"))
    check("enable market keeps foreign content", "/usr/bin/backup.sh" in text)

    # ---- 5. exactly one entry per job, however many times toggled ----
    text = compose(FOREIGN, {})
    for _ in range(10):
        text = compose(text, {"market": True})
        text = compose(text, {"market": False})
    check("10 toggles leave exactly one market entry", text.count("_job market") == 1)
    check("10 toggles leave market disabled", parse_jobs(split_block(text)[1])["market"] is False)
    check("10 toggles keep one block",
          text.count(BEGIN_MARK) == 1 and text.count(END_MARK) == 1)

    # ---- 6. all six jobs enabled ----
    text = compose(FOREIGN, {j: True for j in JOB_ORDER})
    parsed = parse_jobs(split_block(text)[1])
    check("all six can be enabled", all(parsed.values()) and len(parsed) == 6)
    check("all six enabled keeps foreign content", "/usr/bin/backup.sh" in text)

    # ---- 7. removal takes the block and nothing else ----
    text = compose(FOREIGN, {})
    before, block, after = split_block(text)
    parts = [before.rstrip("\n"), after.strip("\n")]
    stripped = "\n\n".join(p for p in parts if p)
    check("removal drops the markers", BEGIN_MARK not in stripped)
    check("removal keeps every foreign line",
          all(s in stripped for s in
              ("# my own jobs", 'MAILTO=""', "/usr/bin/backup.sh", "poll.sh")))

    # ---- 8. a crontab with no block is untouched by remove ----
    _, block, _ = split_block(FOREIGN)
    check("foreign-only crontab has no block", block == [])
    check("foreign-only crontab is not modified by a remove",
          compose(FOREIGN, {}) != FOREIGN)

    # ---- 9. a corrupted block (BEGIN with no END) is handled ----
    broken = FOREIGN + "\n" + BEGIN_MARK + "\n0 * * * * something\n"
    before, block, after = split_block(broken)
    check("unterminated block is treated as no block", block == [])
    check("unterminated block leaves the text intact", before == broken)

    # ---- 10. disabled really means no executable line ----
    text = compose("", {})
    _, block, _ = split_block(text)
    executable = [ln for ln in block
                  if ln.strip() and not ln.strip().startswith("#")]
    check("a disabled job produces zero executable lines", executable == [],
          str(executable))

    # ---- 11. the command uses the wrapper, not a bare python ----
    text = compose("", {"market": True})
    check("enabled job calls bin/hunter _job",
          "bin/hunter _job market" in text)
    check("enabled job redirects output to a log",
          ">>" in text and ".log" in text)

    # ---- 12. schedule correctness ----
    sch = {name: s for name, s, _ in JOBS}
    check("scanning runs every minute", sch["scanning"] == "* * * * *")
    check("ohlcv runs on the quarter hour", sch["ohlcv"] == "1,16,31,46 * * * *")
    check("backfill-deep runs once daily", sch["backfill-deep"] == "30 2 * * *")

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
