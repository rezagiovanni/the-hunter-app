#!/usr/bin/env python3
"""
setup_watch.py — run one setup job and record its real outcome.

Called as:  setup_watch.py <action> <script_path> <log_path>

Architecture, and why it is this way
------------------------------------
The dashboard must return immediately; a deep backfill runs for tens of minutes.
So the dashboard starts THIS watcher, detached, and forgets it. The watcher is
then the parent of the actual job, which is what makes a trustworthy verdict
possible at all:

    dashboard  ->  setup_watch.py  ->  fetch_market.py
    (returns)      (waits, records)   (does the work)

Two earlier designs failed, both for the same underlying reason - guessing the
outcome instead of observing it:

  1. The job was the dashboard's direct child and the dashboard never wait()ed,
     so the finished job became a ZOMBIE. os.kill(pid, 0) succeeds on a zombie,
     so the action looked alive forever and, since a heavy action holds the
     lock, every other action was blocked behind it.

  2. The watcher polled the process table for a job that was NOT its child. It
     could see the job disappear but could never obtain its wait status, so it
     fell back to scanning the merged log for words like "failed" - which
     reported a successful fetch as failed, because job stdout and stderr share
     one file and normal status lines contain those words.

Being the parent removes the guesswork: subprocess.wait() returns the true exit
code.

Failure is always recorded. If this watcher crashes, the action stays 'running'
and the dashboard's reconcile step turns it into 'failed' with a reason, so the
failure mode is honest rather than silent.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Generous ceiling: a two-year backfill of 20 pairs will not approach this.
MAX_JOB_SECONDS = 6 * 3600


def main():
    if len(sys.argv) != 4:
        print("usage: setup_watch.py <action> <script_path> <log_path>",
              file=sys.stderr)
        return 2

    name, script, log_path = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        import setup_state
    except Exception as exc:
        print(f"setup_watch: cannot import setup_state: {exc}", file=sys.stderr)
        return 2

    if not os.path.exists(script):
        setup_state.record_exit(name, 2, reason=f"missing script {script}")
        return 0

    started = time.time()
    try:
        with open(log_path, "a") as log:
            log.write(f"\n--- watcher starting {os.path.basename(script)} "
                      f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            proc = subprocess.Popen(
                [setup_state.engine_python(), script],
                cwd=str(setup_state.paths.APP_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            # This process is the parent, so it can take the real exit status.
            try:
                exit_code = proc.wait(timeout=MAX_JOB_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                setup_state.record_exit(
                    name, 124, reason=f"timed out after {MAX_JOB_SECONDS}s")
                return 0
    except OSError as exc:
        setup_state.record_exit(name, 2, reason=f"could not run: {exc}")
        return 0

    took = time.time() - started
    reason = (f"completed in {took:.0f}s" if exit_code == 0
              else f"exit code {exit_code} after {took:.0f}s")
    setup_state.record_exit(name, exit_code, reason=reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
