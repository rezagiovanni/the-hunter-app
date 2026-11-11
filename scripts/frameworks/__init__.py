#!/usr/bin/env python3
"""
frameworks — the strategy registry.

The engine never imports filters.py or exit_strategy.py directly. It asks
this package for the active framework and calls three methods on it.

Adding a strategy:
    1. create frameworks/framework_c.py subclassing Framework
    2. add it to _REGISTRY below
    Nothing else changes. No edits to run_hunter.py or dashboard.py.

Unknown names fall back to A rather than raising: a corrupt or stale control
file must never leave the engine unable to decide what to do.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frameworks.base import Framework                  # noqa: F401  (re-export)
from frameworks.framework_a import FrameworkA
from frameworks.framework_b import FrameworkB
from frameworks.framework_c import FrameworkC
from frameworks.framework_d import FrameworkD

DEFAULT_FRAMEWORK = "A"

_INSTANCES = (FrameworkA(), FrameworkB(), FrameworkC(), FrameworkD())
_REGISTRY = {f.name: f for f in _INSTANCES}


def get_framework(name: str) -> Framework:
    """Return the framework for `name`, falling back to A."""
    key = (name or DEFAULT_FRAMEWORK).upper()
    return _REGISTRY.get(key, _REGISTRY[DEFAULT_FRAMEWORK])


def framework_exists(name: str) -> bool:
    return (name or "").upper() in _REGISTRY


def list_frameworks() -> list:
    """Metadata for the dashboard picker."""
    return [f.describe() for f in _INSTANCES]


def all_names() -> list:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Self-check (run: python3 scripts/frameworks/__init__.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== frameworks registry self-check ===\n")

    # 1. Both frameworks registered
    names = all_names()
    assert all(x in names for x in ("A", "B", "C", "D")), names
    print(f"[PASS] registry contains: {names}")

    # 2. Lookup works, case-insensitive
    assert get_framework("A").name == "A"
    assert get_framework("a").name == "A"
    assert get_framework("b").name == "B"
    assert get_framework("c").name == "C"
    assert get_framework("d").name == "D"
    print("[PASS] lookup is case-insensitive: 'a'/'A'/'b'")

    # 3. Unknown -> falls back to A (never raises)
    assert get_framework("ZZZ").name == "A"
    assert get_framework(None).name == "A"
    assert get_framework("").name == "A"
    print("[PASS] unknown/None/empty -> falls back to A")

    # 4. framework_exists()
    assert framework_exists("A") and framework_exists("b")
    assert not framework_exists("Z")
    print("[PASS] framework_exists() distinguishes real from unknown")

    # 5. Every framework implements the full interface
    for f in _INSTANCES:
        for meth in ("evaluate_entry", "initialize_exit", "evaluate_exit", "describe"):
            assert callable(getattr(f, meth, None)), f"{f.name} missing {meth}"
        lv = f.initialize_exit(1000.0, 1.0)
        assert lv.get("framework") == f.name, f"{f.name} did not stamp framework"
        assert "target_price" in lv and "stop_trigger_price" in lv, lv
        assert "target_pct" in lv and "stop_pct" in lv, lv
    print("[PASS] interface complete: all frameworks implement 3 methods + stamp name")

    # 6. Frameworks are genuinely different (not copy-paste)
    a = get_framework("A").initialize_exit(1000.0, 1.0)
    b = get_framework("B").initialize_exit(1000.0, 1.0)
    assert a["target_pct"] != b["target_pct"], (a, b)
    assert a["stop_pct"] != b["stop_pct"], (a, b)
    c = get_framework("C").initialize_exit(1000.0, 1.0)
    assert c["target_pct"] == 0.05 and c["stop_pct"] == 0.10, c
    print(f"[PASS] A and B differ: A {a['target_pct']:+.3f}/{a['stop_pct']:+.3f} "
          f"vs B {b['target_pct']:+.3f}/{b['stop_pct']:+.3f}")

    # 7. list_frameworks() shape for the API
    listing = list_frameworks()
    assert len(listing) == 4
    for item in listing:
        for k in ("name", "label", "description", "version", "experimental"):
            assert k in item, (k, item)
    assert sum(1 for x in listing if x["experimental"]) == 3, listing
    print(f"[PASS] list_frameworks() -> {[x['name'] for x in listing]} "
          f"({sum(1 for x in listing if x['experimental'])} experimental)")

    print("\n=== ALL CHECKS PASSED ===")
