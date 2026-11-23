"""Print the layout of a manifest file: ``python -m pa4t1.inspect_manifests <path> [...]``."""
from __future__ import annotations

import sys

from .manifests import load_raw_manifest


def describe(path: str) -> None:
    records, meta = load_raw_manifest(path)
    print(f"== {path}\n  records: {len(records)}\n  meta keys: {list(meta)[:20]}")
    if records:
        r = records[0]
        if isinstance(r, dict):
            print("  first record fields:", {k: (type(v).__name__, len(v) if hasattr(v, '__len__') else v) for k, v in r.items()})
        else:
            print("  first record:", r, "| type:", type(r).__name__)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
