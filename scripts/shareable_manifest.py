#!/usr/bin/env python3
"""Checksums of the datasets that are stored in the repository with Git LFS.

    python scripts/shareable_manifest.py --write     # record what is on disk in data/shareable.sha256.json
    python scripts/shareable_manifest.py             # verify the files on disk against that record

Covers the GNHK archive, the handwriting fonts, and the SROIE and CORD images and annotations. A
folder's hash is the SHA-256 of its sorted "relative path, file SHA-256" lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "shareable.sha256.json")
ITEMS = {"gnhk": "data/raw/gnhk/GNHK-dataset.zip", "fonts": "data/raw/fonts", "sroie": "data/sroie", "cord": "data/cord"}
SKIP = ("licences",)  # plain-text licence files, tracked by git itself


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def measure(rel: str) -> dict:
    p = os.path.join(ROOT, rel)
    if os.path.isfile(p):
        return {"files": 1, "bytes": os.path.getsize(p), "sha256": sha(p)}
    lines, size = [], 0
    for d, dirs, files in os.walk(p):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        for f in sorted(files):
            if f.startswith("words_"):  # cache derived by ocr/data.py, rebuilt on first use
                continue
            full = os.path.join(d, f)
            lines.append(f"{os.path.relpath(full, p).replace(os.sep, '/')},{sha(full)}")
            size += os.path.getsize(full)
    return {"files": len(lines), "bytes": size, "sha256": hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    now = {k: {"path": v, **measure(v)} for k, v in ITEMS.items() if os.path.exists(os.path.join(ROOT, v))}
    if a.write:
        with open(OUT, "w") as f:
            json.dump(now, f, indent=1)
        print(json.dumps(now, indent=1))
        sys.exit(0)
    with open(OUT) as f:
        want = json.load(f)
    bad = [k for k in want if now.get(k, {}).get("sha256") != want[k]["sha256"]]
    for k in want:
        print(k, "MISSING" if k not in now else ("ok" if k not in bad else f"DIFFERENT ({now[k]['files']} files, expected {want[k]['files']})"))
    sys.exit(1 if bad else 0)
