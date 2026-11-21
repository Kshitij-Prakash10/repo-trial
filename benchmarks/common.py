"""Shared helpers of the benchmark runners (no repository code is imported here).

Every runner takes --root (the checkout whose code is scored), --out (a directory) and, where it makes
sense, --limit N (a smoke run that writes <name>_firstN.json instead of <name>.json, so a partial run never
overwrites a full one). Results are deterministic: fixed seeds, sorted keys, and no time stamp other than
the one "generated_at" field. Latency fields (names ending in "_ms" or "_s", or under "latency") vary from
run to run; strip_volatile() removes them for the run-twice comparison.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
VOLATILE_SUFFIXES = ("_ms", "_s", "_seconds")
VOLATILE_KEYS = ("generated_at", "latency", "wall_s")


def limit_threads(n: int = 2) -> None:
    """At most n threads for BLAS / OpenMP / torch (the benchmark machine is shared)."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)


def use_root(root: str) -> str:
    """Put the checkout first on sys.path so `import ocr` / `import scanner` load ITS code. Returns the
    absolute root. Call before importing any repository module."""
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "ocr")) and not os.path.isdir(os.path.join(root, "scanner")):
        raise SystemExit(f"--root {root} is not a checkout of this repository")
    for name in [m for m in sys.modules if m.split(".")[0] in ("ocr", "scanner", "server")]:
        f = os.path.abspath(getattr(sys.modules[name], "__file__", None) or "")
        if not f.startswith(root + os.sep):  # loaded from another checkout: drop it so --root's copy loads
            del sys.modules[name]
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)
    return root


def check_imported_from(module: Any, root: str) -> None:
    path = os.path.abspath(getattr(module, "__file__", "") or "")
    if not path.startswith(root + os.sep):
        raise RuntimeError(f"{module.__name__} was imported from {path}, not from --root {root}")


def git_info(root: str) -> dict:
    """Commit and whether the tracked code under test differs from it (uncommitted edits)."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    dirty = run("status", "--porcelain", "--untracked-files=no", "--", "ocr", "scanner", "server.py")
    return {"commit": run("rev-parse", "HEAD") or None, "code_dirty": bool(dirty),
            "dirty_files": sorted(line[3:] for line in dirty.splitlines())}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frac(num: int, den: int) -> dict:
    """{"num", "den", "rate"} with rate rounded to 4 places (None when den is 0). Never rounds for a
    threshold decision: compare num / den directly."""
    return {"num": int(num), "den": int(den), "rate": round(num / den, 4) if den else None}


def out_name(name: str, limit: int | None) -> str:
    return f"{name}_first{limit}" if limit else name


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float):
        return round(o, 6)
    return o


def write_result(out_dir: str, name: str, result: dict, md: str | None = None) -> str:
    """Write <out_dir>/<name>.json (sorted keys, one generated_at) and optionally <name>.md."""
    os.makedirs(out_dir, exist_ok=True)
    res = _clean(result)
    res["generated_at"] = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    path = os.path.join(out_dir, name + ".json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    if md is not None:
        with open(os.path.join(out_dir, name + ".md"), "w") as f:
            f.write(md.rstrip() + "\n")
    return path


def strip_volatile(o: Any) -> Any:
    """The result without generated_at and latency fields (for comparing two runs)."""
    if isinstance(o, dict):
        return {k: strip_volatile(v) for k, v in o.items()
                if k not in VOLATILE_KEYS and not k.endswith(VOLATILE_SUFFIXES)}
    if isinstance(o, list):
        return [strip_volatile(v) for v in o]
    return o


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    def cell(v):
        if isinstance(v, dict) and "num" in v:
            return f"{v['num']}/{v['den']}" + (f" ({v['rate']:.4f})" if v.get("rate") is not None else "")
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def compare_main(argv: list[str]) -> int:
    """python -m benchmarks.common A.json B.json: exit 0 when equal apart from volatile fields."""
    a, b = (strip_volatile(json.load(open(p))) for p in argv[:2])
    if a == b:
        print("identical (ignoring generated_at and latency fields)")
        return 0
    def diff(x, y, path=""):
        if type(x) is not type(y):
            print("differs at", path or "/")
            return
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    print("key only in one file:", path + "/" + k)
                elif x[k] != y[k]:
                    diff(x[k], y[k], path + "/" + k)
        elif isinstance(x, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                if u != v:
                    diff(u, v, f"{path}[{i}]")
        else:
            print("differs at", path or "/")
    diff(a, b)
    return 1


if __name__ == "__main__":
    raise SystemExit(compare_main(sys.argv[1:]))
