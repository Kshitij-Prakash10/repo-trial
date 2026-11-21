"""Medication-warning benchmark: the "Check the numbers" flags of one checkout on labelled invented lines.

usage: python -m benchmarks.run_warnings --root <checkout> --out <dir> [--limit N] [--fixtures F]

The code under test is imported from --root: ocr.scan.confirm_numbers (the v2 line flag), ocr.common
(has_number_or_unit) and ocr.api._line (what /api/ocr returns per line: confirm_numbers is the prominent
"Check the numbers" warning in js/ocr.js, has_numbers the quieter one). No model is loaded.

Each fixture line is turned into the line dict the pipeline would produce:
  v2: {"text", "box", "confidence": 0.9, "script", "handwriting_text"?, "printed_text"?}, then the adapter
      below sets "confirm_numbers", "low_confidence" and "has_numbers" the way Scanner.run does with every
      stage on (routing known), and _line() gives the API line;
  v1: {"text", "box", "confidence"} straight into _line() (the v1 pipeline sets no flags).
The opt-in misread-digit option (Scanner(flag_misread_digits=True) -> confirm_numbers(...,
misread_digits=True) at 0c71106) is scored both off (the default) and on. Keyword arguments are detected
with inspect.signature, so the runner works with the baseline and with later interfaces; a mode whose
option does not exist in --root is reported as unavailable instead of being guessed.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import common as C  # noqa: E402
from benchmarks.fields import parse_fields  # noqa: E402

FIXTURES = os.path.join(C.HERE, "warnings", "fixtures.jsonl")
THRESHOLD = 0.95  # the original target: >= 95 % of lines that truly hold a number are flagged by either flag
SETS = ("original_39", "extended", "extended_r2", "extended_r3", "extended_r4", "all", "all_with_r2", "all_with_r3",
        "all_with_r4")
# "all" is the round-1 sets (original_39 + extended), as in the round-1 results; "all_with_r2" adds extended_r2
# (and nothing later, so it stays comparable with round-2 results); "all_with_r3" adds extended_r3;
# "all_with_r4" is every record
ROUND1 = ("original_39", "extended")
ROUND2 = ROUND1 + ("extended_r2",)
ROUND3 = ROUND2 + ("extended_r3",)
PATHS = ("v1", "v2", "all")
PRINTED_ROUTINGS = ("printed_only", "none_v1")


def load_fixtures(path: str = FIXTURES) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Adapter:
    """Builds the pipeline's line dict and its flags from a fixture record, for one checkout."""

    def __init__(self, oscan, ocommon, oapi):
        self.oscan, self.ocommon, self.oapi = oscan, ocommon, oapi
        self.confirm_params = list(inspect.signature(oscan.confirm_numbers).parameters)
        self.scanner_params = list(inspect.signature(oscan.Scanner.__init__).parameters)
        self.line_params = list(inspect.signature(oapi._line).parameters)
        # the option's keyword: "misread_digits" at 0c71106, "flag_misread_digits" from 01d993e (old name kept)
        self.misread_kw = next((k for k in ("flag_misread_digits", "misread_digits") if k in self.confirm_params), None)
        self.misread_available = self.misread_kw is not None
        src = inspect.getsource(oscan.Scanner.run)
        # the adapter mirrors this call in Scanner.run; recorded so a later change is visible
        self.scanner_run_call_matches = any(f"confirm_numbers(ln, script_known=routed, {k}=misread)" in src
                                            for k in ("misread_digits", "flag_misread_digits"))

    def describe(self) -> dict:
        return {"confirm_numbers_params": self.confirm_params, "scanner_init_params": self.scanner_params,
                "api_line_params": self.line_params, "misread_option_available": self.misread_available,
                "misread_option_keyword": self.misread_kw,
                "scanner_run_call_matches_adapter": self.scanner_run_call_matches}

    def v2_line(self, r: dict, misread: bool) -> tuple[dict, bool]:
        ln = {"text": r["displayed_text"], "box": [0, 0, 10, 10], "confidence": 0.9,
              "script": "handwritten" if r["routing"] == "handwriting_selected" else "printed"}
        for k in ("handwriting_text", "printed_text"):
            if k in r:
                ln[k] = r[k]
        kw = {}
        if "script_known" in self.confirm_params:
            kw["script_known"] = True  # Scanner.run: routed = clf and hw loaded (the full v2 pipeline)
        if misread:
            kw[self.misread_kw] = True
        flag = bool(self.oscan.confirm_numbers(dict(ln), **kw))
        ln["low_confidence"] = False
        ln["confirm_numbers"] = flag
        ln["has_numbers"] = bool(self.ocommon.has_number_or_unit(ln["text"]))
        return ln, flag

    def evaluate(self, r: dict, misread: bool) -> dict:
        if r["path"] == "v2":
            ln, scan_flag = self.v2_line(r, misread)
        else:
            ln, scan_flag = {"text": r["displayed_text"], "box": [0, 0, 10, 10], "confidence": 0.9}, None
        api = self.oapi._line(dict(ln))
        prominent = bool(api.get("confirm_numbers"))
        quiet = bool(api.get("has_numbers", False))
        other = {k: v for k, v in sorted(api.items()) if isinstance(v, bool)
                 and k not in ("confirm_numbers", "has_numbers", "low_confidence")}
        fields_ok = parse_fields(r["displayed_text"]) == r["expected_fields"]
        return {"id": r["id"], "set": r["set"], "path": r["path"], "routing": r["routing"], "category": r["category"],
                "displayed_text": r["displayed_text"], "scan_flag": scan_flag, "prominent": prominent,
                "has_numbers": quiet, "any_flag": prominent or quiet, "fields_ok": fields_ok,
                "other_api_flags": other or None}


def _in_set(r_set: str, s: str) -> bool:
    return (s == "all_with_r4" or (s == "all_with_r3" and r_set in ROUND3) or (s == "all_with_r2" and r_set in ROUND2)
            or (s == "all" and r_set in ROUND1) or r_set == s)


def _ids(rows, cond):
    return [f"{r['id']}: {r['displayed_text']}" for r in rows if cond(r)]


def pr(rows, truth_key, recs) -> dict:
    t = {r["id"]: recs[r["id"]][truth_key] for r in rows}
    tp = sum(r["prominent"] and t[r["id"]] for r in rows)
    flagged = sum(r["prominent"] for r in rows)
    pos = sum(t.values())
    return {"precision": C.frac(tp, flagged), "recall": C.frac(tp, pos),
            "missed (truth, no prominent warning)": _ids(rows, lambda r: t[r["id"]] and not r["prominent"]),
            "false (prominent warning, truth false)": _ids(rows, lambda r: r["prominent"] and not t[r["id"]])}


def summarise(rows: list[dict], recs: dict) -> dict:
    num = lambda r: recs[r["id"]]["has_numeric_content"]  # noqa: E731
    pos = [r for r in rows if num(r)]
    neg = [r for r in rows if not num(r)]
    ordinary = [r for r in rows if recs[r["id"]]["routing"] in PRINTED_ROUTINGS
                and recs[r["id"]]["source_script"] == "printed"
                and recs[r["id"]]["displayed_text"] == recs[r["id"]]["ground_truth"]]
    with_fields = [r for r in rows if recs[r["id"]]["expected_fields"]]
    dose_err = [r for r in rows if recs[r["id"]]["dose_incorrect_or_lost"]]
    return {
        "n": len(rows),
        "a_numeric_detection_any_flag": {
            "recall": C.frac(sum(r["any_flag"] for r in pos), len(pos)),
            "false_flag_rate_on_lines_without_numbers": C.frac(sum(r["any_flag"] for r in neg), len(neg)),
            "missed": _ids(pos, lambda r: not r["any_flag"]), "false": _ids(neg, lambda r: r["any_flag"])},
        "b_prominent_vs_dose_incorrect_or_lost": pr(rows, "dose_incorrect_or_lost", recs),
        "b_prominent_vs_warning_warranted": pr(rows, "warning_warranted", recs),
        "c_false_prominent_on_ordinary_print": {
            "rate": C.frac(sum(r["prominent"] for r in ordinary), len(ordinary)),
            "with_numbers": C.frac(sum(r["prominent"] for r in ordinary if num(r)), sum(num(r) for r in ordinary)),
            "without_numbers": C.frac(sum(r["prominent"] for r in ordinary if not num(r)),
                                      sum(not num(r) for r in ordinary)),
            "lines": _ids(ordinary, lambda r: r["prominent"])},
        "d_field_agreement": {
            "exact_drug_value_unit": C.frac(sum(r["fields_ok"] for r in with_fields), len(with_fields)),
            "disagreements": _ids(with_fields, lambda r: not r["fields_ok"]),
            "silent_dose_errors (dose wrong or lost, no prominent warning)": C.frac(
                sum(not r["prominent"] for r in dose_err), len(dose_err)),
            "silent_dose_error_lines": _ids(dose_err, lambda r: not r["prominent"])},
        "by_category_prominent": {c: C.frac(sum(r["prominent"] for r in rows if r["category"] == c),
                                            sum(r["category"] == c for r in rows))
                                  for c in sorted({r["category"] for r in rows})},
    }


def by_category(rows: list[dict], recs: dict) -> dict:
    """Per category (all sets, both paths): prominent warnings on lines whose dose is wrong or lost, on lines
    where a warning is warranted, and on lines where it is not; any flag on lines with and without numbers."""
    out = {}
    for c in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == c]
        g = lambda k: [r for r in sub if recs[r["id"]][k]]  # noqa: E731
        ng = lambda k: [r for r in sub if not recs[r["id"]][k]]  # noqa: E731
        out[c] = {"set": sorted({r["set"] for r in sub}), "paths": sorted({r["path"] for r in sub}), "n": len(sub),
                  "prominent_on_dose_wrong": C.frac(sum(r["prominent"] for r in g("dose_incorrect_or_lost")),
                                                    len(g("dose_incorrect_or_lost"))),
                  "prominent_on_warranted": C.frac(sum(r["prominent"] for r in g("warning_warranted")),
                                                   len(g("warning_warranted"))),
                  "prominent_on_not_warranted": C.frac(sum(r["prominent"] for r in ng("warning_warranted")),
                                                       len(ng("warning_warranted"))),
                  "any_flag_on_numeric": C.frac(sum(r["any_flag"] for r in g("has_numeric_content")),
                                                len(g("has_numeric_content"))),
                  "any_flag_on_non_numeric": C.frac(sum(r["any_flag"] for r in ng("has_numeric_content")),
                                                    len(ng("has_numeric_content"))),
                  "false_prominent": _ids(sub, lambda r: r["prominent"] and not recs[r["id"]]["warning_warranted"]),
                  "missed_prominent": _ids(sub, lambda r: not r["prominent"] and recs[r["id"]]["warning_warranted"])}
    return out


def target(rows: list[dict], recs: dict) -> dict:
    sub = [r for r in rows if r["set"] == "original_39" and recs[r["id"]]["has_numeric_content"]]
    k = sum(r["any_flag"] for r in sub)
    ok = len(sub) > 0 and k >= THRESHOLD * len(sub)  # exact comparison, no rounding (37/39 = 0.9487 < 0.95)
    return {"definition": "recall of 'flagged by either flag' (API confirm_numbers or has_numbers) on the "
                          "original_39 lines that truly hold a number; threshold >= 0.95, not rounded",
            "recall": C.frac(k, len(sub)), "threshold": ">= 0.95", "result": "PASS" if ok else "FAIL",
            "missed": _ids(sub, lambda r: not r["any_flag"])}


def run(root: str, fixtures: str = FIXTURES, limit: int | None = None) -> dict:
    root = C.use_root(root)
    from ocr import api as oapi  # noqa: E402
    from ocr import common as ocommon  # noqa: E402
    from ocr import scan as oscan  # noqa: E402  (imports torch and docTR: needs the full environment)
    for m in (oapi, ocommon, oscan):
        C.check_imported_from(m, root)
    recs_list = load_fixtures(fixtures)
    if limit:
        recs_list = recs_list[:limit]
    recs = {r["id"]: r for r in recs_list}
    ad = Adapter(oscan, ocommon, oapi)
    out = {"benchmark": "warnings", "root": root, "git": C.git_info(root), "fixtures": os.path.relpath(fixtures, C.HERE),
           "fixtures_sha256": C.file_sha256(fixtures), "n_records": len(recs_list), "interface": ad.describe(),
           "adapter_note": "v2 flags are set as Scanner.run does with routing on: confirm_numbers(ln, "
                           "script_known=True[, misread_digits=True]) then has_numbers = has_number_or_unit(text); "
                           "the API line is ocr.api._line(ln).",
           "modes": {}}
    for mode, misread in (("misread_option_off (default)", False), ("misread_option_on", True)):
        if misread and not ad.misread_available:
            out["modes"][mode] = {"available": False, "reason": "confirm_numbers has no misread_digits parameter"}
            continue
        rows = [ad.evaluate(r, misread) for r in recs_list]
        m = {"available": True, "target_original_39": target(rows, recs), "splits": {}, "rows": rows,
             "by_category": by_category(rows, recs)}
        for s in SETS:
            for p in PATHS:
                sub = [r for r in rows if _in_set(r["set"], s) and (p == "all" or r["path"] == p)]
                if sub:
                    m["splits"][f"set={s} path={p}"] = summarise(sub, recs)
        out["modes"][mode] = m
    return out


def markdown(res: dict) -> str:
    L = [f"# Medication-warning benchmark: {res['git'].get('commit') or res['root']}", "",
         f"{res['n_records']} invented lines (`{res['fixtures']}`, sha256 `{res['fixtures_sha256'][:12]}`). "
         "Prominent warning = API `confirm_numbers`; any flag = `confirm_numbers` or `has_numbers`. "
         "`set=all` is the round-1 sets (original_39 + extended); `set=all_with_r2` adds extended_r2; "
         "`set=all_with_r3` adds extended_r3; `set=all_with_r4` is every record.", ""]
    for mode, m in res["modes"].items():
        L += [f"## {mode}", ""]
        if not m.get("available"):
            L += [f"unavailable: {m['reason']}", ""]
            continue
        t = m["target_original_39"]
        L += [f"**Original target** ({t['threshold']}): {t['recall']['num']}/{t['recall']['den']} = "
              f"{t['recall']['rate']} -> **{t['result']}**; missed: {t['missed']}", ""]
        rows = []
        for k, s in m["splits"].items():
            rows.append([k, s["a_numeric_detection_any_flag"]["recall"],
                         s["a_numeric_detection_any_flag"]["false_flag_rate_on_lines_without_numbers"],
                         s["b_prominent_vs_dose_incorrect_or_lost"]["precision"],
                         s["b_prominent_vs_dose_incorrect_or_lost"]["recall"],
                         s["b_prominent_vs_warning_warranted"]["precision"], s["b_prominent_vs_warning_warranted"]["recall"],
                         s["c_false_prominent_on_ordinary_print"]["rate"], s["d_field_agreement"]["exact_drug_value_unit"],
                         s["d_field_agreement"]["silent_dose_errors (dose wrong or lost, no prominent warning)"]])
        L += [C.md_table(["split", "(a) any-flag recall", "(a) any-flag on no-number lines",
                          "(b) P vs dose wrong", "(b) R vs dose wrong", "(b) P vs warranted", "(b) R vs warranted",
                          "(c) prominent on correct print", "(d) exact fields", "silent dose errors"], rows), ""]
        allk = "set=all path=all"
        L += ["Silent dose errors (round-1 sets): " + "; ".join(
            m["splits"][allk]["d_field_agreement"]["silent_dose_error_lines"]), ""]
        L += [f"### Per category ({mode})", "",
              C.md_table(["category", "set", "paths", "prominent on dose wrong", "prominent on warranted",
                          "prominent on not warranted", "any flag on numeric", "any flag on non-numeric"],
                         [[c, ",".join(v["set"]), ",".join(v["paths"]), v["prominent_on_dose_wrong"],
                           v["prominent_on_warranted"], v["prominent_on_not_warranted"], v["any_flag_on_numeric"],
                           v["any_flag_on_non_numeric"]] for c, v in m["by_category"].items()]), ""]
    L += ["Field parser limits: see `benchmarks/fields.py` (lexicon names only, first value per name, units "
          "compared without case, frequencies and durations not parsed)."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fixtures", default=FIXTURES)
    ap.add_argument("--name", default="warnings", help="result file name (default warnings; e.g. warnings_r2)")
    a = ap.parse_args(argv)
    C.limit_threads(2)
    res = run(a.root, a.fixtures, a.limit)
    name = C.out_name(a.name, a.limit)
    path = C.write_result(a.out, name, res, markdown(res))
    print(path)
    for mode, m in res["modes"].items():
        if m.get("available"):
            t = m["target_original_39"]
            print(mode, "target:", t["recall"], t["result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
