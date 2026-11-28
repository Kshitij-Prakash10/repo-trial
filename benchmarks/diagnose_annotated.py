"""Rank annotated-crop OCR failures by frequency and consequence, from a run_handwriting result.

    python benchmarks/diagnose_annotated.py <checkout> <run_handwriting result.json> <out.json>

Per line (reference "truth"): exact; recognition loss (both readers wrong); selection loss (a reader was
right but the other reading was shown); routing loss (handwriting right, line not routed). Numbers: lines
whose shown numbers differ from the reference (benchmarks.fields.numbers, exact spelling), split by
whether ocr.common.line_confirm_numbers would show the "Check the numbers" warning (as Scanner.run builds
the line, shipped default for the misread-digit check); silent ones are listed. Every line is compared,
including lines whose reference holds no number, so a number invented there is a numeric error. Its
denominator is every line ("n"); the split by whether the reference holds a number is given as
number_wrong_where_reference_has_number (out of lines_with_reference_number) and
number_invented_where_reference_has_none (out of n - lines_with_reference_number). A warning on a line
that was read exactly is counted too (by design every handwritten line with a number is flagged).
"""
import json, os, sys
if __name__ == "__main__":
    # Run as a script, Python puts benchmarks/ first on sys.path, where benchmarks/warnings would shadow the
    # standard library's warnings module (numpy imports it). Use the checkout given instead.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    sys.path.insert(0, sys.argv[1])
from benchmarks.fields import numbers, has_decimal  # noqa: E402
from ocr.common import line_confirm_numbers, FLAG_MISREAD_DIGITS  # noqa: E402


def norm(s):
    return " ".join((s or "").split())


def diagnose(rows) -> dict:
    """The diagnosis of one split's run_handwriting rows."""
    cats = {"exact": 0, "recognition_loss (both readers wrong)": 0, "selection_loss (rejected the right reading)": 0,
            "routing_loss (not routed, handwriting right)": 0}
    num = {"lines_with_reference_number": 0, "number_wrong": 0, "number_wrong_flagged": 0, "number_wrong_silent": 0,
           "decimal_lines": 0, "decimal_wrong": 0, "decimal_wrong_silent": 0, "false_warning_on_exact_line": 0,
           "exact_lines_shown": 0, "number_wrong_where_reference_has_number": 0,
           "number_invented_where_reference_has_none": 0}
    silent = []
    for r in rows:
        ok_s, ok_h, ok_p = (norm(r[k]) == norm(r["truth"]) for k in ("shipped", "handwriting", "printed"))
        if ok_s: cats["exact"] += 1
        elif not (ok_h or ok_p): cats["recognition_loss (both readers wrong)"] += 1
        elif ok_h and not r["routed"]: cats["routing_loss (not routed, handwriting right)"] += 1
        else: cats["selection_loss (rejected the right reading)"] += 1
        # the line as Scanner.run would flag it (routing known; handwriting reading present when routed)
        ln = {"text": r["shipped"], "script": "handwritten" if (r["routed"] and r["shipped"] == r["handwriting"] and r["shipped"] != r["printed"]) else "printed"}
        if r["routed"]:
            ln["handwriting_text"] = r["handwriting"]
            if ln["script"] == "handwritten": ln["printed_text"] = r["printed"]
        flag = line_confirm_numbers(ln, script_known=True, flag_misread_digits=FLAG_MISREAD_DIGITS)
        ref_nums, shown_nums = numbers(r["truth"]), numbers(r["shipped"])
        has_num = bool(ref_nums)
        wrong_num = shown_nums != ref_nums  # on every line: a number invented on a line without one counts
        num["lines_with_reference_number"] += has_num
        num["number_wrong"] += wrong_num
        num["number_wrong_where_reference_has_number"] += wrong_num and has_num
        num["number_invented_where_reference_has_none"] += wrong_num and not has_num
        num["number_wrong_flagged"] += wrong_num and flag
        num["number_wrong_silent"] += wrong_num and not flag
        if has_decimal(r["truth"]):
            num["decimal_lines"] += 1
            num["decimal_wrong"] += wrong_num
            num["decimal_wrong_silent"] += wrong_num and not flag
        if ok_s:
            num["exact_lines_shown"] += 1
            num["false_warning_on_exact_line"] += flag
        if wrong_num and not flag:
            silent.append({"truth": r["truth"], "shown": r["shipped"], "printed": r["printed"], "handwriting": r["handwriting"],
                           "routed": r["routed"], "shipped_conf": r["shipped_conf"]})
    return {"n": len(rows), "categories": cats, "numbers": num, "silent_numeric_errors": silent}


def main(argv) -> int:
    res = json.load(open(argv[2]))
    out = {split: diagnose(res[split]["rows"]) for split in ("validation", "test")}
    json.dump(out, open(argv[3], "w"), indent=1)
    for split, o in out.items():
        print(split, o["n"], o["categories"]); print("   ", o["numbers"])
        for s in o["silent_numeric_errors"]: print("    SILENT", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
