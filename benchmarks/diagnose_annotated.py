"""Rank annotated-crop OCR failures by frequency and consequence, from a run_handwriting result.

    python benchmarks/diagnose_annotated.py <checkout> <run_handwriting result.json> <out.json>

Per line (reference "truth"): exact; recognition loss (both readers wrong); selection loss (a reader was
right but the other reading was shown); routing loss (handwriting right, line not routed). Numbers: lines
whose shown numbers differ from the reference (benchmarks.fields.numbers, exact spelling), split by
whether ocr.common.line_confirm_numbers would show the "Check the numbers" warning (as Scanner.run builds
the line, shipped default for the misread-digit check); silent ones are listed. A warning on a line that
was read exactly is counted too (by design every handwritten line with a number is flagged).
"""
import json, sys
sys.path.insert(0, sys.argv[1])
from benchmarks.fields import numbers, has_decimal, parse_fields, match_fields
from ocr.common import line_confirm_numbers, FLAG_MISREAD_DIGITS
res = json.load(open(sys.argv[2]))
out = {}
norm = lambda s: " ".join((s or "").split())
for split in ("validation", "test"):
    rows = res[split]["rows"]
    cats = {"exact": 0, "recognition_loss (both readers wrong)": 0, "selection_loss (rejected the right reading)": 0,
            "routing_loss (not routed, handwriting right)": 0}
    num = {"lines_with_reference_number": 0, "number_wrong": 0, "number_wrong_flagged": 0, "number_wrong_silent": 0,
           "decimal_lines": 0, "decimal_wrong": 0, "decimal_wrong_silent": 0, "false_warning_on_exact_line": 0,
           "exact_lines_shown": 0}
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
        has_num = bool(numbers(r["truth"]))
        wrong_num = has_num and numbers(r["shipped"]) != numbers(r["truth"])
        num["lines_with_reference_number"] += has_num
        num["number_wrong"] += wrong_num
        num["number_wrong_flagged"] += wrong_num and flag
        num["number_wrong_silent"] += wrong_num and not flag
        if has_decimal(r["truth"]):
            num["decimal_lines"] += 1
            dw = numbers(r["shipped"]) != numbers(r["truth"])
            num["decimal_wrong"] += dw
            num["decimal_wrong_silent"] += dw and not flag
        if ok_s:
            num["exact_lines_shown"] += 1
            num["false_warning_on_exact_line"] += flag
        if wrong_num and not flag:
            silent.append({"truth": r["truth"], "shown": r["shipped"], "printed": r["printed"], "handwriting": r["handwriting"],
                           "routed": r["routed"], "shipped_conf": r["shipped_conf"]})
    out[split] = {"n": len(rows), "categories": cats, "numbers": num, "silent_numeric_errors": silent}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for split, o in out.items():
    print(split, o["n"], o["categories"]); print("   ", o["numbers"])
    for s in o["silent_numeric_errors"]: print("    SILENT", s)
