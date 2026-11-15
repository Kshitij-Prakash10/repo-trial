"""Validation study behind two defaults in ocr/scan.py: deskew off, and the routing rule.

    python scripts/val_studies/deskew_routing_study.py reports/val_studies/deskew_routing_val.json

For every validation page (first 60 per printed set, all GNHK val pages): the skew estimate, the page
CER with and without deskew (handwriting reader off), and every line read both by the printed and by
the handwriting reader (lines with mean P(handwritten) >= 0.2). scripts/val_studies/analyse.py turns
it into the tables in reports/val_studies/README.md.

The committed JSON was produced before the routing rule existed, when the handwriting reading always
replaced the printed one; this version recovers both readings from the current Scanner output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image  # noqa: E402

import ocr.scan as S  # noqa: E402
from ocr import hwdata, orientation as O  # noqa: E402
from ocr.common import DATASETS, error_counts, load_pages  # noqa: E402
from ocr.evaluate import reference_text  # noqa: E402
from ocr.evaluate_v2 import gnhk_reference  # noqa: E402


def both_readings(ln: dict) -> dict:
    printed = ln.get("printed_text", ln["text"]) if ln["script"] == "handwritten" else ln["text"]
    pconf = ln.get("printed_confidence", ln["confidence"]) if ln["script"] == "handwritten" else ln["confidence"]
    routed = "handwriting_text" in ln
    return {"text": ln["handwriting_text"] if routed else ln["text"],
            "confidence": ln["handwriting_confidence"] if routed else ln["confidence"],
            "script": "handwritten" if routed else "printed", "handwritten_share": ln["handwritten_share"],
            "printed_text": printed if routed else None, "printed_confidence": pconf if routed else None}


def main(out_path: str) -> None:
    S.HW_LINE_SHARE = 0.2  # read more lines both ways; rules are applied afterwards
    sc = S.Scanner()
    out = []
    items = [(ds, p) for ds in DATASETS for p in load_pages(ds, "val")[:60]] + [("gnhk", p) for p in hwdata.gnhk_pages("val")]
    for ds, p in items:
        img = hwdata.load_image(p["path"])
        ref = gnhk_reference(p) if ds == "gnhk" else reference_text(p)
        ang, gain = O.estimate_skew(img)
        sc.do_deskew = False
        r = sc.run(Image.fromarray(img))
        row = {"ds": ds, "id": p["id"], "ref": ref, "angle": ang, "gain": gain, "lines": [both_readings(ln) for ln in r["lines"]]}
        if abs(ang) >= O.MIN_SKEW and gain >= 1.0:
            hw, sc.hw = sc.hw, None
            sc.do_deskew = True
            r2 = sc.run(Image.fromarray(img))
            sc.do_deskew = False
            r3 = sc.run(Image.fromarray(img))
            sc.hw = hw
            row["cer_deskew"] = error_counts(r2["text"], ref, ds)
            row["cer_plain"] = error_counts(r3["text"], ref, ds)
        out.append(row)
        print(ds, p["id"], round(ang, 2), round(gain, 3), flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
