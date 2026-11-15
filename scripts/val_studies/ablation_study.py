"""Validation ablation: page CER with deskew and the handwriting reader each switched off.

    python scripts/val_studies/ablation_study.py > reports/val_studies/ablation_val.txt

The committed output was produced with the first routing rule (the handwriting reading always
replaced the printed one, deskew on by default); with today's Scanner the numbers differ.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import numpy as np
from PIL import Image
from ocr.common import DATASETS, error_counts, load_pages, rates
from ocr.evaluate import reference_text
from ocr.scan import Scanner
import ocr.scan as S
sc = Scanner()
res = {}
pages = {ds: load_pages(ds, "val")[:60] for ds in DATASETS}
for ds, ps in pages.items():
    rows = []
    for p in ps:
        with Image.open(p["path"]) as im:
            im.load(); im = im.copy()
        row = {}
        for name, (dsk, hw) in {"full": (True, True), "no_deskew": (False, True), "no_hw": (True, False), "neither": (False, False)}.items():
            sc.do_deskew = dsk
            hw_bak = sc.hw
            if not hw: sc.hw = None
            r = sc.run(im)
            sc.hw = hw_bak
            row[name] = error_counts(r["text"], reference_text(p), ds)
        rows.append(row)
    res[ds] = {n: rates([r[n] for r in rows])["cer"] for n in rows[0]}
    print(ds, len(ps), res[ds], flush=True)
