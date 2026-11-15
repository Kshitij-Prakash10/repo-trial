"""Validation study: does printed-recogniser confidence separate upright from 180-degree pages?

    python scripts/val_studies/flip_study.py reports/val_studies/orientation_flip_val.json

The classifier was the run's best checkpoint (now checkpoints/orientation_v1/orientation.pt)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from ocr import orientation as O
from ocr.pipeline import OcrPipeline, MAX_SIDE
from ocr.train_orient import pages, make
torch.set_num_threads(4)
m = O.load_orient()
ocr = OcrPipeline.load("finetuned", use_scanner=False)
rows = []
for img, src in pages("val"):
    h, w = img.shape[:2]; s = min(1.0, MAX_SIDE / max(h, w))
    a = np.asarray(Image.fromarray(img).resize((max(1, int(w*s)), max(1, int(h*s))), Image.BOX)) if s < 1 else img
    boxes, _ = ocr.detect(a)
    area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    boxes = boxes[np.argsort(-area)[:48]]
    crops = [a[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] for x0, y0, x1, y1 in boxes]
    c0 = [c for _, c in ocr.rec(crops)] if crops else []
    c2 = [c for _, c in ocr.rec([np.ascontiguousarray(c[::-1, ::-1]) for c in crops])] if crops else []
    with torch.no_grad():
        p = torch.softmax(O.logits_k(m, O.to_tensor([make(img, k, None) for k in range(4)])), 1).numpy()
    rows.append({"src": src, "n": len(crops), "c0": float(np.mean(c0)) if c0 else 0, "c2": float(np.mean(c2)) if c2 else 0,
                 "p": p.round(5).tolist()})
json.dump(rows, open(sys.argv[1], "w"))
