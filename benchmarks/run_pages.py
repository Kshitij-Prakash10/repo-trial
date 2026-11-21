"""Page detection and geometry benchmark of one checkout (scanner.infer.scan, the app's entry point).

usage: python -m benchmarks.run_pages --root <checkout> --out <dir> [--data DIR] [--limit N] [--per-kind 20]

Needs only requirements.txt. Seed 2026 everywhere; every generated image is made in memory.

1. False alarms on photos without a page, by kind: every background kind of scanner.synth.make_background
   (train style: wood fabric solid light gradient blotch stripes tiles dots; ood style: checker granite light),
   drawn with the generator's own random choice and kept until each kind has --per-kind images, lit with the
   generator's linear gradient and JPEG q90; blank photos (white, grey, paper-coloured, dark; linear lighting
   gradient and mild sensor noise); uniform RGB noise; tiny images (1 to 48 px: white, noise, a page-like square).
2. Recall on pages: the stored synthetic test (200) and OOD (80) photos, and generated hard cases with a known
   quad, built here from the repository's page renderer (scanner.synth.render_page) and its homography:
   regular (a generator background), low contrast (a plain background 6 to 25 grey levels darker than the
   paper), and partly visible (a page turned 15 to 30 degrees, the photo then cropped so that the leftmost
   corner, or the leftmost and the topmost corners, fall outside the frame).
3. The generated examples as raw, cropped (to the page's bounding box plus 4 % margin), rotated 90 and 180
   degrees, and tilted by 4 degrees (the scanner has no deskew of its own; "deskewed" inputs here are the small
   tilts a deskew step would remove). Per transform: outline IoU with the true quad mapped through the
   transform, and lost content: the share of the true page area, and of the rendered text pixels, that lies
   outside the returned quad (scan() returns the whole image when no page is found: nothing is cut then).
4. The stored per-photo results of the fine-tuned model on real photos (models/finetune/eval/
   ft_real_only__doccorner_{val,test}.csv.gz, ft_real_only__smartdoc.csv.gz) for the confidence threshold
   only: chosen on val (the highest threshold that keeps >= 99.5 % of the well-outlined pages, IoU >= 0.9),
   then evaluated once on test and SmartDoc. The stored "found" flags were written at the old threshold 0.35,
   and the blank and noise guards need the pixels, so they cannot be measured from these files.
IoU convention: as the app, the quad scan() returns against the true mask, both clipped to the frame.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import common as C  # noqa: E402
from benchmarks.run_alpha import find_data  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TRAIN_KINDS = ("wood", "fabric", "solid", "light", "gradient", "blotch", "stripes", "tiles", "dots")
OOD_KINDS = ("checker", "granite", "light")
THRESHOLDS = tuple(round(0.35 + 0.05 * i, 2) for i in range(12))  # 0.35 .. 0.90
KEEP_SHARE = 0.995


def _size(rng):
    L = int(rng.integers(480, 1281))
    ar = float(rng.choice([4 / 3, 3 / 2, 16 / 9, 1.0]))
    return (int(L / ar), L) if rng.random() < 0.6 else (L, int(L / ar))


def _light(rng, H, W, lo_min=0.55):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    t = xx * np.cos(ang) + yy * np.sin(ang)
    t = (t - t.min()) / (np.ptp(t) + 1e-6)
    lo = rng.uniform(lo_min, 1.0)
    return lo + (1 - lo) * t


def _jpeg(arr, q=90) -> Image.Image:
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def negatives(synth, per_kind: int):
    """[(group, kind, id, image)] without any page."""
    out = []
    for style, kinds in (("train", TRAIN_KINDS), ("ood", OOD_KINDS)):
        count = {k: 0 for k in kinds}
        i = 0
        while min(count.values()) < per_kind and i < 200 * per_kind:
            rng = np.random.default_rng([C.SEED, 11, 0 if style == "train" else 1, i])
            i += 1
            W, H = _size(rng)
            paper = (int(rng.integers(235, 256)), int(rng.integers(225, 245)), int(rng.integers(195, 225)))
            # make_background draws its kind first; a throw-away copy of the generator tells which one it is
            probe = np.random.default_rng([C.SEED, 11, 0 if style == "train" else 1, i - 1])
            _size(probe)
            for lo, hi in ((235, 256), (225, 245), (195, 225)):
                probe.integers(lo, hi)
            kind = kinds[int(probe.integers(0, len(kinds)))]  # make_background's first draw
            if count[kind] >= per_kind:
                continue
            bg, kind2 = synth.make_background(rng, W, H, paper, style)
            assert kind2 == kind
            count[kind] += 1
            out.append(("background", f"{style}_{kind}", f"bg_{style}_{kind}_{count[kind]:02d}",
                        _jpeg(bg * _light(rng, H, W)[..., None])))
    rng = np.random.default_rng([C.SEED, 12])
    for kind in ("white", "grey", "paper", "dark"):
        for j in range(per_kind):
            W, H = _size(rng)
            base = {"white": np.array([250, 250, 250.]), "grey": np.full(3, rng.uniform(90, 200)),
                    "paper": np.array([rng.uniform(235, 255), rng.uniform(225, 245), rng.uniform(195, 225)]),
                    "dark": np.full(3, rng.uniform(20, 70))}[kind]
            im = base[None, None] * _light(rng, H, W, 0.7)[..., None] + rng.normal(0, rng.uniform(0, 3), (H, W, 1))
            out.append(("blank", f"blank_{kind}", f"blank_{kind}_{j:02d}", Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))))
    for j in range(per_kind):
        W, H = _size(rng)
        out.append(("noise", "noise_uniform", f"noise_{j:02d}", Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))))
    for s in (1, 8, 16, 31, 32, 48):
        out.append(("tiny", "tiny_white", f"tiny_white_{s}", Image.new("RGB", (s, s), (250, 250, 250))))
        out.append(("tiny", "tiny_noise", f"tiny_noise_{s}", Image.fromarray(rng.integers(0, 256, (s, s, 3), dtype=np.uint8))))
        a = np.full((s, s, 3), 40, np.uint8)
        m = max(1, s // 5)
        a[m:s - m, m:s - m] = 245
        out.append(("tiny", "tiny_pagelike", f"tiny_pagelike_{s}", Image.fromarray(a)))
    return out


# ------------------------------------------------------------------ generated pages with a known quad and text mask
def compose(synth, idx: int, mode: str) -> dict:
    """One photo: {"image", "quad" (TL TR BR BL, px, may leave the frame), "text" (H, W) bool, "kind"}."""
    rng = np.random.default_rng([C.SEED, 21, ("regular", "low_contrast", "cut1", "cut2").index(mode), idx])
    W, H = (768, 1024) if rng.random() < 0.6 else (1024, 768)
    kind = str(synth.PAGE_KINDS[int(rng.integers(0, len(synth.PAGE_KINDS)))])
    paper = tuple(int(255 - rng.uniform(0, 20) - [0, 4, 12][i] * rng.uniform(0, 1)) for i in range(3))
    page = synth.render_page(rng, kind, 1.414, paper)
    pw, ph = page.size
    pg = np.asarray(page.convert("L"), np.float32)
    text_flat = pg < float(np.median(pg)) - 60
    if mode in ("cut1", "cut2"):
        rot = float(rng.uniform(15, 30) * rng.choice([-1, 1]))
        frac = rng.uniform(0.30, 0.42)
    else:
        rot = float(rng.normal(0, 6))
        frac = rng.uniform(0.25, 0.5)
    s = np.sqrt(frac * W * H / (pw * ph))
    sq = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) * s
    sq[:2, 0] *= rng.uniform(0.85, 1.0)  # mild keystone
    t = np.deg2rad(rot)
    q = sq @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    q *= min(1.0, float(np.min(0.94 * np.array([W, H]) / (q.max(0) - q.min(0)))))  # the whole page fits first
    q += np.array([W / 2, H / 2]) - (q.max(0) + q.min(0)) / 2
    Hm = synth._homography(q, np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], float))
    coeffs = tuple((Hm / Hm[2, 2]).ravel()[:8])
    warped = np.asarray(page.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC), np.float32)
    tm = Image.fromarray(text_flat.astype(np.uint8) * 255).transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    alpha = synth.poly_mask(q, W, H, ss=3)
    if mode == "low_contrast":
        delta = float(rng.uniform(6, 25))
        bg = np.broadcast_to(np.array(paper, np.float32) - delta, (H, W, 3)) + rng.normal(0, 2, (H, W, 1))
    else:
        bg, _ = synth.make_background(rng, W, H, paper, "train")
    img = bg * (1 - alpha[..., None]) + warped * alpha[..., None]
    img = img * _light(rng, H, W, 0.75)[..., None] + rng.normal(0, rng.uniform(0, 4), (H, W, 1))
    text = (np.asarray(tm, np.float32) > 127) & (alpha > 0.5)
    if mode in ("cut1", "cut2"):  # crop so that the leftmost (and the topmost) corner leave the frame
        order_x = np.argsort(q[:, 0])
        cut = 0.05 * s * pw
        x0 = int(np.floor(q[order_x[0], 0] + cut))
        x0 = min(x0, int(q[order_x[1], 0]) - 8)
        y0 = 0
        if mode == "cut2":
            others = [i for i in np.argsort(q[:, 1]) if i != order_x[0]]
            y0 = int(np.floor(q[others[0], 1] + cut))
            y0 = min(y0, int(min(q[i, 1] for i in others[1:])) - 8)
        img, text = img[y0:, x0:], text[y0:, x0:]
        q = q - np.array([x0, y0], float)
    image = _jpeg(img, 85)
    Wc, Hc = image.size
    out_corners = int(((q[:, 0] < 0) | (q[:, 0] > Wc) | (q[:, 1] < 0) | (q[:, 1] > Hc)).sum())
    want = {"cut1": 1, "cut2": 2}.get(mode, 0)
    if out_corners != want:
        raise AssertionError(f"compose({idx}, {mode!r}): {out_corners} corners outside the frame, not {want}")
    return {"image": image, "quad": q, "text": text, "kind": mode, "corners_outside": out_corners}


def transforms(ex: dict, angle=4.0):
    """[(name, image, quad, text mask)] of one generated example."""
    img, q, tm = ex["image"], ex["quad"], ex["text"]
    W, H = img.size
    out = [("raw", img, q, tm)]
    m = 0.04 * np.array([W, H])
    x0, y0 = np.maximum(np.floor(q.min(0) - m), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(q.max(0) + m), [W, H]).astype(int)
    out.append(("cropped", img.crop((x0, y0, x1, y1)), q - [x0, y0], tm[y0:y1, x0:x1]))
    out.append(("rot90", img.transpose(Image.Transpose.ROTATE_90), np.stack([q[:, 1], W - q[:, 0]], 1), np.rot90(tm, 1)))
    out.append(("rot180", img.transpose(Image.Transpose.ROTATE_180), np.stack([W - q[:, 0], H - q[:, 1]], 1), np.rot90(tm, 2)))
    t = np.deg2rad(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    box = np.array([[0, 0], [W, 0], [W, H], [0, H]], float) @ R.T
    lo, hi = box.min(0), box.max(0)
    Wt, Ht = int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1]))
    Ri = R.T  # inverse rotation: output (u, v) -> input = Ri @ ((u, v) + lo)
    data = (Ri[0, 0], Ri[0, 1], Ri[0, 0] * lo[0] + Ri[0, 1] * lo[1], Ri[1, 0], Ri[1, 1], Ri[1, 0] * lo[0] + Ri[1, 1] * lo[1])
    fill = tuple(int(v) for v in np.median(np.asarray(img).reshape(-1, 3), 0))
    tilted = img.transform((Wt, Ht), Image.AFFINE, data, Image.BICUBIC, fillcolor=fill)
    ttm = np.asarray(Image.fromarray(tm.astype(np.uint8) * 255).transform((Wt, Ht), Image.AFFINE, data, Image.NEAREST)) > 127
    out.append((f"tilt{angle:g}", tilted, q @ R.T - lo, ttm))
    return out


def outline_scores(synth, s, quad, text, W, H) -> dict:
    pred = synth.poly_mask(np.asarray(s["corners"], float), W, H) >= 0.5
    true = synth.poly_mask(np.asarray(quad, float), W, H) >= 0.5
    u = (pred | true).sum()
    return {"found": bool(s["page_found"]), "iou": round(float((pred & true).sum() / u) if u else 0.0, 4),
            "page_area_lost": round(float((true & ~pred).sum() / max(true.sum(), 1)), 4),
            "text_lost": round(float((text & ~pred).sum() / max(text.sum(), 1)), 4) if text is not None and text.any() else None}


def summarise(rows, key) -> dict:
    out = {}
    for v in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == v]
        iou = np.array([r["iou"] for r in sub])
        d = {"n": len(sub), "found": C.frac(sum(r["found"] for r in sub), len(sub)),
             "iou_mean": round(float(iou.mean()), 4), "iou_ge_0.9": C.frac(int((iou >= 0.9).sum()), len(sub))}
        if "page_area_lost" in sub[0]:
            lost = np.array([r["page_area_lost"] for r in sub])
            d["page_area_lost_mean"] = round(float(lost.mean()), 4)
            d["photos_losing_over_2pct_of_page"] = C.frac(int((lost > 0.02).sum()), len(sub))
            tl = [r["text_lost"] for r in sub if r.get("text_lost") is not None]
            if tl:
                d["text_lost_mean"] = round(float(np.mean(tl)), 4)
                d["photos_losing_over_1pct_of_text"] = C.frac(int((np.array(tl) > 0.01).sum()), len(tl))
        out[v] = d
    return out


def stored_threshold(root: str) -> dict:
    E = os.path.join(root, "models", "finetune", "eval")
    def load(name):
        with gzip.open(os.path.join(E, f"ft_real_only__{name}.csv.gz"), "rt") as fh:
            rows = list(csv.DictReader(fh))
        b = lambda v: str(v).strip().lower() in ("true", "1")  # noqa: E731
        neg = np.array([b(x["negative"]) for x in rows])
        found = np.array([b(x["found"]) for x in rows])
        conf = np.array([float(x["confidence"]) for x in rows])
        iou = np.array([float(x["iou_img"]) if x["iou_img"] not in ("", "nan") else np.nan for x in rows])
        return neg, found, conf, iou
    def at(d, thr):
        neg, found, conf, iou = d
        f = found & (conf >= thr)
        pos = ~neg
        good = pos & (iou >= 0.9)
        r = {"found_on_pages": C.frac(int(f[pos].sum()), int(pos.sum())),
             "well_outlined_kept (IoU >= 0.9)": C.frac(int(f[good].sum()), int(good.sum())),
             "found_and_iou_ge_0.9_on_pages": C.frac(int((f & good).sum()), int(pos.sum()))}
        if neg.any():
            r["false_alarms_on_negatives"] = C.frac(int(f[neg].sum()), int(neg.sum()))
        return r
    val = load("doccorner_val")
    sweep = {f"{t:.2f}": at(val, t) for t in THRESHOLDS}
    ok = [t for t in THRESHOLDS if (lambda k: k["num"] >= KEEP_SHARE * k["den"])(sweep[f"{t:.2f}"]["well_outlined_kept (IoU >= 0.9)"])]
    chosen = max(ok) if ok else min(THRESHOLDS)
    return {"rule": f"highest threshold on doccorner_val keeping >= {KEEP_SHARE} of the IoU >= 0.9 pages",
            "note": "stored 'found' was written at threshold 0.35; the blank and noise guards cannot be measured "
                    "from stored rows (they need the photos)",
            "val_sweep": sweep, "chosen": chosen,
            "test_once": {"doccorner_test": at(load("doccorner_test"), chosen), "smartdoc": at(load("smartdoc"), chosen)},
            "check_found_implies_conf_ge_0.35": bool((val[2][val[1]] >= 0.35).all())}


def run(root, data=None, limit=None, per_kind=20) -> dict:
    root = C.use_root(root)
    from scanner import dataset, infer, synth, train  # noqa: E402
    C.check_imported_from(infer, root)
    data = find_data(root, data)
    if limit:
        per_kind = min(per_kind, max(1, limit // 4))
    res = {"benchmark": "pages", "root": root, "git": C.git_info(root), "data": data, "model_path": infer.MODEL_PATH,
           "min_confidence": getattr(infer, "MIN_CONFIDENCE", None), "per_kind": per_kind,
           "iou_convention": "scan() quad (whole image when no page) vs the true mask, both clipped to the frame"}
    # 1 negatives
    rows = []
    for group, kind, name, img in negatives(synth, per_kind):
        t = time.perf_counter()
        s = infer.scan(img, "gray")
        rows.append({"group": group, "kind": kind, "id": name, "size": list(img.size), "found": bool(s["page_found"]),
                     "confidence": round(float(s["confidence"]), 4), "scan_ms": round((time.perf_counter() - t) * 1000, 1)})
    by = {}
    for k in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == k]
        by[k] = {"false_alarms": C.frac(sum(r["found"] for r in sub), len(sub)),
                 "ids": [r["id"] for r in sub if r["found"]]}
    res["false_alarms"] = {"by_kind": by, "by_group": {g: C.frac(sum(r["found"] for r in rows if r["group"] == g),
                                                                 sum(r["group"] == g for r in rows))
                                                       for g in sorted({r["group"] for r in rows})},
                           "rows": rows}
    # 2 recall: stored synthetic sets
    srows = []
    for split in ("test", "ood"):
        items = dataset.load_split(split, data)
        for it in (items[:limit] if limit else items):
            img = dataset.open_image(it)
            W, H = img.size
            s = infer.scan(img, "gray")
            sc = outline_scores(synth, s, it["corners"], None, W, H)
            srows.append({"split": split, "photo": os.path.basename(it["image"]), "background": it["meta"].get("background"),
                          **sc})
    res["synthetic_sets"] = {"summary": summarise(srows, "split"),
                             "by_background": {sp: summarise([r for r in srows if r["split"] == sp], "background")
                                               for sp in ("test", "ood")}, "rows": srows}
    # 2+3 generated hard cases and transforms
    n_each = {"regular": 20, "low_contrast": 20, "cut1": 15, "cut2": 15}
    grows = []
    for mode, n in n_each.items():
        for i in range(min(n, limit) if limit else n):
            ex = compose(synth, i, mode)
            for tname, img, q, tm in transforms(ex):
                W, H = img.size
                s = infer.scan(img, "gray")
                grows.append({"case": mode, "idx": i, "transform": tname, "corners_outside": ex["corners_outside"],
                              **outline_scores(synth, s, q, tm, W, H)})
    res["generated"] = {"n_per_case": n_each,
                        "raw_by_case": summarise([r for r in grows if r["transform"] == "raw"], "case"),
                        "by_transform": summarise(grows, "transform"),
                        "by_case_and_transform": {c: summarise([r for r in grows if r["case"] == c], "transform")
                                                  for c in n_each},
                        "rows": grows}
    # 4 stored real-photo rows
    try:
        res["stored_real_threshold"] = stored_threshold(root)
    except FileNotFoundError as err:
        res["stored_real_threshold"] = {"BLOCKED": f"stored per-photo file missing: {err.filename}"}
    return res


def markdown(res) -> str:
    L = [f"# Page detection benchmark: {res['git'].get('commit') or res['root']}", "",
         f"MIN_CONFIDENCE {res['min_confidence']}; {res['per_kind']} images per negative kind. {res['iou_convention']}.", "",
         "## False alarms (photos without a page)", "",
         C.md_table(["kind", "false alarms"], [[k, v["false_alarms"]] for k, v in res["false_alarms"]["by_kind"].items()]),
         "", "## Stored synthetic sets", "",
         C.md_table(["split", "found", "mean IoU", "IoU >= 0.9"],
                    [[k, v["found"], v["iou_mean"], v["iou_ge_0.9"]] for k, v in res["synthetic_sets"]["summary"].items()]),
         "", "## Generated pages (raw)", "",
         C.md_table(["case", "found", "mean IoU", "IoU >= 0.9", "page lost (mean)", "photos losing > 1 % of text"],
                    [[k, v["found"], v["iou_mean"], v["iou_ge_0.9"], v.get("page_area_lost_mean"),
                      v.get("photos_losing_over_1pct_of_text")] for k, v in res["generated"]["raw_by_case"].items()]),
         "", "## Generated pages by transform (all cases)", "",
         C.md_table(["transform", "found", "mean IoU", "IoU >= 0.9", "page lost (mean)", "photos losing > 1 % of text"],
                    [[k, v["found"], v["iou_mean"], v["iou_ge_0.9"], v.get("page_area_lost_mean"),
                      v.get("photos_losing_over_1pct_of_text")] for k, v in res["generated"]["by_transform"].items()]), ""]
    st = res.get("stored_real_threshold", {})
    if "chosen" in st:
        L += ["## Stored real-photo rows (threshold only)", "", st["rule"] + f": **{st['chosen']}**. " + st["note"] + ".", "",
              C.md_table(["set", "found on pages", "IoU>=0.9 kept", "false alarms"],
                         [[f"val @ {st['chosen']}", st["val_sweep"][f"{st['chosen']:.2f}"]["found_on_pages"],
                           st["val_sweep"][f"{st['chosen']:.2f}"]["well_outlined_kept (IoU >= 0.9)"],
                           st["val_sweep"][f"{st['chosen']:.2f}"].get("false_alarms_on_negatives")]]
                         + [[k, v["found_on_pages"], v["well_outlined_kept (IoU >= 0.9)"], v.get("false_alarms_on_negatives")]
                            for k, v in st["test_once"].items()])]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-kind", type=int, default=20)
    a = ap.parse_args(argv)
    C.limit_threads(1)  # the scanner's OpenMP predictor is faster on one thread when the CPU is shared
    res = run(a.root, a.data, a.limit, a.per_kind)
    print(C.write_result(a.out, C.out_name("pages", a.limit), res, markdown(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
