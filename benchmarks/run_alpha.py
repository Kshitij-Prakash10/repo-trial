"""Alpha-conversion benchmark: how transparent and 16-bit images become RGB before page detection.

usage: python -m benchmarks.run_alpha --root <checkout> --out <dir> [--data DIR] [--limit N] [--baseline alpha.json]

Needs only requirements.txt (no torch); the result records whether torch got imported.

1. Paired images. Two images that differ by ONE fully opaque pixel in the far corner. scanner/convert.py at
   0c71106 composites an image on white only when no pixel at all is opaque, and otherwise keeps the colour
   stored under partly transparent pixels, so the single pixel switches the rule for the whole image. For
   each pair: mean absolute grey difference inside the text mask, text/paper contrast of each, and the page
   outlines (IoU of the two predicted quads and of each with the truth). Cases: antialiased text on a
   transparent background, a semi-transparent overlay over a whole photo, a photo with partly transparent
   surroundings and a 254-alpha page, a cut-out with fully transparent surroundings, and 16-bit grey.
   Entry points: scanner.convert.to_rgb + scanner.infer.scan (direct), and POST /api/scan through
   server.create_app(with_ocr=False) and Flask's test client, with automatic detection and with the true
   corners sent as manual corners.
2. The 20 partial-alpha photos of the earlier alpha_choice.py (seed 2026 picks 20 of the synthetic test
   photos; outside the page alpha 64 / 128 / 192, alpha 128 over black, fully transparent over black) plus
   a 254-alpha page variant, through scanner.infer.scan.
3. The full synthetic test (200) and OOD (80) photos, unchanged RGB, through scanner.infer.scan.
IoU convention (as the app): the quad scan() returns, i.e. the whole image when no page is found, against
the true page mask, both clipped to the frame (scanner.train.iou).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import common as C  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

MAIN_DATA = "/home/user/repo-trial/data/synthetic"


def find_data(root: str, data: str | None) -> str:
    for d in ([data] if data else []) + [os.path.join(root, "data", "synthetic"), MAIN_DATA]:
        if d and os.path.exists(os.path.join(d, "test", "labels.csv")):
            return os.path.abspath(d)
    raise SystemExit("synthetic photos not found: pass --data <dir with test/labels.csv> "
                     "(scripts or scanner.synth regenerate them; they are not in git)")


def quad_iou(a, b, W, H, synth) -> float:
    ma = synth.poly_mask(np.asarray(a, float), W, H) >= 0.5
    mb = synth.poly_mask(np.asarray(b, float), W, H) >= 0.5
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 0.0


def grey(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), np.float64)


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def text_layer(W=640, H=200, alpha_max=200):
    """Antialiased black text, partly transparent, on a fully transparent background (RGB 0 under it)."""
    cov = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(cov)
    d.text((16, 20), "Ramipril 2.5 mg OD", fill=255, font=_font(48))
    d.text((16, 100), "Metformin 500 mg BD", fill=255, font=_font(40))
    c = np.asarray(cov, np.float64) / 255.0
    a = np.zeros((H, W, 4), np.uint8)
    a[..., 3] = np.rint(c * alpha_max).astype(np.uint8)
    return a, c > 0.5


def add_pixel(a: np.ndarray) -> np.ndarray:
    b = a.copy()
    b[-1, -1] = [255, 255, 255, 255]  # one opaque white pixel in the far corner
    return b


def text_mask_of(photo: Image.Image, page: np.ndarray) -> np.ndarray:
    g = grey(photo)
    med = float(np.median(g[page])) if page.any() else 255.0
    return page & (g < 0.6 * med)


def pair_cases(it, dataset, synth, with_text=True):
    """[(case, image A, image B, text mask, true corners or None)]. B = A plus one opaque far pixel unless noted.
    The transparent-text pair does not depend on the photo: with_text=False leaves it out."""
    out = []
    a, m = text_layer()
    if with_text:
        out.append(("antialiased_text_on_transparent (text alpha <= 200, RGB 0 under alpha 0)",
                    Image.fromarray(a, "RGBA"), Image.fromarray(add_pixel(a), "RGBA"), m, None))
    rgb = dataset.open_image(it)
    W, H = rgb.size
    page = synth.poly_mask(it["corners"], W, H) >= 0.5
    tm = text_mask_of(rgb, page)
    p = np.asarray(rgb)
    over = np.dstack([p, np.full((H, W), 128, np.uint8)])
    out.append(("semi_transparent_overlay_over_photo (alpha 128 everywhere)", Image.fromarray(over, "RGBA"),
                Image.fromarray(add_pixel(over), "RGBA"), tm, it["corners"]))
    sur = np.dstack([p, np.where(page, 254, 128).astype(np.uint8)])
    out.append(("partial_alpha_surroundings (page alpha 254, outside 128)", Image.fromarray(sur, "RGBA"),
                Image.fromarray(add_pixel(sur), "RGBA"), tm, it["corners"]))
    sur255 = np.dstack([p, np.where(page, 255, 128).astype(np.uint8)])
    out.append(("partial_alpha_surroundings: page alpha 254 (A) vs 255 (B), outside 128",
                Image.fromarray(sur, "RGBA"), Image.fromarray(sur255, "RGBA"), tm, it["corners"]))
    blk = p.copy()
    blk[~page] = 0
    cut = np.dstack([blk, np.where(page, 254, 0).astype(np.uint8)])
    out.append(("fully_transparent_cutout (page alpha 254, outside alpha 0 over black)", Image.fromarray(cut, "RGBA"),
                Image.fromarray(add_pixel(cut), "RGBA"), tm, it["corners"]))
    g8 = np.asarray(rgb.convert("L"), np.uint16)
    out.append(("16bit_grey: I;16 (x257) (A) vs 8-bit L (B)", Image.fromarray((g8 * 257).astype(np.uint16)),
                Image.fromarray(g8.astype(np.uint8), "L"), tm, it["corners"]))
    out.append(("16bit_grey: I;16 holding 0..255 values (A) vs 8-bit L (B)", Image.fromarray(g8.astype(np.uint16)),
                Image.fromarray(g8.astype(np.uint8), "L"), tm, it["corners"]))
    return out


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def api_scan(client, im: Image.Image, corners=None) -> dict:
    data = {"image": (io.BytesIO(png_bytes(im)), "x.png"), "mode": "gray"}
    if corners is not None:
        data["corners"] = json.dumps([[float(x), float(y)] for x, y in corners])
    r = client.post("/api/scan", data=data, content_type="multipart/form-data")
    j = r.get_json()
    if r.status_code != 200:
        return {"status": r.status_code, "error": j.get("error") if j else None}
    img = Image.open(io.BytesIO(base64.b64decode(j["image_png_b64"])))
    return {"status": 200, "page_found": j["page_found"], "corners": j["corners"], "image": img, "method": j["method"]}


def contrast(g: np.ndarray, m: np.ndarray) -> float | None:
    if not m.any() or m.all():
        return None
    return float(np.median(g[~m]) - g[m].mean())


def run_pairs(items, mods, client) -> list[dict]:
    convert, infer, synth, dataset = mods["convert"], mods["infer"], mods["synth"], mods["dataset"]
    rows = []
    for n, it in enumerate(items):
        for case, A, B, m, truth in pair_cases(it, dataset, synth, with_text=n == 0):
            W, H = A.size
            ga, gb = grey(convert.to_rgb(A)), grey(convert.to_rgb(B))
            sa, sb = infer.scan(A, "gray"), infer.scan(B, "gray")
            row = {"photo": os.path.basename(it["image"]) if truth is not None else None, "case": case,
                   "direct": {"text_mean_abs_diff": round(float(np.abs(ga - gb)[m].mean()), 3) if m.any() else None,
                              "text_grey_A": round(float(ga[m].mean()), 2), "text_grey_B": round(float(gb[m].mean()), 2),
                              "contrast_A": contrast(ga, m), "contrast_B": contrast(gb, m),
                              "found_A": bool(sa["page_found"]), "found_B": bool(sb["page_found"]),
                              "iou_A_vs_B": round(quad_iou(sa["corners"], sb["corners"], W, H, synth), 4)}}
            if truth is not None:
                tmask = synth.poly_mask(np.asarray(truth, float), W, H) >= 0.5
                for k, s in (("A", sa), ("B", sb)):
                    pm = synth.poly_mask(np.asarray(s["corners"], float), W, H) >= 0.5
                    row["direct"][f"iou_{k}_vs_truth"] = round(float((pm & tmask).sum() / max((pm | tmask).sum(), 1)), 4)
            up = {}
            if client is not None:
                ua, ub = api_scan(client, A), api_scan(client, B)
                up["auto"] = {"status": [ua["status"], ub["status"]]}
                if ua["status"] == ub["status"] == 200:
                    up["auto"].update(found_A=ua["page_found"], found_B=ub["page_found"],
                                      iou_A_vs_B=round(quad_iou(ua["corners"], ub["corners"], W, H, synth), 4))
                corners = truth if truth is not None else [[0, 0], [W, 0], [W, H], [0, H]]
                ma, mb = api_scan(client, A, corners), api_scan(client, B, corners)
                up["manual"] = {"status": [ma["status"], mb["status"]]}
                if ma["status"] == mb["status"] == 200:
                    ia, ib = grey(ma["image"]), grey(mb["image"].resize(ma["image"].size))
                    up["manual"]["output_mean_abs_diff"] = round(float(np.abs(ia - ib).mean()), 3)
                    up["manual"]["output_grey_mean"] = [round(float(ia.mean()), 2), round(float(ib.mean()), 2)]
            row["upload"] = up
            rows.append(row)
    return rows


def partial_alpha_photos(items, mods, rng_seed=C.SEED, n=20) -> tuple[dict, list]:
    infer, synth, dataset, train = mods["infer"], mods["synth"], mods["dataset"], mods["train"]
    rng = np.random.default_rng(rng_seed)
    pick = sorted(rng.choice(len(items), n, replace=False).tolist())  # same draw as alpha_choice.py
    rows = []
    for k in pick:
        it = items[k]
        rgb = dataset.open_image(it)
        W, H = rgb.size
        mask = train.true_mask(it["corners"], W, H)
        page = synth.poly_mask(it["corners"], W, H) >= 0.5
        a = np.asarray(rgb)
        blk = a.copy()
        blk[~page] = 0
        variants = {f"alpha{al}_outside": np.dstack([a, np.where(page, 255, al).astype(np.uint8)]) for al in (64, 128, 192)}
        variants["alpha128_outside_rgb0"] = np.dstack([blk, np.where(page, 255, 128).astype(np.uint8)])
        variants["cutout_alpha0_rgb0"] = np.dstack([blk, np.where(page, 255, 0).astype(np.uint8)])
        variants["alpha128_outside_page254"] = np.dstack([a, np.where(page, 254, 128).astype(np.uint8)])
        variants["rgb_reference"] = None
        for v, arr in variants.items():
            im = rgb if arr is None else Image.fromarray(arr, "RGBA")
            s = infer.scan(im, "gray")
            rows.append({"photo": os.path.basename(it["image"]), "variant": v, "found": bool(s["page_found"]),
                         "iou": round(train.iou(np.asarray(s["corners"], float), mask), 4)})
    return summarise_variants(rows, "variant"), rows


def summarise_variants(rows, key) -> dict:
    out = {}
    for v in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == v]
        iou = np.array([r["iou"] for r in sub])
        out[v] = {"n": len(sub), "found": C.frac(sum(r["found"] for r in sub), len(sub)),
                  "iou_mean": round(float(iou.mean()), 4), "iou_ge_0.9": C.frac(int((iou >= 0.9).sum()), len(sub))}
    return out


def full_sets(data, mods, limit) -> tuple[dict, list]:
    infer, dataset, train = mods["infer"], mods["dataset"], mods["train"]
    rows = []
    for split in ("test", "ood"):
        items = dataset.load_split(split, data)
        if limit:
            items = items[:limit]
        for it in items:
            img = dataset.open_image(it)
            W, H = img.size
            t = time.perf_counter()
            s = infer.scan(img, "gray")
            rows.append({"split": split, "photo": os.path.basename(it["image"]), "found": bool(s["page_found"]),
                         "iou": round(train.iou(np.asarray(s["corners"], float), train.true_mask(it["corners"], W, H)), 4),
                         "scan_ms": round((time.perf_counter() - t) * 1000, 1)})
    return summarise_variants(rows, "split"), rows


def changes(rows, base_rows, keys) -> list:
    if base_rows is None:
        return []
    b = {tuple(r[k] for k in keys): r for r in base_rows}
    out = []
    for r in rows:
        o = b.get(tuple(r[k] for k in keys))
        if o and (o["found"] != r["found"] or (o["iou"] >= 0.9) != (r["iou"] >= 0.9) or abs(o["iou"] - r["iou"]) > 0.02):
            out.append({**{k: r[k] for k in keys}, "found": [o["found"], r["found"]], "iou": [o["iou"], r["iou"]]})
    return out


def run(root, data=None, limit=None, baseline=None) -> dict:
    root = C.use_root(root)
    from scanner import convert, dataset, infer, synth, train  # noqa: E402
    for m in (convert, infer):
        C.check_imported_from(m, root)
    mods = {"convert": convert, "infer": infer, "synth": synth, "dataset": dataset, "train": train}
    data = find_data(root, data)
    client = None
    try:
        import server  # noqa: E402
        C.check_imported_from(server, root)
        client = server.create_app(with_scanner=True, with_ocr=False).test_client()
    except Exception as err:  # noqa: BLE001
        client_err = f"{type(err).__name__}: {err}"
    else:
        client_err = None
    test_items = dataset.load_split("test", data)
    pair_items = test_items[:3] if not limit else test_items[:1]
    res = {"benchmark": "alpha", "root": root, "git": C.git_info(root), "data": data,
           "model_path": infer.MODEL_PATH, "upload_client_error": client_err}
    res["pairs"] = run_pairs(pair_items, mods, client)
    s20, r20 = partial_alpha_photos(test_items, mods, n=min(20, limit) if limit else 20)
    sfull, rfull = full_sets(data, mods, limit)
    base = json.load(open(baseline)) if baseline else None
    res["partial_alpha_20"] = {"summary": s20, "rows": r20,
                               "changes_vs_baseline": changes(r20, base and base["partial_alpha_20"]["rows"],
                                                              ("photo", "variant"))}
    res["synthetic_full"] = {"summary": sfull, "rows": rfull,
                             "changes_vs_baseline": changes(rfull, base and base["synthetic_full"]["rows"],
                                                            ("split", "photo"))}
    res["iou_convention"] = "scan() quad (whole image when no page is found) vs the true page mask, clipped to the frame"
    res["baseline_compared"] = os.path.abspath(baseline) if baseline else None
    res["torch_imported"] = "torch" in sys.modules
    return res


def markdown(res) -> str:
    L = [f"# Alpha-conversion benchmark: {res['git'].get('commit') or res['root']}", "",
         f"torch imported: {res['torch_imported']}. IoU convention: {res['iou_convention']}.", "",
         "## Paired images (B = A + one opaque far pixel unless the case says otherwise)", ""]
    rows = []
    for r in res["pairs"]:
        d, u = r["direct"], r["upload"]
        rows.append([r["case"], r["photo"] or "-", d["text_mean_abs_diff"], f"{d['contrast_A']:.1f} / {d['contrast_B']:.1f}"
                     if d["contrast_A"] is not None else "-", f"{d['found_A']} / {d['found_B']}", d["iou_A_vs_B"],
                     f"{d.get('iou_A_vs_truth', '-')} / {d.get('iou_B_vs_truth', '-')}",
                     u.get("auto", {}).get("iou_A_vs_B"), u.get("manual", {}).get("output_mean_abs_diff")])
    L += [C.md_table(["case", "photo", "text |A-B| (grey)", "contrast A / B", "found A / B", "quad IoU A-B",
                      "IoU vs truth A / B", "upload auto IoU A-B", "upload manual output |A-B|"], rows), ""]
    L += ["## 20 partial-alpha photos", "",
          C.md_table(["variant", "found", "mean IoU", "IoU >= 0.9"],
                     [[k, v["found"], v["iou_mean"], v["iou_ge_0.9"]] for k, v in res["partial_alpha_20"]["summary"].items()]),
          "", "## Full synthetic sets (RGB)", "",
          C.md_table(["split", "found", "mean IoU", "IoU >= 0.9"],
                     [[k, v["found"], v["iou_mean"], v["iou_ge_0.9"]] for k, v in res["synthetic_full"]["summary"].items()]),
          "", f"Photo-level changes vs baseline ({res['baseline_compared']}): partial-alpha "
              f"{len(res['partial_alpha_20']['changes_vs_baseline'])}, full sets {len(res['synthetic_full']['changes_vs_baseline'])}."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline", default=None, help="an earlier alpha.json to list photo-level changes against")
    a = ap.parse_args(argv)
    C.limit_threads(1)  # the scanner's OpenMP predictor is faster on one thread when the CPU is shared
    res = run(a.root, a.data, a.limit, a.baseline)
    print(C.write_result(a.out, C.out_name("alpha", a.limit), res, markdown(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
