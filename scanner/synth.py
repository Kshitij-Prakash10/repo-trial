"""Seeded synthetic generator of document photos (SYNTHETIC data, labelled so everywhere).

A personal health paper (prescription, lab report, medical certificate, discharge summary, insurance
form) is rendered with Pillow, warped with a random perspective onto a random background, then lit, shadowed, blurred, noised
and JPEG-compressed. Ground truth: the four page corners (TL, TR, BR, BL of the page content, in photo
pixels, may lie slightly outside the frame) and the page mask (bool, H x W).

style "train" is the setting used for training, validation and test. style "ood" is a different setting
never used in training (other page layouts, tablecloth and granite backgrounds, stronger tilt, warm
light) for an out-of-distribution check. All names and values on the pages are invented.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATA_LABEL = "synthetic"
PAGE_KINDS = ("prescription", "lab_report", "medical_certificate", "discharge_summary", "insurance_form")
OOD_KINDS = ("dense_letter", "receipt", "grid_log")
PAGE_W = 640  # px, width of the rendered flat page before warping

WORDS = ("name", "date", "age", "sex", "patient", "doctor", "ward", "bed", "fit", "rest", "remarks", "blood",
         "pressure", "pulse", "diagnosis", "allergy", "history", "dose", "tab", "mg", "daily", "after", "food",
         "haemoglobin", "glucose", "sodium", "potassium", "result", "unit", "range", "fasting", "normal",
         "admitted", "discharged", "follow", "up", "policy", "member", "claim", "insurer", "amount", "signature",
         "medical", "valid", "until", "review", "sample", "reported", "cholesterol", "thyroid", "advice")
TITLES = {"medical_certificate": "MEDICAL CERTIFICATE", "prescription": "PRESCRIPTION",
          "lab_report": "LABORATORY REPORT", "discharge_summary": "DISCHARGE SUMMARY",
          "insurance_form": "HEALTH INSURANCE CLAIM FORM", "dense_letter": "REFERRAL LETTER",
          "receipt": "PHARMACY BILL", "grid_log": "BLOOD SUGAR LOG"}


@dataclass
class Sample:
    """One synthetic photo. corners: (4, 2) float, photo px, TL TR BR BL of the page. mask: (H, W) bool."""
    image: Image.Image
    corners: np.ndarray
    mask: np.ndarray
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size)


def _rand_word(rng: np.random.Generator, n: int = 1) -> str:
    return " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), n))


def _ink(rng: np.random.Generator, kind: str = "black") -> tuple[int, int, int]:
    if kind == "blue":
        return (int(rng.integers(10, 50)), int(rng.integers(30, 70)), int(rng.integers(120, 190)))
    v = int(rng.integers(15, 70))
    return (v, v, v + int(rng.integers(0, 15)))


def _text_lines(d: ImageDraw.ImageDraw, rng, x0: int, y0: int, x1: int, n: int, lh: int, color) -> int:
    """Draw n lines of word-like blocks between x0 and x1; return the next y (px)."""
    y = y0
    for i in range(n):
        x = x0
        end = x1 if i < n - 1 else x0 + int((x1 - x0) * rng.uniform(0.3, 0.9))
        th = max(2, int(lh * rng.uniform(0.35, 0.5)))
        while x < end - 10:
            w = int(rng.integers(8, 60))
            d.rectangle([x, y, min(x + w, end), y + th], fill=color)
            x += w + int(rng.integers(5, 10))
        y += lh
    return y


def _write(d, rng, xy, text: str, size: int, color) -> None:
    d.text(xy, text, fill=color, font=_font(size))


def _signature(d, rng, x: int, y: int, w: int, color) -> None:
    t = np.linspace(0, 1, 60)
    k = rng.uniform(3, 8)
    xs = x + t * w
    ys = y + np.sin(t * k * np.pi + rng.uniform(0, 6)) * rng.uniform(4, 12) + rng.normal(0, 1.2, t.size)
    d.line(list(zip(xs.tolist(), ys.tolist())), fill=color, width=int(rng.integers(1, 4)))


def _stamp(img: Image.Image, rng, cx: int, cy: int, r: int) -> None:
    col = (int(rng.integers(140, 210)), int(rng.integers(20, 60)), int(rng.integers(40, 90))) if rng.random() < 0.5 \
        else (int(rng.integers(30, 70)), int(rng.integers(50, 90)), int(rng.integers(150, 210)))
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    a = int(rng.integers(110, 200))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col + (a,), width=3)
    d.ellipse([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8], outline=col + (a,), width=2)
    _write(d, rng, (cx - r + 14, cy - 8), _rand_word(rng).upper(), 14, col + (a,))
    img.paste(Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB"))


def _table(d, rng, x0: int, y0: int, x1: int, rows: int, cols: int, rh: int, color, header: bool) -> int:
    ws = rng.uniform(0.6, 1.6, cols)
    xs = (x0 + np.concatenate([[0], np.cumsum(ws / ws.sum())]) * (x1 - x0)).astype(int)
    if header:
        d.rectangle([x0, y0, x1, y0 + rh], fill=tuple(int(c) for c in rng.integers(200, 235, 3)))
    for r in range(rows + 1):
        d.line([x0, y0 + r * rh, x1, y0 + r * rh], fill=color, width=1)
    for x in xs:
        d.line([x, y0, x, y0 + rows * rh], fill=color, width=1)
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.8:
                if r == 0 and header:
                    _write(d, rng, (xs[c] + 4, y0 + 3), _rand_word(rng).capitalize(), max(10, rh - 8), color)
                else:
                    w = int((xs[c + 1] - xs[c]) * rng.uniform(0.2, 0.8))
                    d.rectangle([xs[c] + 4, y0 + r * rh + rh // 3, xs[c] + 4 + w, y0 + r * rh + rh // 3 + rh // 3],
                                fill=color)
    return y0 + rows * rh


def render_page(rng: np.random.Generator, kind: str, aspect: float, paper: tuple[int, int, int]) -> Image.Image:
    """Render a flat page, PAGE_W wide and PAGE_W * aspect tall (px)."""
    w, h = PAGE_W, int(round(PAGE_W * aspect))
    img = Image.new("RGB", (w, h), paper)
    d = ImageDraw.Draw(img)
    m = int(w * rng.uniform(0.05, 0.1))
    ink = _ink(rng)
    y = m
    if kind in ("receipt",):
        for _ in range(int(h / 22) - 4):
            if y > h - m - 20:
                break
            _write(d, rng, (m, y), _rand_word(rng, 2).upper(), 13, ink)
            _write(d, rng, (w - m - 60, y), f"{rng.integers(1, 999)}.{rng.integers(0, 99):02d}", 13, ink)
            y += 22
        return img
    if rng.random() < 0.5:  # coloured header band
        band = tuple(int(c) for c in rng.integers(30, 200, 3))
        bh = int(rng.integers(40, 80))
        d.rectangle([0, 0, w, bh], fill=band)
        _write(d, rng, (m, bh // 2 - 10), f"{_rand_word(rng, 2).upper()} CLINIC", 18, (245, 245, 245))
        y = bh + 15
    else:
        if rng.random() < 0.6:
            d.ellipse([m, y, m + 50, y + 50], outline=ink, width=3)
        _write(d, rng, (m + 65, y + 5), f"{_rand_word(rng, 2).title()} Hospital", 18, ink)
        y += 60
        d.line([m, y, w - m, y], fill=ink, width=2)
        y += 10
    _write(d, rng, (w // 2 - 130, y + 5), TITLES.get(kind, kind.upper()), 22, ink)
    y += 45
    if kind in ("medical_certificate", "prescription", "discharge_summary", "insurance_form", "dense_letter"):
        for _ in range(int(rng.integers(2, 5))):
            _write(d, rng, (m, y), _rand_word(rng).capitalize() + ":", 15, ink)
            d.line([m + 90, y + 17, w - m - int(rng.integers(0, 200)), y + 17], fill=ink, width=1)
            if rng.random() < 0.7:
                _write(d, rng, (m + 100, y - 2), _rand_word(rng, 2).title(), 16, _ink(rng, "blue"))
            y += 30
    if kind == "prescription":
        _write(d, rng, (m, y), "Rx", 30, ink)
        y = _text_lines(d, rng, m + 50, y + 10, w - m, int(rng.integers(3, 7)), 26, _ink(rng, "blue"))
    elif kind in ("lab_report", "grid_log"):
        rows = int(rng.integers(6, 14)) if kind == "lab_report" else int((h - y - 2 * m) / 24)
        y = _table(d, rng, m, y, w - m, rows, int(rng.integers(3, 6)), 24, ink, True) + 15
    elif kind == "discharge_summary":
        for _ in range(int(rng.integers(2, 5))):
            if y > h - 5 * m:
                break
            _write(d, rng, (m, y), _rand_word(rng, 2).upper(), 15, ink)
            y = _text_lines(d, rng, m, y + 24, w - m, int(rng.integers(2, 5)), 18, ink) + 8
    elif kind == "insurance_form":
        for _ in range(int(rng.integers(3, 7))):  # boxed character cells
            if y > h - 4 * m:
                break
            _write(d, rng, (m, y + 2), _rand_word(rng).capitalize(), 13, ink)
            for k in range(int(rng.integers(8, 20))):
                x = m + 110 + k * 22
                if x + 20 > w - m:
                    break
                d.rectangle([x, y, x + 20, y + 22], outline=ink, width=1)
            y += 32
        for _ in range(int(rng.integers(3, 8))):
            if y > h - 3 * m:
                break
            d.rectangle([m, y, m + 14, y + 14], outline=ink, width=2)
            if rng.random() < 0.7:
                d.line([m + 2, y + 7, m + 6, y + 12, m + 13, y + 1], fill=_ink(rng, "blue"), width=2)
            _write(d, rng, (m + 24, y - 2), _rand_word(rng, int(rng.integers(2, 5))).capitalize(), 15, ink)
            y += 26
    elif kind == "dense_letter":
        y = _text_lines(d, rng, m, y, w - m, int((h - y - 3 * m) / 18), 18, ink)
    if y < h - 4 * m:
        y = _text_lines(d, rng, m, y + 10, w - m, int(rng.integers(2, 6)), 20, ink)
    if rng.random() < 0.8:
        _signature(d, rng, w - m - 180, min(h - m - 40, y + 50), 150, _ink(rng, "blue"))
    if rng.random() < 0.6:
        _stamp(img, rng, int(w * rng.uniform(0.2, 0.5)), int(min(h - m - 60, y + 70)), int(rng.integers(40, 60)))
    if rng.random() < 0.3:  # printed border box
        ImageDraw.Draw(img).rectangle([m // 2, m // 2, w - m // 2, h - m // 2], outline=ink, width=2)
    return img


def _smooth_noise(rng, H: int, W: int, cells: int) -> np.ndarray:
    """Low-frequency noise in 0..1, (H, W)."""
    g = rng.random((cells, max(2, int(cells * W / H)))).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


def make_background(rng: np.random.Generator, W: int, H: int, paper: tuple[int, int, int], style: str) -> tuple[np.ndarray, str]:
    """(H, W, 3) float32 0..255 background, and its kind."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    kinds = ["wood", "fabric", "solid", "light", "gradient", "blotch", "stripes", "tiles", "dots"] if style == "train" \
        else ["checker", "granite", "light"]
    kind = kinds[int(rng.integers(0, len(kinds)))]
    base = rng.integers(20, 235, 3).astype(np.float32)
    if kind == "wood":
        base = np.array([rng.uniform(90, 200), rng.uniform(55, 130), rng.uniform(20, 80)], np.float32)
        a = rng.uniform(-0.3, 0.3)
        f = rng.uniform(0.02, 0.08)
        warp = _smooth_noise(rng, H, W, 6) * rng.uniform(10, 40)
        s = np.sin((xx * np.cos(a) + yy * np.sin(a)) * f + warp) * 0.5 + 0.5
        g = s[..., None] * rng.uniform(0.15, 0.35)
        bg = base * (1 - g) + rng.normal(0, 4, (H, W, 1))
    elif kind == "fabric":
        f = rng.uniform(0.5, 1.5)
        weave = (np.sin(xx * f) * np.sin(yy * f)) * rng.uniform(5, 20)
        bg = base + weave[..., None] + rng.normal(0, rng.uniform(5, 18), (H, W, 1))
    elif kind == "solid":
        bg = np.broadcast_to(base, (H, W, 3)) + rng.normal(0, 3, (H, W, 1))
    elif kind == "light":  # desk close to paper colour: the hard case
        base = np.array(paper, np.float32) - rng.uniform(5, 45) + rng.normal(0, 6, 3)
        bg = np.broadcast_to(base, (H, W, 3)) + rng.normal(0, 3, (H, W, 1))
    elif kind == "gradient":
        b2 = rng.integers(20, 235, 3).astype(np.float32)
        a = rng.uniform(0, 2 * np.pi)
        t = (xx * np.cos(a) + yy * np.sin(a))
        t = ((t - t.min()) / (np.ptp(t) + 1e-6))[..., None]
        bg = base * (1 - t) + b2 * t
    elif kind == "blotch":
        n = _smooth_noise(rng, H, W, int(rng.integers(3, 10)))[..., None]
        b2 = rng.integers(20, 235, 3).astype(np.float32)
        bg = base * (1 - n) + b2 * n + rng.normal(0, 5, (H, W, 1))
    elif kind == "stripes":  # tablecloth or rug stripes, one colour often light
        b2 = np.array([rng.uniform(170, 245)] * 3, np.float32) + rng.normal(0, 10, 3)
        a = rng.uniform(0, np.pi)
        per = rng.uniform(15, 90)
        t = (((xx * np.cos(a) + yy * np.sin(a)) / per) % 1.0 < rng.uniform(0.3, 0.7))[..., None]
        bg = base * (1 - t) + b2 * t + rng.normal(0, 4, (H, W, 1))
    elif kind == "tiles":  # light floor or wall tiles with dark grout
        per = rng.uniform(40, 160)
        tile = np.array(paper, np.float32) - rng.uniform(10, 60) + rng.normal(0, 8, 3)
        grout = ((xx % per) < 3) | ((yy % per) < 3)
        bg = np.where(grout[..., None], tile * 0.55, tile) + rng.normal(0, 4, (H, W, 1))
    elif kind == "dots":
        per = rng.uniform(20, 60)
        r = per * rng.uniform(0.15, 0.35)
        d = np.hypot((xx % per) - per / 2, (yy % per) - per / 2) < r
        b2 = rng.integers(20, 256, 3).astype(np.float32)
        bg = np.where(d[..., None], b2, base) + rng.normal(0, 4, (H, W, 1))
    elif kind == "checker":
        p = rng.uniform(25, 70)
        c = ((xx // p + yy // p) % 2)[..., None]
        bg = base * (1 - c) + np.array([235, 235, 230], np.float32) * c
    else:  # granite
        bg = base * 0.4 + rng.normal(0, 30, (H, W, 1)) * (rng.random((H, W, 1)) < 0.3)
    bg = np.clip(bg, 0, 255).astype(np.float32)
    img = Image.fromarray(bg.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(0, 5))):  # clutter: phones, books, mugs, other papers at the edges
        cw, ch = rng.uniform(0.1, 0.35) * W, rng.uniform(0.1, 0.35) * H
        cx, cy = (rng.choice([rng.uniform(-0.1, 0.15), rng.uniform(0.85, 1.1)]) * W, rng.uniform(0, 1) * H) \
            if rng.random() < 0.5 else (rng.uniform(0, 1) * W, rng.choice([rng.uniform(-0.1, 0.15), rng.uniform(0.85, 1.1)]) * H)
        col = tuple(int(c) for c in rng.integers(0, 256, 3))
        box = [cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2]
        (d.ellipse if rng.random() < 0.3 else d.rectangle)(box, fill=col)
    return np.asarray(img, np.float32), kind


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src for 4 point pairs (x, y)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def _poly_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def poly_mask(q: np.ndarray, W: int, H: int, ss: int = 1) -> np.ndarray:
    """(H, W) float coverage 0..1 of polygon q (photo px); ss is supersampling for antialiasing."""
    im = Image.new("L", (W * ss, H * ss), 0)
    ImageDraw.Draw(im).polygon([(float(x) * ss, float(y) * ss) for x, y in q], fill=255)
    if ss > 1:
        im = im.resize((W, H), Image.BOX)
    return np.asarray(im, np.float32) / 255.0


def _place_quad(rng, W: int, H: int, aspect: float, style: str) -> np.ndarray:
    """Random page quad (TL TR BR BL) covering about 20..90 percent of the photo."""
    frac = rng.uniform(0.2, 0.9)
    cut = rng.random() < 0.25
    rot = rng.normal(0, 7) if rng.random() < 0.75 else rng.uniform(-35, 35)
    if W > H and rng.random() < 0.5:
        rot += 90 * rng.choice([-1, 1])
    if rng.random() < 0.05:
        rot += 180
    keystone = rng.uniform(0.7, 1.0) if style == "train" else rng.uniform(0.55, 0.8)
    jit = 0.03 if style == "train" else 0.05
    for _ in range(60):
        pw = np.sqrt(frac * W * H / aspect)
        ph = pw * aspect
        sq = np.array([[-pw / 2 * keystone, -ph / 2], [pw / 2 * keystone, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
        sq += rng.normal(0, jit * np.hypot(pw, ph), (4, 2))
        t = np.deg2rad(rot)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        q = sq @ R.T
        ext = q.max(0) - q.min(0)
        slack = np.array([W, H]) - ext
        c = np.array([W / 2, H / 2]) + rng.uniform(-0.5, 0.5, 2) * np.maximum(slack, 0)
        q = q + c - (q.max(0) + q.min(0)) / 2
        inside = _poly_area(_clip_poly(q, W, H)) / max(_poly_area(q), 1e-6)
        if (cut and 0.88 <= inside < 0.995) or (not cut and inside >= 0.995):
            return q
        if not cut or inside < 0.88:
            frac *= 0.95
        if cut and inside >= 0.995:  # push towards a border
            c2 = q + rng.choice([-1, 1], 2) * (np.maximum(-slack, 0) + rng.uniform(0.02, 0.08) * np.array([W, H]))
            inside2 = _poly_area(_clip_poly(c2, W, H)) / max(_poly_area(c2), 1e-6)
            if 0.88 <= inside2 < 0.995:
                return c2
    return q


def _clip_poly(q: np.ndarray, W: int, H: int) -> np.ndarray:
    """Sutherland-Hodgman clip of polygon q to the frame [0, W] x [0, H]."""
    pts = [tuple(p) for p in q]
    for axis, lim, keep_less in ((0, 0, False), (0, W, True), (1, 0, False), (1, H, True)):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            ina = (a[axis] <= lim) if keep_less else (a[axis] >= lim)
            inb = (b[axis] <= lim) if keep_less else (b[axis] >= lim)
            if inb:
                if not ina:
                    out.append(_isect(a, b, axis, lim))
                out.append(b)
            elif ina:
                out.append(_isect(a, b, axis, lim))
        pts = out
        if not pts:
            return np.zeros((0, 2))
    return np.array(pts, float)


def _isect(a, b, axis, lim):
    t = (lim - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def make_sample(seed: int, style: str = "train") -> Sample:
    """Generate one synthetic photo from an integer seed (deterministic)."""
    rng = np.random.default_rng([seed, 0 if style == "train" else 1])
    L = int(rng.integers(480, 1281))
    ar = float(rng.choice([4 / 3, 3 / 2, 16 / 9, 1.0]))
    portrait = rng.random() < 0.6
    W, H = (int(L / ar), L) if portrait else (L, int(L / ar))
    kinds = PAGE_KINDS if style == "train" else OOD_KINDS
    kind = str(kinds[int(rng.integers(0, len(kinds)))])
    aspect = float(rng.choice([1.414, 1.414, 1.294, rng.uniform(0.9, 2.0)])) if kind != "receipt" else rng.uniform(2.0, 2.8)
    if style == "train":
        paper = tuple(int(255 - rng.uniform(0, 25) - rng.uniform(0, 1) * np.array([0, 4, 15])[i]) for i in range(3)) \
            if rng.random() < 0.8 else tuple(int(c) for c in rng.integers(205, 250, 3))
    else:
        paper = (int(rng.integers(235, 256)), int(rng.integers(225, 245)), int(rng.integers(195, 225)))
    page = render_page(rng, kind, aspect, paper)
    pw, ph = page.size
    pnp = np.asarray(page, np.float32)
    fold = 0.85 + 0.15 * _smooth_noise(rng, ph, pw, 3)  # paper curl shading
    page = Image.fromarray(np.clip(pnp * fold[..., None], 0, 255).astype(np.uint8))

    q = _place_quad(rng, W, H, ph / pw, style)
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], float)
    Hm = _homography(q, src)  # photo -> page
    coeffs = (Hm / Hm[2, 2]).ravel()[:8]
    warped = np.asarray(page.transform((W, H), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC), np.float32)
    alpha = poly_mask(q, W, H, ss=3)

    bg, bg_kind = make_background(rng, W, H, paper, style)
    sh = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.005, 0.03) * L))
    off = rng.normal(0, 0.01 * L, 2).astype(int)
    shadow = np.roll(np.asarray(sh, np.float32) / 255.0, (off[1], off[0]), (0, 1))
    bg = bg * (1 - rng.uniform(0.1, 0.5) * shadow)[..., None]
    img = bg * (1 - alpha[..., None]) + warped * alpha[..., None]

    if rng.random() < 0.15:  # thumb or pen over a page edge
        e = q[int(rng.integers(0, 4))] * 0.5 + q[int(rng.integers(0, 4))] * 0.5
        occ = Image.new("L", (W, H), 0)
        r = rng.uniform(0.03, 0.07) * L
        ImageDraw.Draw(occ).ellipse([e[0] - r, e[1] - 1.6 * r, e[0] + r, e[1] + 1.6 * r], fill=255)
        o = np.asarray(occ, np.float32)[..., None] / 255.0
        col = np.array([rng.uniform(150, 220), rng.uniform(100, 160), rng.uniform(80, 130)], np.float32)
        img = img * (1 - o) + col * o

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    t = xx * np.cos(a) + yy * np.sin(a)
    t = (t - t.min()) / (np.ptp(t) + 1e-6)
    lo = rng.uniform(0.55, 1.0)
    light = lo + (1 - lo) * t
    if rng.random() < 0.3:  # soft shadow of the phone or hand
        light = light * (1 - rng.uniform(0.15, 0.4) * _smooth_noise(rng, H, W, 3) ** 2)
    cast = np.array([1.0, 1.0, 1.0]) + (rng.normal(0, 0.04, 3) if style == "train" else np.array([0.06, 0.0, -0.12]))
    img = img * light[..., None] * cast * rng.uniform(0.85, 1.1)
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    br = rng.uniform(0, 1.6) * L / 1000
    if br > 0.3:
        out = out.filter(ImageFilter.GaussianBlur(br))
    arr = np.asarray(out, np.float32) + rng.normal(0, rng.uniform(0, 7), (H, W, 1))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(rng.integers(35, 95)))
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    mask = poly_mask(q, W, H) >= 0.5
    return Sample(out, q.astype(float), mask, {"seed": seed, "style": style, "kind": kind, "data": DATA_LABEL,
                                                "page_frac": float(mask.mean()), "background": bg_kind,
                                                "cut": bool(mask.sum() < 0.995 * _poly_area(q))})
