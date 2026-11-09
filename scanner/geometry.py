"""From a page probability map to four page corners, and the perspective warp.

Steps: threshold, largest connected component, hole fill, sub-cell boundary points, convex hull reduced to
a 4-gon, robust line fit on each side (lines extend past corners cut off by the frame), optional edge
refinement on a higher-resolution image. Coordinates are continuous: pixel i covers [i, i + 1).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError


def clean_mask(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Largest 4-connected component of prob >= thr, holes filled. (gh, gw) bool."""
    m = ndi.binary_opening(prob >= thr, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))


def boundary_points(mask: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Sub-cell boundary points (x, y) where the mask changes between 4-neighbours (not at the frame)."""
    pts = []
    p = np.clip(prob, 0, 1)
    for axis in (0, 1):
        a, b = (mask[:-1], mask[1:]) if axis == 0 else (mask[:, :-1], mask[:, 1:])
        pa, pb = (p[:-1], p[1:]) if axis == 0 else (p[:, :-1], p[:, 1:])
        r, c = np.nonzero(a != b)
        va, vb = pa[r, c], pb[r, c]
        den = va - vb
        t = np.where(np.abs(den) > 1e-3, (va - thr) / np.where(np.abs(den) > 1e-3, den, 1), 0.5)
        t = np.clip(t, 0, 1)
        if axis == 0:
            pts.append(np.stack([c + 0.5, r + 0.5 + t], 1))
        else:
            pts.append(np.stack([c + 0.5 + t, r + 0.5], 1))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def _area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def reduce_to_quad(poly: np.ndarray) -> np.ndarray:
    """Greedily drop the vertex whose removal loses the least area until 4 remain."""
    return reduce_to_quad_n(poly, 4)


def _fit_line(pts: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Total least squares line: (point, unit direction)."""
    w = np.ones(len(pts)) if w is None else w
    c = (pts * w[:, None]).sum(0) / w.sum()
    d = pts - c
    cov = (d * w[:, None]).T @ d
    vals, vecs = np.linalg.eigh(cov)
    return c, vecs[:, 1]


def _intersect(l1, l2) -> np.ndarray | None:
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def order_corners(q: np.ndarray) -> np.ndarray:
    """Clockwise in image coordinates (y down), starting at the corner with the smallest x + y."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)


def _cross(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def is_convex(q: np.ndarray) -> bool:
    """True when the 4-gon is strictly convex."""
    s = [_cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    return all(v > 0 for v in s) or all(v < 0 for v in s)


def lines_to_quad(lines: list) -> np.ndarray | None:
    pts = [_intersect(lines[i - 1], lines[i]) for i in range(4)]
    if any(p is None for p in pts):
        return None
    return np.array(pts)


def _raster(q: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """(gh, gw) bool: cells whose centre is inside polygon q (grid coordinates)."""
    im = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(im).polygon([(float(x) - 0.5, float(y) - 0.5) for x, y in q], fill=1)
    return np.asarray(im, bool)


def _seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((pts - a) @ d) / (d @ d + 1e-12), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


def _support(q: np.ndarray, pts: np.ndarray, tol: float = 1.5) -> float:
    """Fraction of boundary points within tol cells of a side of q."""
    if len(pts) == 0:
        return 0.0
    d = np.min([_seg_dist(pts, q[i], q[(i + 1) % 4]) for i in range(4)], axis=0)
    return float((d < tol).mean())


def angles(q: np.ndarray) -> np.ndarray:
    """Interior angles of the quad, degrees."""
    out = []
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(out)


def _angle_penalty(q: np.ndarray, free: float = 25.0) -> float:
    """0 when every angle is within 90 +- free degrees, growing by 1 per 45 degrees beyond."""
    return float(np.maximum(np.abs(angles(q) - 90) - free, 0).sum() / 45)


def _soft_iou(q: np.ndarray, pm: np.ndarray) -> float:
    qm = _raster(q, pm.shape[1], pm.shape[0])
    inter = float(pm[qm].sum())
    return inter / (float(pm.sum()) + float(qm.sum()) - inter + 1e-9)


def _side_line(a: np.ndarray, b: np.ndarray, pts: np.ndarray, tol: float = 2.0):
    """Line through side a-b refitted on the boundary points near it (robust, two passes)."""
    d = b - a
    L = float(np.linalg.norm(d))
    d = d / max(L, 1e-9)
    n = np.array([-d[1], d[0]])
    rel = pts - a
    t = rel @ d
    sel = (np.abs(rel @ n) < tol) & (t > -0.1 * L) & (t < 1.1 * L)
    if sel.sum() < 4:
        return (a, d)
    line = _fit_line(pts[sel])
    for tl in (1.5, 1.0):
        p, dd = line
        nn = np.array([-dd[1], dd[0]])
        s2 = (np.abs((pts - p) @ nn) < tl) & (t > -0.1 * L) & (t < 1.1 * L)
        if s2.sum() >= 4:
            line = _fit_line(pts[s2])
    return line


def fit_quad(mask: np.ndarray, prob: np.ndarray, max_sides: int = 10) -> np.ndarray | None:
    """Quad (4, 2) in grid coordinates from a clean mask and the probability map, or None.

    Candidate lines come from the sides of the convex hull simplified to max_sides vertices, each refitted
    on nearby boundary points. Every 4 of them (in hull order) form a candidate quad; the best one
    maximises soft IoU with the probability map plus 0.5 x the fraction of boundary points on its sides.
    Lines extend past corners that the frame cuts off.
    """
    gh, gw = mask.shape
    if mask.sum() < 30:
        return None
    bp = boundary_points(mask, prob)
    fr = np.zeros_like(mask)
    fr[0], fr[-1], fr[:, 0], fr[:, -1] = mask[0], mask[-1], mask[:, 0], mask[:, -1]
    r, c = np.nonzero(fr)
    framepts = np.concatenate([np.stack([c + dx, r + dy], 1) for dx in (0, 1) for dy in (0, 1)]) if len(r) else np.zeros((0, 2))
    allp = np.concatenate([bp, framepts.astype(float)])
    if len(allp) < 8:
        return None
    try:
        hull = ConvexHull(allp)
    except QhullError:
        return None
    Hp = allp[hull.vertices]
    on_frame = (bp[:, 0] < 1.01) | (bp[:, 0] > gw - 1.01) | (bp[:, 1] < 1.01) | (bp[:, 1] > gh - 1.01)
    edge_pts = bp[~on_frame]
    pm = np.clip(prob, 0, 1) * ndi.binary_dilation(mask, iterations=3)
    span = np.array([gw, gh], float)

    def score(q):
        if q is None or not is_convex(q) or (q < -0.3 * span).any() or (q > 1.3 * span).any():
            return -1.0
        return _soft_iou(q, pm) + 0.5 * _support(q, edge_pts) - 0.5 * _angle_penalty(q)

    q0 = reduce_to_quad(Hp)
    best, best_s = q0, score(q0)
    P = reduce_to_quad_n(Hp, max_sides)
    lines = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if np.linalg.norm(b - a) < 2:
            continue
        ln = _side_line(a, b, edge_pts)
        dup = False
        for p2, d2 in lines:
            if abs(float(np.dot(ln[1], d2))) > np.cos(np.radians(4)) and abs(float(np.dot(ln[0] - p2, [-d2[1], d2[0]]))) < 1.5:
                dup = True
                break
        if not dup:
            lines.append(ln)
    if len(lines) >= 4:
        for idx in combinations(range(len(lines)), 4):
            q = lines_to_quad([lines[k] for k in idx])
            s_ = score(q)
            if s_ > best_s:
                best, best_s = q, s_
    # final robust refit of each side on its own boundary points
    best = order_corners(best)
    q = lines_to_quad([_side_line(best[i], best[(i + 1) % 4], edge_pts, 1.5) for i in range(4)])
    if score(q) >= best_s - 0.01:
        best = order_corners(q)
    return best


def reduce_to_quad_n(poly: np.ndarray, n: int) -> np.ndarray:
    """Greedy vertex removal (least area loss) down to n vertices."""
    P = [tuple(p) for p in poly]
    while len(P) > n:
        losses = []
        for i in range(len(P)):
            a, b, c = np.array(P[i - 1]), np.array(P[i]), np.array(P[(i + 1) % len(P)])
            losses.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
        P.pop(int(np.argmin(losses)))
    return np.array(P, float)


def refine_edges(q: np.ndarray, img: np.ndarray, search: float) -> np.ndarray:
    """Refine a quad (photo px of img) by snapping each side to the strongest colour edge within +-search px.

    img: (H, W, 3) float, lightly smoothed. Returns the refined quad, or q unchanged when a side has too
    little support or moves too far.
    """
    H, W = img.shape[:2]
    gy = np.stack([ndi.sobel(img[..., k], 0) for k in range(3)], -1)
    gx = np.stack([ndi.sobel(img[..., k], 1) for k in range(3)], -1)
    c0 = q.mean(0)
    offs = np.arange(-search, search + 0.01, 0.5)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot((a + b) / 2 - c0, n) < 0:
            n = -n
        ts = np.linspace(0.1, 0.9, 40)
        base = a[None] + ts[:, None] * (b - a)[None]
        P = base[:, None, :] + offs[None, :, None] * n[None, None, :]  # (40, k, 2)
        xs, ys = P[..., 0] - 0.5, P[..., 1] - 0.5
        ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
        coords = [ys.ravel(), xs.ravel()]
        gxn = np.stack([ndi.map_coordinates(gx[..., k], coords, order=1) for k in range(3)], -1)
        gyn = np.stack([ndi.map_coordinates(gy[..., k], coords, order=1) for k in range(3)], -1)
        g = np.linalg.norm(gxn * n[0] + gyn * n[1], axis=-1).reshape(P.shape[:2])
        g = np.where(ok, g, 0) * np.exp(-0.5 * (offs[None] / (0.6 * search)) ** 2)
        k = np.argmax(g, 1)
        strong = g[np.arange(len(ts)), k] > 2.0 * np.median(g, 1) + 1e-6
        valid = strong & ok[np.arange(len(ts)), k]
        line0 = (a, d)
        if valid.sum() < 12:
            lines.append(line0)
            continue
        pts = P[np.arange(len(ts)), k][valid]
        line = _fit_line(pts)
        for _ in range(2):
            p, dd = line
            nn = np.array([-dd[1], dd[0]])
            res = np.abs((pts - p) @ nn)
            keep = res < max(1.0, 2.5 * np.median(res))
            if keep.sum() < 10:
                break
            line = _fit_line(pts[keep])
        p, dd = line
        ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(dd, d))))))
        shift = abs(float(np.dot(p - a, n)))
        lines.append(line if ang < 4 and shift < search else line0)
    q2 = lines_to_quad(lines)
    if q2 is None or not is_convex(q2) or np.abs(q2 - q).max() > 3 * search:
        return q
    return order_corners(q2)


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src for 4 point pairs (x, y)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


A4 = 2 ** 0.5
MAX_OUT = 2000  # px, long side of the output page


def output_size(q: np.ndarray) -> tuple[int, int]:
    """(w, h) of the rectified page from the side lengths; snapped to A4 when within 15 percent."""
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bot, left = np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0])
    w, h = max((top + bot) / 2, 8), max((left + right) / 2, 8)
    r = h / w
    for target in (A4, 1 / A4):
        if abs(r / target - 1) < 0.15:
            r = target
    long = max(w, h)
    w, h = (long / r, long) if r >= 1 else (long, long * r)
    s = min(1.0, MAX_OUT / max(w, h))
    return max(8, int(round(w * s))), max(8, int(round(h * s)))


def warp(img: Image.Image, q: np.ndarray) -> Image.Image:
    """Rectify quad q (TL TR BR BL, photo px) of img to a flat page."""
    w, h = output_size(q)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    Hm = homography(dst, q)
    return img.transform((w, h), Image.PERSPECTIVE, tuple((Hm / Hm[2, 2]).ravel()[:8]), Image.BICUBIC)


def corner_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean corner distance as a percentage of the true page diagonal, minimised over cyclic order."""
    diag = max(np.linalg.norm(true[2] - true[0]), np.linalg.norm(true[3] - true[1]))
    best = np.inf
    for t in (true, true[::-1]):
        for s in range(4):
            best = min(best, float(np.linalg.norm(pred - np.roll(t, s, 0), axis=1).mean()))
    return 100.0 * best / diag
