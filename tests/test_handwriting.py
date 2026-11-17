"""Fast tests for the handwriting and orientation upgrade (no datasets; model tests skip without weights)."""
from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # optional (requirements-ocr.txt): skip, do not fail, without it
torch = pytest.importorskip("torch")

from ocr import orientation as O  # noqa: E402
from ocr import scriptclf  # noqa: E402
from ocr.hwdata import rect_crop  # noqa: E402


def _text_page(h: int = 600, w: int = 800) -> np.ndarray:
    """A white page with ten lines of black printed text."""
    img = np.full((h, w, 3), 255, np.uint8)
    for i in range(10):
        cv2.putText(img, "The quick brown fox jumps over", (40, 60 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


# ---------------------------------------------------------------- orientation

@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_upright_undoes_counter_clockwise_turns(k):
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(O.upright(np.ascontiguousarray(np.rot90(img, k)), k), img)


def test_letterbox_is_square_and_keeps_content_centred():
    out = O.letterbox(np.full((100, 300, 3), 200, np.uint8), size=64)
    assert out.shape == (64, 64, 3)
    assert out[32, 32].tolist() == [200, 200, 200] and out[0, 32].tolist() == [0, 0, 0]


def test_logits_k_maps_doctr_classes_to_quarter_turns():
    class Fake(torch.nn.Module):  # one distinct logit per docTR class, in docTR's order [0, -90, 180, 90]
        def forward(self, x):
            return torch.tensor([[0.0, 1.0, 2.0, 3.0]]).repeat(len(x), 1)
    out = O.logits_k(Fake(), torch.zeros(1, 3, 8, 8))
    # docTR signs angles clockwise-positive, so its "-90" is k=1 (one turn counter-clockwise).
    # Pinned here because the validated accuracy on 90 / 270 pages depends on it.
    assert out[0].tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("angle", [-8.0, 5.0])
def test_deskew_recovers_small_rotation(angle):
    page = _text_page()
    fixed, est = O.deskew(O.rotate(page, angle))
    assert abs(est + angle) < 1.0  # the correction is the opposite of the applied skew
    assert fixed.ndim == 3


def test_deskew_leaves_straight_page_alone():
    page = _text_page()
    fixed, est = O.deskew(page)
    assert est == 0.0 and np.array_equal(fixed, page)


def test_flip_margin_sign_follows_recogniser_confidence():
    img = np.zeros((40, 200, 3), np.uint8)
    img[:, :100] = 255  # crops are bright on the left when upright
    boxes = np.array([[0, 0, 200, 40], [0, 0, 150, 40]], float)

    def rec(crops):  # pretends to read confidently only when the left edge is bright
        return [("x", 0.9 if c[:, 0].mean() > 128 else 0.2) for c in crops]
    d, n = O.flip_margin(rec, img, boxes)
    assert n == 2 and d == pytest.approx(0.7)
    d2, _ = O.flip_margin(rec, np.ascontiguousarray(img[::-1, ::-1]), boxes)
    assert d2 == pytest.approx(-0.7)
    assert O.flip_margin(rec, img, np.zeros((0, 4)))[1] == 0


# ---------------------------------------------------------------- routing and crops

def test_script_prep_shape_and_range():
    x = scriptclf.prep(np.full((20, 50, 3), 128, np.uint8))
    assert x.shape == (32, 96) and x.dtype == np.uint8


def test_script_net_runs_on_a_batch():
    model = scriptclf.ScriptNet().eval()
    p = scriptclf.handwritten_prob(model, [np.zeros((10, 30, 3), np.uint8), np.full((40, 90, 3), 255, np.uint8)])
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))


def test_rect_crop_of_axis_aligned_box():
    img = np.zeros((100, 200, 3), np.uint8)
    img[20:40, 50:150] = 255
    pts = np.array([[50, 20], [150, 20], [150, 40], [50, 40]], float)
    c = rect_crop(img, pts, pad=0.0)
    assert abs(c.shape[1] - 100) <= 2 and abs(c.shape[0] - 20) <= 2
    assert c.mean() > 200


# ---------------------------------------------------------------- synthetic handwriting

def test_synthetic_line_is_seeded():
    from ocr import synth_hw
    fonts = synth_hw.font_files()
    if not fonts:
        pytest.skip("handwriting fonts not downloaded (scripts/download_handwriting.py --only fonts)")
    a = synth_hw.render_line("hello world", np.random.default_rng(1), fonts)
    b = synth_hw.render_line("hello world", np.random.default_rng(1), fonts)
    assert a.dtype == np.uint8 and a.ndim in (2, 3) and np.array_equal(a, b)


def test_prep_line_returns_grey_rgb_of_same_size():
    pytest.importorskip("transformers")
    from ocr.handwriting import prep_line
    crop = np.random.default_rng(0).integers(0, 255, (48, 160, 3), dtype=np.uint8)
    out = np.asarray(prep_line(crop))
    assert out.shape[:2] == (48, 160)


# ---------------------------------------------------------------- post-correction

def test_postcorrect_fixes_one_typo_and_leaves_numbers():
    pytest.importorskip("symspellpy")
    from ocr.postcorrect import correct_line
    assert correct_line("the quik brown 1234") == "the quick brown 1234"
    out = correct_line("meeting tomorow at 10:30")
    assert out.endswith("at 10:30") and "tomorrow" in out


# ---------------------------------------------------------------- end to end (needs the trained weights)

HAVE_WEIGHTS = all(os.path.exists(p) for p in (O.ORIENT_CKPT, scriptclf.CKPT,
                                                os.path.join(O.ROOT, "checkpoints", "handwriting_v1", "config.json")))


@pytest.mark.skipif(not HAVE_WEIGHTS or os.environ.get("OCR_SLOW") != "1",
                    reason="slow: set OCR_SLOW=1 (needs checkpoints/ and the docTR weights)")
def test_scan_reads_rotated_page(tmp_path):
    from PIL import Image
    from ocr.scan import scan
    page = _text_page()
    p = tmp_path / "turned.png"
    Image.fromarray(np.ascontiguousarray(np.rot90(page, 1))).save(p)
    text, conf = scan(str(p))
    assert "quick" in text.lower() and len(conf) == len(text.splitlines())


@pytest.mark.skipif(not HAVE_WEIGHTS or os.environ.get("OCR_SLOW") != "1",
                    reason="slow: set OCR_SLOW=1 (needs checkpoints/ and the docTR weights)")
def test_scan_reads_handwriting_line_page(tmp_path):
    """A page of synthetic handwriting lines is routed to the handwriting reader and read."""
    from PIL import Image
    from ocr import synth_hw
    from ocr.scan import Scanner
    fonts = synth_hw.font_files()
    if not fonts:
        pytest.skip("handwriting fonts not downloaded")
    rng = np.random.default_rng(3)
    lines = [synth_hw.render_line(t, rng, fonts) for t in ("meeting notes for tuesday", "call the bank about the loan")]
    lines = [np.repeat(x[..., None], 3, 2) if x.ndim == 2 else x for x in lines]
    w = max(x.shape[1] for x in lines) + 80
    page = np.full((sum(x.shape[0] for x in lines) + 200, w, 3), 255, np.uint8)
    y = 60
    for x in lines:
        page[y:y + x.shape[0], 40:40 + x.shape[1]] = x
        y += x.shape[0] + 40
    r = Scanner().run(Image.fromarray(page))
    assert any(ln["script"] == "handwritten" for ln in r["lines"])
    assert "bank" in r["text"].lower() or "meeting" in r["text"].lower()


def test_to_rgb_scales_16_bit_grey():
    from PIL import Image
    from ocr.pipeline import to_rgb
    a = (np.linspace(0, 65535, 64 * 64).reshape(64, 64)).astype(np.uint16)
    out = np.asarray(to_rgb(Image.fromarray(a)))
    assert out.dtype == np.uint8 and out.shape == (64, 64, 3)
    assert out[0, 0, 0] == 0 and out[-1, -1, 0] == 255 and 100 < out[32, 32, 0] < 160


def test_scanner_refuses_lexicon_correction():
    """It rewrites medicine names and units (digoxin -> dioxin, mmol -> mol), so it cannot be switched on."""
    from ocr.scan import Scanner
    with pytest.raises(ValueError, match="medicine names and units"):
        Scanner(postcorrect=True)
