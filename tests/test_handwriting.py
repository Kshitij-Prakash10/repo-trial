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

def _missing_weights() -> list[str]:
    """What the end-to-end tests need and this computer lacks: the v2 checkpoints, and the detector and
    recogniser weights with the size rule of ocr.api (an empty cache file is missing)."""
    from ocr.api import missing_weights
    v2 = {"orientation": O.ORIENT_CKPT, "routing": scriptclf.CKPT,
          "handwriting": os.path.join(O.ROOT, "checkpoints", "handwriting_v1", "config.json")}
    return [k for k, p in v2.items() if not os.path.exists(p)] + missing_weights()


MISSING_WEIGHTS = _missing_weights()
HAVE_WEIGHTS = not MISSING_WEIGHTS
SLOW = os.environ.get("OCR_SLOW") == "1"
SLOW_REASON = ("slow: set OCR_SLOW=1 to run it" if not SLOW else
               f"model weights missing on this computer: {', '.join(MISSING_WEIGHTS)}")


@pytest.mark.skipif(not HAVE_WEIGHTS or not SLOW, reason=SLOW_REASON)
def test_scan_reads_rotated_page(tmp_path):
    from PIL import Image
    from ocr.scan import scan
    page = _text_page()
    p = tmp_path / "turned.png"
    Image.fromarray(np.ascontiguousarray(np.rot90(page, 1))).save(p)
    text, conf = scan(str(p))
    assert "quick" in text.lower() and len(conf) == len(text.splitlines())


@pytest.mark.skipif(not HAVE_WEIGHTS or not SLOW, reason=SLOW_REASON)
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


# ---------------------------------------------------------------- line crops: narrow words, neighbouring lines

def test_rect_crop_keeps_a_narrow_single_word_upright():
    """A 15 x 40 word box (taller than wide, e.g. "I" or "1") used to be turned 90 degrees."""
    img = np.full((200, 200, 3), 255, np.uint8)
    img[52:88, 57:59] = 0  # a vertical stroke, like the letter I
    box = np.array([[50, 50], [65, 50], [65, 90], [50, 90]], float)
    c = rect_crop(img, box)
    assert c.shape[0] > c.shape[1]  # still taller than wide
    dark_rows = np.flatnonzero((c.mean(2) < 128).any(1))
    assert dark_rows[-1] - dark_rows[0] > 25  # the stroke is still vertical


@pytest.mark.parametrize("deg", [-30, -8, 0, 12, 40])
def test_rect_crop_turns_a_tilted_line_horizontal(deg):
    img = np.full((400, 600, 3), 255, np.uint8)
    a = np.radians(deg)
    d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c0 = np.array([300.0, 200.0])
    for x, y in np.array([c0 + t * d for t in np.linspace(-150, 150, 400)]).round().astype(int):
        img[y - 1:y + 2, x - 1:x + 2] = 0
    words = [np.array([c0 + s * d * 150 + u * d * 140 + v * n * 12 for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
             for s in (-0.5, 0.5)]  # two words of 280 x 24 px along the line
    crop = rect_crop(img, np.concatenate(words))
    assert crop.shape[1] > 4 * crop.shape[0]
    rows = np.flatnonzero((crop.mean(2) < 128).mean(1) > 0.5)
    assert len(rows) and rows[-1] - rows[0] <= 4  # the ink is one horizontal band


def _two_line_page(with_b: bool = True):
    """A gappy line A (two words, the second lower) and a word of line B under A's first word."""
    img = np.full((200, 400, 3), 230, np.uint8)  # grey-white paper
    a1, a2 = [20, 40, 170, 70], [200, 80, 380, 110]
    b1 = [30, 85, 160, 112]  # the next line's word, inside line A's rectangle
    for x0, y0, x1, y1 in (a1, a2, b1) if with_b else (a1, a2):
        img[y0 + 8:y1 - 8, x0 + 5:x1 - 5] = 20
    quads = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) for x0, y0, x1, y1 in (a1, a2)]
    return img, quads


def test_masking_paints_out_a_neighbouring_line_only_when_asked():
    img, quads = _two_line_page()
    pts = np.concatenate(quads)
    plain = rect_crop(img, pts)
    masked = rect_crop(img, pts, words=quads)
    only_a = rect_crop(_two_line_page(with_b=False)[0], pts)  # the crop if line B were not there
    assert plain.shape == masked.shape == only_a.shape
    ink = lambda c: int((c.mean(2) < 128).sum())
    assert ink(plain) > ink(only_a) + 300  # without masking, part of line B's word is in the crop
    assert abs(ink(masked) - ink(only_a)) <= 0.05 * ink(only_a)  # with masking only line A's ink is left
    painted = (np.abs(masked.astype(int) - plain.astype(int)).sum(2) > 0)
    assert np.all(np.abs(masked[painted].astype(int) - 230) <= 2)  # painted with the paper colour, not white
    assert np.array_equal(rect_crop(img, pts, words=None), plain)


# ---------------------------------------------------------------- reading choice and safety flag (no models)

def test_dash_line_keeps_the_printed_reading():
    from ocr.scan import keep_printed
    assert keep_printed("-----------", "sentence Valuable")[0] is True
    assert keep_printed("- - - - -", "sentence Valuable")[0] is True
    assert keep_printed("......", "Mr Smith")[0] is True
    assert keep_printed("==", "an")[0] is True
    assert keep_printed("-", "Mr Smith")[0] is False  # a single character is not a rule
    assert keep_printed("Mr Snith", "Mr Smith")[0] is False
    assert keep_printed("2-3", "two to three")[0] is False
    assert keep_printed("-----", "----")[0] is False  # nothing alphabetic to replace it with


def test_rule_line_crop_is_recognised_from_its_ink():
    from ocr.scan import keep_printed, looks_like_rule
    rule = np.full((40, 300, 3), 240, np.uint8)
    for x in range(10, 290, 12):
        rule[19:22, x:x + 8] = 30  # a row of dashes
    text = np.full((40, 300, 3), 240, np.uint8)
    text[8:32, 10:60] = 30
    text[14:28, 70:200] = 30
    assert looks_like_rule(rule) and not looks_like_rule(text)
    assert not looks_like_rule(np.full((40, 300, 3), 240, np.uint8))  # blank
    assert keep_printed("", "sentence Valuable", rule)[0] is True  # printed saw nothing, the ink is a rule
    assert keep_printed("", "meeting notes", text)[0] is False
    assert keep_printed("ok", "sentence Valuable", rule)[0] is False  # the printed reader saw letters


@pytest.mark.parametrize("line,flag", [
    ({"text": "Ramipril 2.5 mg", "handwriting_text": "Ramipril 2.5 mg"}, True),
    ({"text": "Ramipril Z.S my", "handwriting_text": "Ramipril Z.S my", "printed_text": "Ramipril 2.5 mg"}, True),
    ({"text": "Ramipril 2.5", "handwriting_text": "Ramipril Z.S my"}, True),  # printed kept, hw lost the digits
    ({"text": "Ramipril Z.S", "handwriting_text": "Ramipril 25 mg"}, True),  # only the hw reader saw them
    ({"text": "call the bank", "handwriting_text": "call the bank"}, False),
    ({"text": "Ramipril 2.5 mo"}, True),  # missed unit and line never routed to handwriting
    ({"text": "InSulin O Units"}, True),  # missed digits and title-case unit, original regression
    ({"text": "call the bank"}, False),
])
def test_confirm_numbers_looks_at_every_reading(line, flag):
    from ocr.scan import confirm_numbers
    assert confirm_numbers(line) is flag


@pytest.mark.parametrize("text,flag,misread", [
    ("Haemoglobin 13.2 g/dL 12.0-15.5", False, False),  # printed numbers: the quieter has_numbers only
    ("1. Tab Amlodipine 5 mg OD x 30 days", False, False),
    ("Omeprazole mg", True, True),  # a unit whose number was lost
    ("Insulin glargine IZ Units", True, True),
    ("Metformin SOO my twice daily", False, True),  # misread digits: flagged only when asked for
    ("Mehta OD Clinic, Station Road", False, False),
])
def test_printed_lines_are_flagged_only_when_a_number_looks_lost(text, flag, misread):
    from ocr.common import has_number_or_unit
    from ocr.scan import confirm_numbers
    line = {"text": text, "script": "printed"}
    assert confirm_numbers(line) is flag
    assert confirm_numbers(line, misread_digits=True) is misread
    # without printed / handwritten routing the line may be handwriting: any number or unit counts
    assert confirm_numbers(line, script_known=False) is has_number_or_unit(text)


class _FakeOcr:
    """Stands in for OcrPipeline: one page with given printed lines (one word per line)."""

    def __init__(self, page, lines):
        self.page, self.lines, self.rec = page, lines, None

    def run(self, image, keep_page=False):
        words, lines = [], []
        for k, (text, conf, box) in enumerate(self.lines):
            words.append({"text": text, "confidence": conf, "box": box, "line": k})
            lines.append({"text": text, "words": [k], "box": box, "confidence": conf})
        return {"text": "\n".join(t for t, _, _ in self.lines), "lines": lines, "words": words,
                "page_image": self.page, "page": {"width": self.page.shape[1], "height": self.page.shape[0],
                                                  "is_scanner_crop": False},
                "source": {}, "scanner": {}, "models": {"detector": "fake", "recogniser": "fake"}}


class _FakeHw:
    name = "fake-trocr"

    def __init__(self, readings):
        self.readings, self.crops = readings, None

    def read(self, crops):
        self.crops = crops
        return self.readings[:len(crops)]


def test_scanner_keeps_dashes_and_flags_lost_digits(monkeypatch):
    """Scanner.run with fake readers: the dash line of IAM test 02658 and a dose whose digits the
    handwriting reader turned into letters."""
    from PIL import Image
    from ocr import scan as S
    page = np.full((300, 600, 3), 235, np.uint8)
    printed = [("------------------", 0.31, [20, 30, 580, 55]),
               ("Ramipril 2.5 mg", 0.42, [20, 100, 400, 140]),
               ("call the bank", 0.40, [20, 200, 400, 240])]
    handwritten = [("sentence Valuable", 0.6255), ("Ramipril Z.S my", 0.71), ("call the bank", 0.80)]
    sc = object.__new__(S.Scanner)
    sc.ocr, sc.hw = _FakeOcr(page, printed), _FakeHw(handwritten)
    sc.orient, sc.do_deskew, sc.flip_check, sc.clf, sc.postcorrect = None, False, True, object(), False
    sc.mask_neighbours = False
    monkeypatch.setattr(S.scriptclf, "handwritten_prob", lambda clf, crops: np.ones(len(crops)))
    r = sc.run(Image.new("RGB", (600, 300), "white"))
    by = {ln["handwriting_text"]: ln for ln in r["lines"]}
    dash = by["sentence Valuable"]
    assert dash["text"] == "------------------" and dash["script"] == "printed" and dash["handwriting_rejected"]
    dose = by["Ramipril Z.S my"]
    assert dose["text"] == "Ramipril Z.S my" and dose["printed_text"] == "Ramipril 2.5 mg"
    assert dose["confirm_numbers"] is True  # the shown text lost its digits; the printed reading has them
    assert by["call the bank"]["confirm_numbers"] is False and by["call the bank"]["script"] == "handwritten"
    assert len(sc.hw.crops) == 3


@pytest.mark.parametrize("routing", [True, False])
def test_scanner_gives_printed_numbers_the_quieter_flag(monkeypatch, routing):
    """Printed lines (never sent to the handwriting reader) with numbers get has_numbers, not
    confirm_numbers; without routing nobody checked the script, so they get both."""
    from PIL import Image
    from ocr import scan as S
    page = np.full((300, 600, 3), 235, np.uint8)
    printed = [("Haemoglobin 13.2 g/dL", 0.9, [20, 30, 580, 55]), ("Omeprazole mg", 0.9, [20, 100, 400, 140]),
               ("Dr L Example", 0.9, [20, 200, 400, 240])]
    sc = object.__new__(S.Scanner)
    sc.ocr, sc.hw = _FakeOcr(page, printed), (_FakeHw([]) if routing else None)
    sc.orient, sc.do_deskew, sc.flip_check, sc.clf, sc.postcorrect = None, False, True, (object() if routing else None), False
    sc.mask_neighbours = False
    monkeypatch.setattr(S.scriptclf, "handwritten_prob", lambda clf, crops: np.zeros(len(crops)))
    r = sc.run(Image.new("RGB", (600, 300), "white"))
    flags = [(ln["has_numbers"], ln["confirm_numbers"]) for ln in r["lines"]]
    assert all(ln["script"] == "printed" for ln in r["lines"])
    assert flags == ([(True, False), (True, True), (False, False)] if routing
                     else [(True, True), (True, True), (False, False)])


def test_confirm_numbers_keeps_its_old_call_and_takes_the_option_by_name():
    from ocr.scan import confirm_numbers
    ln = {"text": "Metformin twice daily", "script": "printed", "handwriting_text": "Metformin SOO my"}
    assert confirm_numbers(ln) is False and confirm_numbers(ln, True, False) is False
    assert confirm_numbers(ln, misread_digits=True) is True  # the older keyword
    assert confirm_numbers(ln, script_known=True, flag_misread_digits=True) is True
    assert confirm_numbers(ln, script_known=True, flag_misread_digits=False) is False
    assert confirm_numbers({"text": "1. Omeprazole mg", "script": "printed"}, script_known=True) is True
    assert confirm_numbers({"text": "1. Omeprazole 20 mg", "script": "printed"}, script_known=True) is False


@pytest.mark.parametrize("flag", [False, True])
def test_scanner_looks_for_misread_digits_in_every_reading_when_asked(monkeypatch, flag):
    """Scanner.run with fake readers, "Metformin SOO my" in each position: (a) printed only, (b) the
    handwriting reading shown (misread in the shown text or in printed_text), (c) the printed reading
    kept after handwriting (misread shown or in handwriting_text). The shown text stays as read."""
    from PIL import Image
    from ocr import scan as S
    M, P = "Metformin SOO my", "Metformin twice daily"
    page = np.full((500, 600, 3), 235, np.uint8)
    printed = [(M, 0.9, [20, 20, 400, 60]),  # (a): not routed
               (P, 0.30, [20, 100, 400, 140]),  # (b) hw reading M is surer: shown
               (M, 0.30, [20, 180, 400, 220]),  # (b) hw reading P is surer: M goes to printed_text
               (M, 0.90, [20, 260, 400, 300]),  # (c) printed M kept, hw P in handwriting_text
               (P, 0.90, [20, 340, 400, 380])]  # (c) printed P kept, hw M in handwriting_text
    hw = [(M, 0.8), (P, 0.8), (P, 0.5), (M, 0.5)]
    sc = object.__new__(S.Scanner)
    sc.ocr, sc.hw = _FakeOcr(page, printed), _FakeHw(hw)
    sc.orient, sc.do_deskew, sc.flip_check, sc.clf, sc.postcorrect = None, False, True, object(), False
    sc.mask_neighbours, sc.flag_misread_digits = False, flag
    monkeypatch.setattr(S.scriptclf, "handwritten_prob",
                        lambda clf, crops: np.array([0.0] + [1.0] * (len(crops) - 1)))
    r = sc.run(Image.new("RGB", (600, 500), "white"))
    got = [(ln["text"], ln["script"], ln.get("handwriting_text"), ln.get("printed_text"), ln["confirm_numbers"])
           for ln in r["lines"]]
    assert got == [(M, "printed", None, None, flag), (M, "handwritten", M, P, flag),
                   (P, "handwritten", P, M, flag), (M, "printed", P, None, flag), (P, "printed", M, None, flag)]


def test_handwriting_beam_width_is_an_option_and_greedy_by_default(monkeypatch):
    """HandwritingReader(num_beams=...) sets the decoder's beam width for read(); 1 (greedy) by default, and
    read(num_beams=...) still overrides it. Scanner(hw_beams=...) passes it on (experiment, opt-in)."""
    import inspect
    from ocr import handwriting as H
    from ocr import scan as S
    assert inspect.signature(H.HandwritingReader.__init__).parameters["num_beams"].default == 1
    assert inspect.signature(S.Scanner.__init__).parameters["hw_beams"].default == 1
    seen = []

    class Gen:
        def __init__(self, n):
            self.sequences = torch.ones((n, 3), dtype=torch.long)
            self.scores = [torch.zeros((n, 5))] * 2

    class Model:
        def generate(self, px, num_beams, **kw):
            seen.append(num_beams)
            return Gen(px.shape[0])

        def compute_transition_scores(self, seq, scores, beam_indices, normalize_logits):
            return torch.zeros((seq.shape[0], 2))

    class Proc:
        tokenizer = type("T", (), {"pad_token_id": 0})()

        def __call__(self, images, return_tensors):
            return type("P", (), {"pixel_values": torch.zeros((len(images), 3, 4, 4))})()

        def batch_decode(self, seq, skip_special_tokens):
            return ["x"] * seq.shape[0]

    r = object.__new__(H.HandwritingReader)
    r.model, r.processor = Model(), Proc()
    line = np.full((20, 60, 3), 255, np.uint8)
    r.read([line])  # an old pickled / hand-built reader without the attribute: greedy
    r.num_beams = 2
    r.read([line])
    r.read([line], num_beams=4)
    assert seen == [1, 2, 4]
    made = []
    monkeypatch.setattr(S, "HandwritingReader", lambda **kw: made.append(kw) or object())
    for name in ("OcrPipeline",):
        monkeypatch.setattr(S, name, type("P", (), {"load": staticmethod(lambda *a, **k: object())}))
    monkeypatch.setattr(S.O, "load_orient", lambda: object())
    monkeypatch.setattr(S.scriptclf, "load_clf", lambda: object())
    S.Scanner()
    S.Scanner(hw_beams=2)
    assert made == [{"num_beams": 1}, {"num_beams": 2}]


def test_scanner_checks_misread_digits_by_default_and_can_be_told_not_to(monkeypatch):
    """The v2 Scanner ships with the misread-digit check on (ocr.common.FLAG_MISREAD_DIGITS), like v1;
    Scanner(flag_misread_digits=False) or the switch turn it off. The shown text stays as read."""
    from PIL import Image
    from ocr import common
    from ocr import scan as S
    for name in ("OcrPipeline",):
        monkeypatch.setattr(S, name, type("P", (), {"load": staticmethod(lambda *a, **k: object())}))
    monkeypatch.setattr(S.O, "load_orient", lambda: object())
    monkeypatch.setattr(S.scriptclf, "load_clf", lambda: object())
    monkeypatch.setattr(S, "HandwritingReader", lambda **kw: object())
    assert S.Scanner().flag_misread_digits is True
    assert S.Scanner(flag_misread_digits=False).flag_misread_digits is False
    page = np.full((200, 600, 3), 235, np.uint8)
    printed = [("Metformin SOO my", 0.9, [20, 20, 400, 60]), ("Review in clinic", 0.9, [20, 100, 400, 140])]

    def run(setting="unset"):
        sc = object.__new__(S.Scanner)
        sc.ocr, sc.hw = _FakeOcr(page, printed), _FakeHw([])
        sc.orient, sc.do_deskew, sc.flip_check, sc.clf, sc.postcorrect = None, False, True, object(), False
        sc.mask_neighbours = False
        if setting != "unset":
            sc.flag_misread_digits = setting
        monkeypatch.setattr(S.scriptclf, "handwritten_prob", lambda clf, crops: np.zeros(len(crops)))
        r = sc.run(Image.new("RGB", (600, 200), "white"))
        assert [ln["text"] for ln in r["lines"]] == ["Metformin SOO my", "Review in clinic"]
        return [ln["confirm_numbers"] for ln in r["lines"]]

    assert run() == [True, False] and run(True) == [True, False] and run(False) == [False, False]
    monkeypatch.setattr(common, "FLAG_MISREAD_DIGITS", False)
    assert run() == [False, False] and S.Scanner().flag_misread_digits is False
