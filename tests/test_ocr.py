"""Fast unit tests for the OCR code (no datasets, no model weights)."""
from __future__ import annotations

import numpy as np
import pytest

from ocr import common
from ocr.data import split_line


# ---------------------------------------------------------------- metrics

def test_normalise_collapses_whitespace_and_keeps_case():
    assert common.normalise("  Total \n  9.00\tRM ") == "Total 9.00 RM"


def test_error_counts_and_corpus_rates():
    pytest.importorskip("rapidfuzz")
    a = common.error_counts("helo world", "hello world")  # one missing character, one wrong word
    assert (a["char_err"], a["char_n"], a["word_err"], a["word_n"]) == (1, 11, 1, 2)
    b = common.error_counts("abc", "abc")
    r = common.rates([a, b])
    assert r["cer"] == round(1 / 14, 4) and r["wer"] == round(1 / 3, 4)  # total edits / total length


def test_rates_on_empty_reference():
    assert common.rates([])["cer"] is None


def test_iou_and_matching_are_one_to_one():
    true = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]], float)
    m = common.iou_matrix(pred, true)
    assert m[0, 0] == pytest.approx(1.0) and m[2].max() == 0
    pairs = common.match_boxes(pred, true)
    assert pairs == [(0, 0)]  # the second prediction overlaps the same box and must not match again
    assert common.prf(1, 3, 2) == {"precision": 0.3333, "recall": 0.5, "f1": 0.4, "tp": 1, "n_pred": 3, "n_true": 2}


def test_matching_respects_threshold_and_empty_inputs():
    a, b = np.array([[0, 0, 10, 10]], float), np.array([[6, 0, 16, 10]], float)  # IoU 4/16
    assert common.match_boxes(a, b) == []
    assert common.match_boxes(np.zeros((0, 4)), b) == [] and common.match_boxes(a, np.zeros((0, 4))) == []
    assert common.prf(0, 0, 0)["f1"] == 0.0


# ---------------------------------------------------------------- reading order

def test_group_lines_orders_top_to_bottom_left_to_right():
    boxes = np.array([[60, 52, 100, 62],   # line 2, right
                      [10, 10, 50, 20],    # line 1, left
                      [10, 50, 50, 60],    # line 2, left
                      [60, 11, 100, 21]],  # line 1, right (1 px lower than its neighbour)
                     float)
    assert common.group_lines(boxes) == [[1, 3], [2, 0]]
    assert common.page_text(["d", "a", "c", "b"], boxes) == "a b\nc d"


def test_group_lines_keeps_a_small_word_on_its_line_and_handles_empty():
    boxes = np.array([[0, 0, 40, 20], [50, 12, 55, 18], [0, 30, 40, 50]], float)  # a comma-sized box
    assert common.group_lines(boxes) == [[0, 1], [2]]
    assert common.group_lines(np.zeros((0, 4))) == [] and common.page_text([], np.zeros((0, 4))) == ""


def test_split_segments_at_wide_gaps():
    boxes = np.array([[0, 0, 30, 10], [34, 0, 60, 10], [200, 0, 230, 10]], float)
    assert common.split_segments([0, 1, 2], boxes) == [[0, 1], [2]]
    assert common.union(boxes[:2]) == [0, 0, 60, 10]


def test_group_lines_with_many_small_boxes_between_two_words_of_a_line():
    """Eight dot-sized boxes (punctuation, specks) sit between two words of one line and come between
    them in top-to-bottom order; the second word must still join the first word's line."""
    words = [[0, 100, 40, 110], [300, 101, 340, 111]]
    small = [[60 + 30 * k, 103, 64 + 30 * k, 108] for k in range(8)]
    other = [[0, 130, 40, 140]]  # the next line
    boxes = np.array(words + small + other, float)
    lines = common.group_lines(boxes)
    assert lines == [[0, 2, 3, 4, 5, 6, 7, 8, 9, 1], [10]]


def _group_lines_reference(boxes):
    """The rule of group_lines written plainly, comparing every earlier line (no lookback limit)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    lines, ext = [], []
    for i in np.argsort(cy, kind="stable"):
        h = boxes[i, 3] - boxes[i, 1]
        for k in range(len(lines) - 1, -1, -1):
            top, bot = ext[k]
            if top <= cy[i] <= bot and min(bot, boxes[i, 3]) - max(top, boxes[i, 1]) >= 0.4 * h:
                lines[k].append(int(i))
                b = boxes[lines[k]]
                ext[k] = (float(np.median(b[:, 1])), float(np.median(b[:, 3])))
                break
        else:
            lines.append([int(i)])
            ext.append((float(boxes[i, 1]), float(boxes[i, 3])))
    order = np.argsort([np.median(boxes[ln][:, 1] + boxes[ln][:, 3]) / 2 for ln in lines], kind="stable")
    return [sorted(lines[k], key=lambda j: boxes[j, 0]) for k in order]


def test_group_lines_matches_the_unlimited_reference_on_random_layouts():
    rng = np.random.default_rng(2026)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        h = np.exp(rng.uniform(np.log(2), np.log(200), n))
        cy, x = rng.uniform(0, 400, n), rng.uniform(0, 400, n)
        boxes = np.stack([x, cy - h / 2, x + 20, cy + h / 2], 1)
        assert common.group_lines(boxes) == _group_lines_reference(boxes)


def test_group_lines_column_split_is_opt_in():
    # two columns side by side: a label on the left, a value far to the right
    boxes = np.array([[0, 0, 30, 10], [34, 0, 60, 10], [300, 1, 330, 11], [0, 20, 30, 30]], float)
    assert common.group_lines(boxes) == [[0, 1, 2], [3]]  # default: unchanged reading order
    assert common.group_lines(boxes, split_gap=common.SEGMENT_GAP) == [[0, 1], [2], [3]]
    assert common.group_lines(np.zeros((0, 4)), split_gap=2.0) == []


# ---------------------------------------------------------------- SROIE line -> word boxes

def _line_image(spans: list[tuple[int, int]], h: int = 20, w: int = 200) -> np.ndarray:
    page = np.full((60, w), 255, np.uint8)
    for a, b in spans:
        page[25:25 + h - 8, a:b] = 0  # ink
    return page


def test_split_line_finds_the_widest_gaps():
    page = _line_image([(10, 40), (44, 70), (100, 150)])  # gaps of 4 px and 30 px
    assert split_line(page, [5, 20, 195, 45], 2) == [[10, 25, 70, 37], [100, 25, 150, 37]]
    three = split_line(page, [5, 20, 195, 45], 3)
    assert [b[0] for b in three] == [10, 44, 100] and [b[2] for b in three] == [40, 70, 150]


def test_split_line_gives_up_without_enough_gaps_or_ink():
    page = _line_image([(10, 40), (44, 70)])
    assert split_line(page, [5, 20, 195, 45], 3) is None
    assert split_line(np.full((60, 200), 255, np.uint8), [5, 20, 195, 45], 1) is None
    assert split_line(page, [5, 20, 6, 21], 1) is None  # box too small


def test_split_line_handles_light_text_on_dark():
    page = 255 - _line_image([(10, 40), (100, 150)])
    assert [b[0] for b in split_line(page, [5, 20, 195, 45], 2)] == [10, 100]


# ---------------------------------------------------------------- augmentation and letterbox

def test_perspective_moves_boxes_with_the_image():
    cv2 = pytest.importorskip("cv2")
    from ocr import augment
    img = np.full((100, 200, 3), 255, np.uint8)
    img[40:60, 50:150] = 0
    out, boxes = augment.perspective(img, np.random.default_rng(0), 0.05, np.array([[50, 40, 150, 60]], float))
    ys, xs = np.where(out[..., 0] < 128)
    ink = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
    assert np.abs(boxes[0] - ink).max() <= 2.5  # the box still encloses the dark block


def test_phone_capture_is_seeded_and_keeps_type():
    pytest.importorskip("cv2")
    from ocr import augment
    img = np.random.default_rng(1).integers(0, 255, (48, 120, 3), dtype=np.uint8)
    a, _ = augment.phone_capture(img, np.random.default_rng(5), p=1.0)
    b, _ = augment.phone_capture(img, np.random.default_rng(5), p=1.0)
    assert a.dtype == np.uint8 and a.ndim == 3 and np.array_equal(a, b)
    c, boxes = augment.phone_capture(img, np.random.default_rng(5), np.zeros((0, 4)), p=0.0)
    assert np.array_equal(c, img) and len(boxes) == 0


def test_letterbox_keeps_aspect_and_maps_boxes():
    pytest.importorskip("torch")
    from ocr.train_det import letterbox
    img = np.full((100, 200, 3), 200, np.uint8)
    out, rel = letterbox(img, np.array([[0, 0, 200, 100], [50, 25, 150, 75], [10, 10, 10.5, 10.5]], np.float32), 400)
    assert out.shape == (400, 400, 3) and out[0, 0, 0] == 0 and out[200, 200, 0] == 200  # padded top and bottom
    assert np.allclose(rel[0], [0, 0.25, 1, 0.75]) and np.allclose(rel[1], [0.25, 0.375, 0.75, 0.625])
    assert len(rel) == 2  # the sub-pixel box is dropped


# ---------------------------------------------------------------- scanner stage decision

def test_scanner_crop_is_used_only_when_safe():
    pytest.importorskip("torch")
    from ocr.pipeline import scanner_decision
    quad = [[100, 100], [700, 100], [700, 900], [100, 900]]
    inside = np.array([[200, 200, 300, 220], [400, 500, 500, 520]], float)
    page = {"page_found": True, "confidence": 0.9, "corners": quad}
    assert scanner_decision(page, inside, 1000, 1000)[0] is True
    outside = np.vstack([inside, [[800, 950, 900, 970]]])
    assert scanner_decision(page, outside, 1000, 1000)[0] is False  # the crop would lose a word
    assert scanner_decision(dict(page, page_found=False), inside, 1000, 1000)[0] is False
    assert scanner_decision(dict(page, confidence=0.3), inside, 1000, 1000)[0] is False
    assert scanner_decision(page, np.zeros((0, 4)), 1000, 1000)[0] is False
    full = dict(page, corners=[[5, 5], [995, 5], [995, 995], [5, 995]])
    assert scanner_decision(full, inside, 1000, 1000)[0] is False  # nothing to gain from the warp


def test_scanner_crop_threshold_is_the_scanners_own():
    pytest.importorskip("torch")
    from ocr import pipeline
    from scanner import infer
    assert pipeline.SCAN_MIN_CONF == infer.MIN_CONFIDENCE
    quad = [[100, 100], [700, 100], [700, 900], [100, 900]]
    page = {"page_found": True, "confidence": infer.MIN_CONFIDENCE - 0.01, "corners": quad}
    assert pipeline.scanner_decision(page, np.array([[200, 200, 300, 220]], float), 1000, 1000)[0] is False


@pytest.mark.parametrize("reverse", [False, True], ids=["clockwise", "counter-clockwise"])
def test_inside_quad_is_the_same_for_both_windings(reverse):
    pytest.importorskip("torch")
    from ocr.pipeline import _inside
    quad = np.array([[0, 0], [10, 0], [12, 10], [-2, 10]], float)
    quad = quad[::-1] if reverse else quad
    on_edge = np.array([[5, 0], [11, 5], [5, 10], [-1, 5], [0, 0], [12, 10]], float)  # sides and corners
    assert _inside(on_edge, quad).all()
    assert _inside(np.array([[5, 5], [0.1, 0.1]], float), quad).all()
    assert not _inside(np.array([[5, -0.01], [12.6, 5], [5, 10.01], [-20, 5]], float), quad).any()


def test_recogniser_loads_offline_from_its_checkpoint(monkeypatch, tmp_path):
    """With a checkpoint file, nothing is downloaded: the docTR cache is pointed at an empty folder
    and any download attempt fails the test."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("doctr")
    from ocr import models
    import os
    if not os.path.exists(models.REC_CKPT):
        pytest.skip("models/ocr recogniser checkpoint not present")
    import importlib
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path))

    def no_download(*a, **k):
        raise AssertionError("tried to download pretrained weights")
    for name in ("doctr.models.utils.pytorch", "doctr.utils.data"):
        monkeypatch.setattr(importlib.import_module(name), "download_from_url", no_download)
    model = models.load_rec(ckpt=models.REC_CKPT)
    ck = torch.load(models.REC_CKPT, map_location="cpu")["model"]
    sd = model.state_dict()
    assert list(sd) == list(ck) and all(torch.equal(sd[k], ck[k]) for k in sd)


def test_sroie_is_scored_without_case():
    pytest.importorskip("rapidfuzz")
    assert common.error_counts("Total", "TOTAL", "sroie")["char_err"] == 0
    assert common.error_counts("Total", "TOTAL", "funsd")["char_err"] == 4


def test_transparent_background_becomes_white():
    pytest.importorskip("torch")  # ocr.pipeline imports ocr.models, which needs torch
    from PIL import Image
    from ocr.pipeline import to_rgb
    im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_rgb(im).getpixel((0, 0)) == (255, 255, 255)


# ---------------------------------------------------------------- numbers and units (safety flag)

@pytest.mark.parametrize("text", ["Ramipril 2.5mg once a day", "take two tabs nocte", "Hb 13.5 g/dL",
                                  "sats 97%", "5 mmol", "one puff prn", "Digoxin 62.5 mcg"])
def test_lines_with_numbers_or_units_are_found(text):
    assert common.has_number_or_unit(text)


@pytest.mark.parametrize("text", ["", "Review in clinic", "Dr Meera Lang", "no known allergies", "morning walk"])
def test_plain_text_lines_are_not_flagged(text):
    assert not common.has_number_or_unit(text)


# Names, initials and headings that hold a unit-like word but no number (invented).
COLLISIONS = ["Mehta OD Clinic, Station Road", "Dr L Example", "G Sample, Pharmacist", "Department of Mol Biology",
              "Tab Pharmacy, High Street", "Caps and gowns must be worn", "IU Health partner clinic",
              "Prn as directed by the doctor", "BD Diagnostics Ltd", "Mane Street Surgery", "Name: L Sample",
              "ML Sample", "Dl Sample", "Initials: A G", "TDS Medical Group", "Test Result Units Reference range"]


@pytest.mark.parametrize("text", COLLISIONS)
def test_unit_like_words_without_a_number_are_not_flagged(text):
    assert not common.has_number_or_unit(text)
    assert not common.number_lost(text) and not common.has_misread_number(text)


@pytest.mark.parametrize("text", [
    "1 tab od", "l tab od",  # a unit after a number, or after another unit that counted
    "InSulin O Units", "Insulin glargine IZ Units", "lnsulin lO Units",  # digits read as look-alike letters
    "Morphine S ML", "5 ML", "ten units", "Omeprazole mg", "Dose in mL", "Hb in g/dL", "Na in mmol/L",
])
def test_units_next_to_a_number_or_in_their_usual_case_are_flagged(text):
    assert common.has_number_or_unit(text)


def test_a_unit_without_its_number_is_a_lost_number():
    assert common.number_lost("Omeprazole mg") and common.number_lost("InSulin O Units")
    assert not common.number_lost("Omeprazole 20 mg") and not common.number_lost("Review in clinic")


@pytest.mark.parametrize("text,flag", [
    ("Metformin SOO my twice daily", True), ("Atorvastatin ZO mg at night", True), ("Ramipril Z.S my", True),
    ("Metformin 500 mg", False),  # real digits: not a misreading
    ("So my doctor said to continue", False), ("Mehta OD Clinic", False), ("Dr L Example", False),
])
def test_digits_misread_as_letters_before_a_unit(text, flag):
    assert common.has_misread_number(text) is flag


def test_limited_evaluation_preserves_full_results(tmp_path, monkeypatch):
    import json
    pytest.importorskip("cv2")
    from ocr import evaluate, evaluate_v2

    monkeypatch.setattr(evaluate_v2, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(evaluate_v2, "REPORTS", str(tmp_path / "reports"))
    for limit, page in ((None, "full"), (1, "sample")):
        evaluate_v2._save("gnhk", "test", {"page": page}, {"pages": [{"id": page}]}, limit)
    for suffix, page in (("", "full"), ("_first1", "sample")):
        assert json.loads((tmp_path / "reports" / f"gnhk{suffix}.json").read_text())["page"] == page
        assert json.loads((tmp_path / "runs" / "gnhk" / f"pages_test{suffix}.jsonl").read_text())["id"] == page
    assert evaluate._run_name("finetuned", "val", 1) != evaluate._run_name("finetuned", "val", None)


def test_evaluation_counts_handwriting_attempts_and_selections():
    pytest.importorskip("cv2")
    from ocr.evaluate_v2 import _scripts

    counts = _scripts({"lines": [
        {"script": "printed", "low_confidence": False},
        {"script": "printed", "handwriting_text": "", "low_confidence": True},
        {"script": "handwritten", "handwriting_text": "read", "low_confidence": False},
    ]})
    assert counts["lines_sent_to_handwriting"] == 2
    assert counts["handwritten_lines"] == 1


def test_numeric_line_accuracy_uses_reference_to_include_lost_doses():
    assert common.numeric_line_accuracy([
        ("Insulin O Units", "Insulin 10 units"), ("25 mg", "2.5 mg"),
        ("0.5   mL", "0.5 mL"), ("10", "no number"),
    ]) == {"exact": 1, "total": 3, "accuracy": 0.3333}
    assert common.numeric_line_accuracy([])["accuracy"] is None
