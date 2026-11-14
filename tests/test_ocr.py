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
