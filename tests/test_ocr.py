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


# Each unit needs its own quantity right before it (invented lines). A list number, a duration or another
# medicine's dose on the same line does not stand in for it.
LOCAL_DOSES = [
    ("Omeprazole mg", True), ("1. Omeprazole mg", True), ("Omeprazole mg for 7 days", True),
    ("Omeprazole mg x 30 days", True), ("Amlodipine 5 mg and Omeprazole mg", True),
    ("Omeprazole mg, Amlodipine 5 mg", True), ("2) Omeprazole mg", True), ("InSulin O Units", True),
    ("Insulin lO Units at 8 pm", True),  # look-alike letters are no quantity, whatever else is on the line
    ("1. Omeprazole 20 mg", False), ("Omeprazole 20 mg for 7 days", False), ("Ramipril 2.5 mg", False),
    ("Ramipril .5 mg", False), ("Ramipril 0.5 mg", False), ("Ramipril 2.5mg once a day", False),
    ("Paracetamol 1-2 tabs", False), ("1 tab od", False), ("Sodium 139 mmol/L", False),
    ("Haemoglobin 13.2 g/dL 12.0-15.5", False), ("1. Tab Amlodipine 5 mg OD x 30 days", False),
    ("Syrup 125 mg/5 ml", False), ("Paracetamol 500 mg as needed prn", False), ("BP 120/80 mmHg", False),
]


@pytest.mark.parametrize("text,lost", LOCAL_DOSES)
def test_each_unit_needs_its_own_quantity(text, lost):
    assert common.has_number_or_unit(text)
    assert common.number_lost(text) is lost


@pytest.mark.parametrize("text,lost", LOCAL_DOSES)
def test_printed_line_warning_follows_its_own_quantity(text, lost):
    """A v2 printed-only line: the prominent confirm_numbers warning for a unit without its number, the
    quieter has_numbers note otherwise; the same through the API's _line, which keeps the text as read."""
    from ocr.api import _line
    ln = {"text": text, "box": [0, 0, 10, 10], "confidence": 0.95, "script": "printed"}
    ln["confirm_numbers"] = common.line_confirm_numbers(ln, script_known=True)
    ln["has_numbers"] = common.has_number_or_unit(text)
    out = _line(ln)
    assert out["confirm_numbers"] is lost and out["has_numbers"] is True and out["text"] == text


# Correct print that holds a count, a frequency or a form but no dose: no "number lost" warning. Doses that
# are gone keep it (invented lines).
NO_LOST_NUMBER = ["Take one tablet at night", "Take two tablets twice daily", "Half a tablet in the morning",
                  "One puff when needed", "Two drops in each eye tds", "Eye drops nocte", "Apply the cream bd",
                  "Salbutamol inhaler prn", "Paracetamol tablets as needed", "Bo Tab and Sons Chemists",
                  "Paracetamol 500 mg two tablets", "tab Metformin 500 mg", "cap Omeprazole 20 mg",
                  "1. tab Ramipril 2.5 mg", "tabs Paracetamol 1 g prn", "Glucose (mmol/L) 5.4", "Glucose mmol/L 5.4",
                  "Haemoglobin (g/dL) 13.2", "Potassium mmol/L 4.2 3.5-5.0", "Test (units) Result Reference range",
                  "Test Units Result"]
STILL_LOST = ["Omeprazole mg", "1. Omeprazole mg", "Amlodipine 5 mg and Omeprazole mg", "Omeprazole mg, Amlodipine 5 mg",
              "Omeprazole mg for 7 days", "Omeprazole mg 7 days", "Omeprazole mg 1 tab od", "Omeprazole mg x 14 days",
              "InSulin O Units", "Take tab nocte", "Amitriptyline nocte", "Bisoprolol od", "Paracetamol prn",
              "Paracetamol g prn", "Amitriptyline mg nocte", "tab Metformin mg", "Glucose (mmol/L)",
              "Glucose mmol/L S.4", "Haemoglobin (g/dL) l3.2", "Ferritin (ug/L) 4S", "Haemoglobin 11.8 q/dL"]


@pytest.mark.parametrize("text", NO_LOST_NUMBER)
def test_counts_frequencies_forms_and_lab_columns_are_not_lost_numbers(text):
    assert not common.number_lost(text), common.units_without_quantity(text)
    ln = {"text": text, "script": "printed"}
    assert common.line_confirm_numbers(ln, script_known=True) is False
    assert common.line_confirm_numbers(ln, script_known=True, flag_misread_digits=True) is False


@pytest.mark.parametrize("text", STILL_LOST)
def test_doses_that_are_gone_are_still_lost_numbers(text):
    assert common.number_lost(text), text
    assert common.line_confirm_numbers({"text": text, "script": "printed"}, script_known=True) is True


@pytest.mark.parametrize("text,flag", [
    ("Levothyroxine 50 mcq OD", True), ("Lactulose 10 mI bd", True), ("Haemoglobin 11.8 q/dL", True),
    ("Metformin 500 my bd", True), ("Atorvastatin 20 rng", True),
    ("Levothyroxine 50 mcg OD", False), ("Lactulose 10 mL bd", False), ("Haemoglobin 11.8 g/dL", False),
    ("Since 2019 my blood pressure is fine", False), ("In 2024 my dose was reduced", False),
])
def test_misread_units_after_digits_only_with_the_option(text, flag):
    assert common.has_misread_unit(text) is flag
    ln = {"text": text, "script": "printed"}
    assert common.line_confirm_numbers(ln, script_known=True, flag_misread_digits=True) is (flag or common.number_lost(text))
    if text != "Haemoglobin 11.8 q/dL":  # that one also lost the unit of its value ("dL" without a number)
        assert common.line_confirm_numbers(ln, script_known=True) is False  # off by default


@pytest.mark.parametrize("text,flag", [
    ("Metformin SOO my", True), ("Warfarin B my", True), ("Levothyroxine SO mcq", True), ("Ramipril Z.S my", True),
    ("So my doctor said to continue", False), ("Ask Bo my pharmacist", False), ("Bo Tab and Sons Chemists", False),
    ("OBS stable overnight", False), ("ZOO Road Pharmacy", False), ("Is my dose right", False), ("Boss my manager", False),
])
def test_capitalised_names_are_not_misread_numbers(text, flag):
    assert common.has_misread_number(text) is flag


@pytest.mark.parametrize("text", ["Insulin Il units", "Insulin Sl units", "Insulin Ill units", "Metformin Bl mg"])
def test_capitalised_look_alikes_without_a_vowel_are_misread_numbers(text):
    """"Il" (11), "Sl" (51), "Bl" (81): name-shaped but with no vowel, so still digits read as letters."""
    assert common.has_number_or_unit(text) and common.number_lost(text)
    assert common.line_confirm_numbers({"text": text, "script": "printed"}, script_known=True) is True
    routed = {"text": text, "script": "handwritten", "handwriting_text": text, "printed_text": text}
    assert common.line_confirm_numbers(routed, script_known=True) is True


@pytest.mark.parametrize("text", ["Insulin Io units", "Insulin Zo units", "Insulin So units", "Aspirin Bo mg",
                                  "Insulin Io units at night"])
def test_capitalised_look_alikes_with_o_before_a_unit_are_misread_numbers(text):
    """"o" is the look-alike of 0, so "Io" (10), "Zo" (20), "So" (50) right before a unit are numbers, on
    printed, handwriting-routed and (see test_ocr_api) v1 lines."""
    assert common.has_number_or_unit(text) and common.number_lost(text)
    assert common.has_misread_number(text)
    assert common.line_confirm_numbers({"text": text, "script": "printed"}, script_known=True) is True
    routed = {"text": text, "script": "handwritten", "handwriting_text": text, "printed_text": text}
    assert common.line_confirm_numbers(routed, script_known=True) is True


@pytest.mark.parametrize("text", ["Bo Tab and Sons Chemists", "So my doctor said to continue", "Ask Bo my pharmacist",
                                  "Timolol OS drops", "Latanoprost OS nocte", "Eye drops nocte", "So tablets were stopped"])
def test_capitalised_words_before_a_name_or_word_stay_clean(text):
    for misread in (False, True):
        assert common.line_confirm_numbers({"text": text, "script": "printed"}, script_known=True,
                                           flag_misread_digits=misread) is False


@pytest.mark.parametrize("text,lost", [
    ("Salbutamol puffs prn", True), ("Salbutamol tabs bd", True),  # the count went missing
    ("Paracetamol tablets as needed", False), ("Eye drops nocte", False), ("Ear drops bd", False),
    ("Analyte (mmol/L)", False), ("Test Units Result", False),  # column headers
    ("Glucose (mmol/L)", True), ("Potassium mmol/L", True),  # a test whose value went missing
])
def test_plural_counts_and_unit_headers(text, lost):
    assert common.number_lost(text) is lost
    assert common.line_confirm_numbers({"text": text, "script": "printed"}, script_known=True) is lost


@pytest.mark.parametrize("text,flag", [
    ("Timolol OS drops", False), ("Timolol OS drops bd", False), ("Apply to OS", False),  # OS: left eye
    ("Levothyroxine SO mcq", True), ("Warfarin SO mg", True),  # capitals before a dose unit: 50
])
def test_look_alike_words_are_words_except_in_capitals_before_a_dose_unit(text, flag):
    assert common.has_misread_number(text) is flag
    ln = {"text": text, "script": "printed"}
    assert common.line_confirm_numbers(ln, script_known=True, flag_misread_digits=True) is flag


M = "Metformin SOO my"
ROUTINGS = {  # (a) printed only, (b) handwriting reading shown, (c) printed reading kept after handwriting
    "a": [{"text": M, "script": "printed"}],
    "b": [{"text": M, "script": "handwritten", "handwriting_text": M, "printed_text": "Metformin twice daily"},
          {"text": "Metformin twice daily", "script": "handwritten", "handwriting_text": "Metformin twice daily",
           "printed_text": M}],
    "c": [{"text": M, "script": "printed", "handwriting_text": "Metformin twice daily"},
          {"text": "Metformin twice daily", "script": "printed", "handwriting_text": M}],
}


@pytest.mark.parametrize("routing,k", [(r, k) for r, lines in ROUTINGS.items() for k in range(len(lines))])
def test_misread_digits_are_looked_for_in_every_reading_when_asked(routing, k):
    ln = ROUTINGS[routing][k]
    assert common.line_confirm_numbers(ln, script_known=True, flag_misread_digits=True) is True
    assert common.line_confirm_numbers(ln, script_known=True) is False  # off: as before


@pytest.mark.parametrize("ln,flag", [
    ({"text": "InSulin O Units", "script": "printed"}, True),
    ({"text": "Ramipril Z.S my", "script": "handwritten", "handwriting_text": "Ramipril Z.S my",
      "printed_text": "Ramipril 2.5 mg"}, True),  # the dose is only in the other reading
    ({"text": "Dr L Example", "script": "printed"}, False), ({"text": "Mehta OD Clinic", "script": "printed"}, False),
    ({"text": "Test Result Units Reference range", "script": "printed"}, False),
])
def test_confirm_numbers_coverage(ln, flag):
    for misread in (False, True):
        assert common.line_confirm_numbers(ln, script_known=True, flag_misread_digits=misread) is flag
    if not flag:
        assert not common.has_number_or_unit(ln["text"])


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
