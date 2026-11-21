"""Benchmark runners (benchmarks/): fixture schema, determinism, and the pieces that need no model.

Heavy parts are guarded: the warning runner imports ocr.scan (torch, docTR); the rest needs only
requirements.txt. Nothing here reads the stored synthetic photos (they are not in git)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from benchmarks import common as C
from benchmarks.fields import numbers, parse_fields
from benchmarks.warnings import build_fixtures as BF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES = os.path.join(ROOT, "benchmarks", "warnings", "fixtures.jsonl")
ROUTINGS = {"printed_only", "handwriting_selected", "printed_selected_after_handwriting", "none_v1"}


@pytest.fixture(scope="module")
def recs():
    with open(FIXTURES) as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------------ fixtures
def test_fixture_file_is_what_the_builder_writes(recs):
    with open(FIXTURES) as f:
        assert f.read() == BF.dumps(BF.build())


def test_fixture_schema(recs):
    assert len(recs) >= 150
    assert len({r["id"] for r in recs}) == len(recs)
    for r in recs:
        for k in ("id", "set", "category", "path", "routing", "source_script", "ground_truth", "displayed_text",
                  "has_numeric_content", "dose_incorrect_or_lost", "warning_warranted", "expected_fields"):
            assert k in r, (r["id"], k)
        assert r["routing"] in ROUTINGS
        assert r["path"] == ("v1" if r["routing"] == "none_v1" else "v2")
        assert r["set"] in ("original_39", "extended", "extended_r2", "extended_r3", "extended_r4")
        assert r["expected_fields"] == parse_fields(r["ground_truth"])
        for f in r["expected_fields"]:
            assert set(f) == {"drug", "value", "unit"}
        if r["routing"] == "handwriting_selected":
            assert r["handwriting_text"] == r["displayed_text"] and "printed_text" in r
        if r["routing"] == "printed_selected_after_handwriting":
            assert "handwriting_text" in r and "printed_text" not in r
        if r["routing"] in ("printed_only", "none_v1"):
            assert "handwriting_text" not in r and "printed_text" not in r
        if r["dose_incorrect_or_lost"]:
            assert r["warning_warranted"]
        assert r["warning_warranted"] == (r["dose_incorrect_or_lost"]
                                          or (r["source_script"] == "handwritten" and r["has_numeric_content"]))


def test_original_39_is_the_earlier_set(recs):
    o = [r for r in recs if r["set"] == "original_39"]
    assert len(o) == 39 and all(r["has_numeric_content"] for r in o)
    assert [r["id"] for r in o] == [f"o39-{i:02d}" for i in range(1, 40)]
    assert sum(r["path"] == "v1" for r in o) == 3
    assert o[15]["displayed_text"] == "Metformin SOO my twice daily" and o[38]["displayed_text"] == "Metformin SOO my"


def test_required_cases_are_present(recs):
    shown = {(r["routing"], r["displayed_text"]) for r in recs}
    texts = {r["displayed_text"] for r in recs}
    for t in ("Omeprazole mg", "1. Omeprazole mg", "Omeprazole mg for 7 days", "1. Omeprazole 20 mg",
              "Amlodipine 5 mg and Omeprazole mg", "InSulin O Units", "Dr L Example", "Mehta OD Clinic",
              "Test Result Units Reference range", "Ramipril 25 mg OD", "Lorazepam .5 mg", "Lorazepam 0.5 mg"):
        assert t in texts, t
    soo = [r for r in recs if "Metformin SOO my" in (r["displayed_text"], r.get("handwriting_text"), r.get("printed_text"))]
    for routing in ("printed_only", "handwriting_selected", "printed_selected_after_handwriting", "none_v1"):
        assert any(r["routing"] == routing for r in soo), routing
    for routing in ("handwriting_selected", "printed_selected_after_handwriting"):  # misread shown, and misread alternate
        assert (routing, "Metformin SOO my") in shown
        assert any(r["routing"] == routing and r["displayed_text"] == "Metformin 500 mg" for r in soo)
    assert any(r["category"] == "dose_only_in_alternate" for r in recs)
    for unit in ("mcg", "mL", "IU", "mmol/L", "g/dL"):
        assert any(unit in r["ground_truth"] for r in recs), unit
    for freq in (" od", " bd", " tds", " prn", " nocte", "twice daily"):
        assert any(freq in r["ground_truth"] for r in recs), freq


# ------------------------------------------------------------------ field parser
@pytest.mark.parametrize("text, fields", [
    ("1. Tab Amlodipine 5 mg OD and Omeprazole mg",
     [{"drug": "amlodipine", "value": "5", "unit": "mg"}, {"drug": "omeprazole", "value": None, "unit": "mg"}]),
    ("Omeprazole mg for 7 days", [{"drug": "omeprazole", "value": None, "unit": "mg"}]),
    ("Haemoglobin 13.2 g/dL 12.0-15.5", [{"drug": "haemoglobin", "value": "13.2", "unit": "g/dl"}]),
    ("Metformin SOO my", [{"drug": "metformin", "value": None, "unit": None}]),
    ("Metformin 500mg bd", [{"drug": "metformin", "value": "500", "unit": "mg"}]),
    ("Insulin glargine 12 units", [{"drug": "insulin glargine", "value": "12", "unit": "units"}]),
    ("Increase Amlodipine to 10 mg", [{"drug": "amlodipine", "value": "10", "unit": "mg"}]),
    ("Dr L Example", []),
])
def test_parse_fields(text, fields):
    assert parse_fields(text) == fields


def test_numbers():
    assert numbers("Review on 2l/04") == ["2", "04"]
    assert numbers("Lorazepam .5 mg") == [".5"]


# ------------------------------------------------------------------ common helpers
def test_results_are_deterministic_apart_from_volatile_fields(tmp_path):
    res = {"b": [1, 2.0000001], "a": {"x_ms": 3.0, "latency": {"m": 1}, "y": 1}}
    p1 = C.write_result(str(tmp_path / "1"), "r", res)
    p2 = C.write_result(str(tmp_path / "2"), "r", {"a": {"y": 1, "x_ms": 9.0}, "b": [1, 2.0]})
    a, b = json.load(open(p1)), json.load(open(p2))
    assert "generated_at" in a
    assert C.strip_volatile(a) == C.strip_volatile(b)
    assert C.out_name("pages", 5) == "pages_first5" and C.out_name("pages", None) == "pages"
    assert C.frac(37, 39) == {"num": 37, "den": 39, "rate": 0.9487}


# ------------------------------------------------------------------ warning runner (needs torch: imports ocr.scan)
def test_warning_runner_is_deterministic_and_scores_the_target():
    pytest.importorskip("torch")
    pytest.importorskip("doctr")
    from benchmarks import run_warnings as RW
    a = RW.run(ROOT, limit=60)
    b = RW.run(ROOT, limit=60)
    assert C.strip_volatile(C._clean(a)) == C.strip_volatile(C._clean(b))
    t = a["modes"]["misread_option_off (default)"]["target_original_39"]
    assert t["recall"]["den"] == 39 and t["threshold"] == ">= 0.95"
    assert t["result"] == ("PASS" if t["recall"]["num"] >= 0.95 * 39 else "FAIL")


def test_target_is_not_rounded():
    from benchmarks import run_warnings as RW
    recs = {f"o{i}": {"has_numeric_content": True} for i in range(39)}
    rows = [{"id": f"o{i}", "set": "original_39", "any_flag": i >= 2, "displayed_text": ""} for i in range(39)]
    assert RW.target(rows, recs)["result"] == "FAIL"  # 37/39 = 0.9487
    rows[0]["any_flag"] = True
    assert RW.target(rows, recs)["result"] == "PASS"  # 38/39


# ------------------------------------------------------------------ handwriting runner pieces (no model)
def test_match_lines_with_a_fake_detector():
    from benchmarks import run_handwriting as RH
    from ocr.common import group_lines, union
    true = [[0, 0, 400, 40], [0, 60, 400, 100], [0, 120, 400, 160], [0, 180, 400, 220], [0, 240, 400, 280],
            [0, 300, 400, 340]]
    # line 0 found; line 1 split in two; lines 2 and 3 merged; line 4 missed; line 5 found; one extra line
    pred = [[0, 300, 400, 340], [0, 0, 400, 40], [0, 60, 190, 100], [210, 60, 400, 100], [0, 120, 400, 220],
            [600, 600, 700, 640]]
    m = RH.match_lines(pred, true)
    assert m == {"true": 6, "pred": 6, "matched": 2, "split": 1, "merged": 1, "missed": 1, "extra": 1,
                 "misordered": 1, "inversions": 1}
    def fake_detect(img):  # word boxes of two lines, words out of order
        return np.array([[220, 60, 400, 100], [0, 0, 180, 40], [0, 60, 200, 100], [200, 0, 400, 40]], float)
    lines = RH.detected_lines(fake_detect, group_lines, union, None)
    assert RH.match_lines(lines, true[:2])["matched"] == 2


def test_handwriting_pages_are_deterministic_and_annotated():
    from benchmarks import run_handwriting as RH
    fonts = RH.font_files(ROOT)
    if not fonts:
        pytest.skip("no handwriting fonts in data/raw/fonts")
    a, b = RH.render_page(3, fonts), RH.render_page(3, fonts)
    assert np.array_equal(a["image"], b["image"]) and [x["text"] for x in a["lines"]] == [x["text"] for x in b["lines"]]
    for ln in a["lines"]:
        assert ln["quad"].shape == (4, 2) and " ".join(w["text"] for w in ln["words"]) == ln["text"]
    assert set(RH.VAL_SEEDS).isdisjoint(RH.TEST_SEEDS)


def test_handwriting_metrics():
    from benchmarks import run_handwriting as RH
    assert RH.cer_wer([("Ramipril 25 mg", "Ramipril 2.5 mg")])["char_errors"] == 1
    assert RH.wrong_decimal("Ramipril 25 mg", "Ramipril 2.5 mg") and not RH.wrong_decimal("Ramipril 2.5 mg", "Ramipril 2.5 mg")
    rows = [{"truth": "Ramipril 2.5 mg", "handwriting": "Ramipril 25 mg", "handwriting_conf": 0.97},
            {"truth": "See me", "handwriting": "See me", "handwriting_conf": 0.55}]
    cal = RH.calibration(rows, "handwriting", "handwriting_conf")
    assert cal["by_threshold"]["0.95"]["wrong_decimals_among_accepted"] == {"num": 1, "den": 1, "rate": 1.0}
    assert cal["by_threshold"]["0.50"]["accepted"]["num"] == 2


# ------------------------------------------------------------------ page and alpha runner pieces (base environment)
def test_page_generators_are_deterministic_and_transforms_map_the_quad():
    from benchmarks import run_pages as RP
    from scanner import synth
    a, b = RP.compose(synth, 1, "cut2"), RP.compose(synth, 1, "cut2")
    assert np.array_equal(np.asarray(a["image"]), np.asarray(b["image"])) and np.allclose(a["quad"], b["quad"])
    assert a["corners_outside"] == 2
    assert RP.compose(synth, 0, "cut1")["corners_outside"] == 1
    ex = RP.compose(synth, 0, "regular")
    W, H = ex["image"].size
    ref = (np.asarray(ex["image"].convert("L"), float), synth.poly_mask(ex["quad"], W, H) >= 0.5)
    for name, img, q, tm in RP.transforms(ex):
        w, h = img.size
        assert tm.shape == (h, w), name
        m = synth.poly_mask(q, w, h) >= 0.5
        assert abs(int(m.sum()) - int(ref[1].sum())) <= 0.03 * ref[1].sum() or name == "cropped", name
        assert (tm & ~m).sum() <= 0.01 * max(tm.sum(), 1), name  # the text stays inside the mapped page
    kinds = {k for _, k, _, _ in RP.negatives(synth, 1)}
    for k in RP.TRAIN_KINDS:
        assert f"train_{k}" in kinds
    for k in RP.OOD_KINDS:
        assert f"ood_{k}" in kinds


def test_alpha_pairs_differ_by_one_pixel():
    from benchmarks import run_alpha as RA
    a, m = RA.text_layer()
    b = RA.add_pixel(a)
    assert (a != b).any(axis=2).sum() == 1 and m.any()
    assert not (a[..., 3] == 255).any() and (b[..., 3] == 255).sum() == 1


def test_round2_set_is_appended_and_covers_each_kind(recs):
    sets = [r["set"] for r in recs]
    first_r2, first_r3 = sets.index("extended_r2"), sets.index("extended_r3")
    assert set(sets[first_r2:first_r3]) == {"extended_r2"} and first_r3 - first_r2 == 76  # appended after round 1
    first_r4 = sets.index("extended_r4")
    assert set(sets[first_r3:first_r4]) == {"extended_r3"} and first_r4 - first_r3 == 40  # round 3 after round 2
    assert set(sets[first_r4:]) == {"extended_r4"}  # round 4 after round 3
    assert first_r2 == 295  # the round-1 records stay as they were
    r2 = [r for r in recs if r["set"] == "extended_r2"]
    for cat in ("r2_count_freq_no_digit", "r2_unit_before_value", "r2_form_word", "r2_v1_misread", "r2_misread_unit"):
        sub = [r for r in r2 if r["category"] == cat]
        assert len(sub) >= 8, cat
        assert any(r["dose_incorrect_or_lost"] for r in sub) and any(not r["warning_warranted"] for r in sub), cat
        assert {r["path"] for r in sub} == {"v1", "v2"} or cat == "r2_v1_misread", cat


def test_hw_options_are_routed_or_ignored():
    from benchmarks import run_handwriting as RH

    class Reader:
        def __init__(self, path_or_name=None, local_only=False):
            pass

        def read(self, lines, batch_size=8, num_beams=1):
            pass
    o = RH.split_hw_options(["num_beams=4", "local_only=true", "hw_beams=4", "x=abc"], Reader)
    assert o["read"] == {"num_beams": 4} and o["constructor"] == {"local_only": True}
    assert o["ignored"] == {"hw_beams": 4, "x": "abc"}
    assert RH.split_hw_options(None, Reader) == {"requested": {}, "constructor": {}, "read": {}, "ignored": {}}


def test_round3_set_covers_the_review_cases(recs):
    r3 = [r for r in recs if r["set"] == "extended_r3"]
    shown = {(r["path"], r["displayed_text"]) for r in r3}
    for path in ("v1", "v2"):
        for t in ("Insulin Il units", "Insulin Sl units", "Timolol OS drops", "Ear drops bd", "Salbutamol puffs prn",
                  "Analyte (mmol/L)", "Glucose (mmol/L)", "Bo Tab and Sons Chemists", "Ask Bo my pharmacist",
                  "So my doctor said to continue"):
            assert (path, t) in shown, (path, t)
    assert any(r["routing"] == "handwriting_selected" and r["displayed_text"] == "Insulin Il units" for r in r3)
    assert sum(r["dose_incorrect_or_lost"] for r in r3) >= 15 and sum(not r["warning_warranted"] for r in r3) >= 15


def test_round4_set_covers_capitalised_o_look_alikes(recs):
    r4 = [r for r in recs if r["set"] == "extended_r4"]
    shown = {(r["routing"], r["displayed_text"]) for r in r4}
    for t in ("Insulin Io units", "Insulin Zo units", "Insulin So units", "Insulin Io units at night"):
        for routing in ("printed_only", "none_v1"):
            assert (routing, t) in shown, (routing, t)
        assert any(r["displayed_text"] == t and r["path"] == "v2" and "handwriting_text" in r for r in r4), t
    for t in ("So tablets were stopped", "So my doctor said to continue", "Io Health Clinic", "Zo Pharmacy"):
        assert {("printed_only", t), ("none_v1", t)} <= shown, t
    assert all(r["dose_incorrect_or_lost"] == (r["category"] == "r4_capital_o") for r in r4)
