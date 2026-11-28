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
        assert r["set"] in ("original_39", "extended", "extended_r2", "extended_r3", "extended_r4", "extended_r5")
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


@pytest.mark.parametrize("text, value, unit", [
    ("Metformin ,5 mg", ",5", "mg"),    # a leading decimal comma is part of the number, not punctuation
    ("Metformin 0,5 mg", "0,5", "mg"),
    ("Metformin .5 mg", ".5", "mg"),
    ("Metformin 0.5mg", "0.5", "mg"),   # glued to its unit
    ("Metformin ,5mg", ",5", "mg"),
    ("Metformin 500 mg, then stop", "500", "mg"),  # a word-final comma is prose punctuation
    ("Metformin 500, twice daily", "500", None),
])
def test_parse_fields_keeps_decimal_spelling(text, value, unit):
    assert parse_fields(text) == [{"drug": "metformin", "value": value, "unit": unit}]


def test_a_dropped_leading_decimal_comma_is_a_field_error():
    from benchmarks.fields import field_match_scores
    s = field_match_scores([("Metformin 5 mg", "Metformin ,5 mg")])
    assert s["exact"] == {"num": 0, "den": 1, "rate": 0.0} and s["counts"]["substitutions"] == 1
    assert s["value"]["num"] == 0


def test_decimal_definition():
    from benchmarks.fields import has_decimal, is_decimal
    assert all(is_decimal(n) for n in (".5", ",5", "0,5", "2.5", "0,125", "13.2"))
    assert not any(is_decimal(n) for n in ("5", "500", "1,000", "12,500", ""))
    assert has_decimal("Metformin ,5 mg") and has_decimal("Lorazepam 0,5 mg")
    assert not has_decimal("Review 12/04/2026") and not has_decimal("BP 139/96")
    assert not has_decimal("Metformin 500 mg, then stop")
    assert numbers("Review 12/04/2026") == ["12", "04", "2026"]


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
    assert set(a["modes"]) == {"shipped_default", "explicit_off", "explicit_on"}
    assert a["modes"]["explicit_off"]["misread_check"]["v2"] == "off"
    t = a["modes"]["shipped_default"]["target_original_39"]
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
    # decimal commas count as decimals too, with the same exact spelling required
    assert RH.wrong_decimal("Lorazepam 5 mg", "Lorazepam 0,5 mg") and RH.wrong_decimal("Lorazepam 5 mg", "Lorazepam ,5 mg")
    assert RH.wrong_decimal("Lorazepam 0.5 mg", "Lorazepam 0,5 mg")
    assert not RH.wrong_decimal("Review 12/04/2026", "Review 12/04/2026") and not RH.wrong_decimal("BP 139/9", "BP 139/96")
    rows = [{"truth": "Lorazepam 0,5 mg", "handwriting": "Lorazepam 5 mg", "handwriting_conf": 0.97}]
    cal = RH.calibration(rows, "handwriting", "handwriting_conf")
    assert cal["by_threshold"]["0.95"]["wrong_decimals_among_accepted"] == {"num": 1, "den": 1, "rate": 1.0}


def _counts(pred, ref):
    from benchmarks import run_handwriting as RH
    s = RH.field_scores([(pred, ref)])
    return s, s["counts"]


def test_field_occurrences_are_matched_at_most_once():
    # a. a repeated reference name against one prediction: one exact, one omission
    s, c = _counts("Metformin 500 mg", "Metformin 500 mg and Metformin 500 mg")
    assert (c["exact"], c["omissions"], c["extras"], c["substitutions"]) == (1, 1, 0, 0)
    assert s["precision"] == {"num": 1, "den": 1, "rate": 1.0} and s["recall"] == {"num": 1, "den": 2, "rate": 0.5}
    assert s["exact"] == s["recall"]
    # b. reversed: one exact match, one extra, no omission
    s, c = _counts("Metformin 500 mg and Metformin 500 mg", "Metformin 500 mg")
    assert (c["exact"], c["omissions"], c["extras"]) == (1, 0, 1)
    assert s["precision"] == {"num": 1, "den": 2, "rate": 0.5} and s["recall"] == {"num": 1, "den": 1, "rate": 1.0}


def test_same_name_with_different_values():
    # c. one value changed: the other occurrence is still exact, the changed one is a substitution
    s, c = _counts("Insulin 10 units then Insulin 12 units", "Insulin 10 units then Insulin 20 units")
    assert (c["exact"], c["substitutions"], c["omissions"], c["extras"]) == (1, 1, 0, 0)
    assert s["value"]["num"] == 1 and s["drug"]["num"] == 2
    # swapping the two occurrences: both exact without order, one in order
    s, c = _counts("Insulin 20 units then Insulin 10 units", "Insulin 10 units then Insulin 20 units")
    assert c["exact"] == 2 and s["exact_in_order"] == {"num": 1, "den": 2, "rate": 0.5}
    s, _ = _counts("Insulin 10 units then Insulin 20 units", "Insulin 10 units then Insulin 20 units")
    assert s["exact_in_order"]["num"] == 2


def test_field_scores_with_empty_text_and_missing_units():
    # d. empty output: every reference field is an omission; precision undefined (rate None), recall 0
    s, c = _counts("", "Metformin 500 mg")
    assert c["omissions"] == 1 and s["precision"] == {"num": 0, "den": 0, "rate": None}
    assert s["recall"] == {"num": 0, "den": 1, "rate": 0.0}
    # empty reference: every predicted field is an extra; recall undefined, precision 0
    s, c = _counts("Metformin 500 mg", "")
    assert c["extras"] == 1 and s["recall"]["rate"] is None and s["precision"] == {"num": 0, "den": 1, "rate": 0.0}
    # both empty: nothing to score, both undefined
    s, c = _counts("", "")
    assert s["precision"]["rate"] is None and s["recall"]["rate"] is None and not any(c.values())
    # a recognised name with its unit missing: substitution, value right, unit wrong
    s, c = _counts("Metformin 500", "Metformin 500 mg")
    assert (c["exact"], c["substitutions"]) == (0, 1) and s["value"]["num"] == 1 and s["unit"]["num"] == 0


def test_legacy_field_metric_is_kept_for_rescoring():
    from benchmarks import run_handwriting as RH
    pairs = [("Metformin 500 mg", "Metformin 500 mg and Metformin 500 mg"), ("Lorazepam 5 mg", "Lorazepam ,5 mg")]
    legacy = RH.field_scores_legacy(pairs)
    assert legacy["exact"] == {"num": 3, "den": 3, "rate": 1.0}  # the old answer: duplicates and ",5" collapse
    assert RH.field_scores(pairs)["exact"] == {"num": 1, "den": 3, "rate": 0.3333}


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
    first_r5 = sets.index("extended_r5")
    assert set(sets[first_r4:first_r5]) == {"extended_r4"} and first_r5 - first_r4 == 25  # round 4 after round 3
    assert set(sets[first_r5:]) == {"extended_r5"}  # round 5 after round 4
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


def test_round5_set_mixes_prose_doses_and_headers(recs):
    r5 = [r for r in recs if r["set"] == "extended_r5"]
    for cat in ("r5_sentence_start", "r5_header"):
        sub = [r for r in r5 if r["category"] == cat]
        assert {r["path"] for r in sub} == {"v1", "v2"}
        assert any(r["dose_incorrect_or_lost"] for r in sub) and any(not r["warning_warranted"] for r in sub)


def test_an_empty_cached_detector_file_is_not_usable(tmp_path, monkeypatch):
    from benchmarks import run_handwriting as RH
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path))
    (models / "fast_base-688a8b34.pt").write_bytes(b"")  # left by a failed download
    assert RH.detector_prerequisite(str(tmp_path), str(tmp_path / "missing.pt")).startswith("BLOCKED")
    (models / "fast_base-688a8b34.pt").write_bytes(b"x" * 2048)
    assert RH.detector_prerequisite(str(tmp_path), str(tmp_path / "missing.pt")) is None


# ------------------------------------------------------------------ annotated-crop failure diagnosis
ANNOTATED = os.path.join(ROOT, "reports", "review_2026-11", "e2e_ocr_2026-11-28")


def _diagnose(tmp_path, rows):
    """Run benchmarks/diagnose_annotated.py (the published command) on validation rows; its output."""
    import subprocess
    import sys
    src, out = tmp_path / "result.json", tmp_path / "diagnosis.json"
    src.write_text(json.dumps({"validation": {"rows": rows}, "test": {"rows": []}}))
    subprocess.run([sys.executable, os.path.join(ROOT, "benchmarks", "diagnose_annotated.py"), ROOT, str(src),
                    str(out)], check=True, capture_output=True, cwd=str(tmp_path))
    return json.loads(out.read_text())["validation"]


def _row(truth, shown, *, flagged):
    """A run_handwriting row showing `shown`. flagged: routed, with a handwriting reading that holds the
    shown text (ocr.common.line_confirm_numbers warns); otherwise a printed line, not routed (no warning
    unless a unit lost its number)."""
    return {"truth": truth, "shipped": shown, "printed": shown if not flagged else "Review tomorrow",
            "handwriting": shown, "routed": flagged, "shipped_conf": 0.9}


@pytest.mark.parametrize("truth, shown, wrong, with_reference_number", [
    ("Review tomorrow", "Review 7 tomorrow", 1, 0),  # number invented on a line without one
    ("Review in 7 days", "Review in days", 1, 1),
    ("Review in 7 days", "Review in 2 days", 1, 1),
    ("Review in 7 days", "Review in 7 days", 0, 1),
    ("Review tomorrow", "Review tomorrow", 0, 0),
])
def test_diagnosis_compares_numbers_on_every_line(tmp_path, truth, shown, wrong, with_reference_number):
    num = (d := _diagnose(tmp_path, [_row(truth, shown, flagged=False)]))["numbers"]
    assert num["number_wrong"] == wrong
    assert num["lines_with_reference_number"] == with_reference_number
    assert num["number_wrong"] == num["number_wrong_flagged"] + num["number_wrong_silent"]
    assert num["number_wrong_silent"] == wrong and num["number_wrong_flagged"] == 0
    assert [s["shown"] for s in d["silent_numeric_errors"]] == [shown] * wrong
    # the reference-number-only count never includes an invented number
    assert num["number_wrong_where_reference_has_number"] == wrong * with_reference_number
    assert num["number_invented_where_reference_has_none"] == wrong * (1 - with_reference_number)


def test_invented_number_is_flagged_or_silent_by_the_warning(tmp_path):
    silent = _diagnose(tmp_path, [_row("Review tomorrow", "Review 7 tomorrow", flagged=False)])
    assert silent["numbers"]["number_wrong_silent"] == 1 and silent["numbers"]["number_wrong_flagged"] == 0
    assert silent["silent_numeric_errors"] == [{"truth": "Review tomorrow", "shown": "Review 7 tomorrow",
                                                "printed": "Review 7 tomorrow", "handwriting": "Review 7 tomorrow",
                                                "routed": False, "shipped_conf": 0.9}]
    flagged = _diagnose(tmp_path, [_row("Review tomorrow", "Review 7 tomorrow", flagged=True)])
    num = flagged["numbers"]
    assert num["number_wrong"] == 1 and num["number_wrong_flagged"] == 1 and num["number_wrong_silent"] == 0
    assert flagged["silent_numeric_errors"] == []
    assert num["lines_with_reference_number"] == 0 and num["number_invented_where_reference_has_none"] == 1


def test_diagnosis_totals_over_mixed_lines(tmp_path):
    rows = [_row("Review tomorrow", "Review 7 tomorrow", flagged=True),
            _row("Review tomorrow", "Review 7 tomorrow", flagged=False),
            _row("Review in 7 days", "Review in days", flagged=False),
            _row("Review in 7 days", "Review in 2 days", flagged=True),
            _row("Review in 7 days", "Review in 7 days", flagged=False),
            _row("Review tomorrow", "Review tomorrow", flagged=False)]
    num = _diagnose(tmp_path, rows)["numbers"]
    assert (num["number_wrong"], num["number_wrong_flagged"], num["number_wrong_silent"]) == (4, 2, 2)
    assert num["lines_with_reference_number"] == 3
    assert (num["number_wrong_where_reference_has_number"], num["number_invented_where_reference_has_none"]) == (2, 2)

