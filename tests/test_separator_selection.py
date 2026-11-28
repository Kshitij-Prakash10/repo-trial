"""The benchmark can select a complete OCR candidate but must never edit its contents."""
from benchmarks.run_separator_selection import select


def test_routed_numeric_separator_disagreement():
    assert select("BP 163178", "BP 163/78", True, 0) == "BP 163/78"
    assert select("11. Aspirin 125 mg", "11. Aspirin 1.25 mg", True, 0) == "11. Aspirin 1.25 mg"


def test_preserve_unrouted_or_unrelated_readings():
    assert select("BP 163178", "BP 163/78", False, 1) == "BP 163178"
    assert select("BP 15578", "BP 163/78", True, 1) == "BP 15578"
    assert select("Take 10 units", "Take 10 mg", True, 1) == "Take 10 units"
    assert select("Warfarin 25 mg", "Aspirin 2.5 mg", True, 1) == "Warfarin 25 mg"


def test_known_failure_keeps_the_separator_selector_experimental():
    """On fresh synthetic handwriting the selector replaced the correct "BP 139/96" with the printed reader's
    "BP 13,9/96" (reports/review_2026-11/pr14_followup/separator-selection.md). It stays a benchmark-only
    experiment; the shipped rule (ocr.scan.choose_reading) keeps the handwriting reading here, see
    tests/test_handwriting.py::test_shipped_selection_keeps_a_blood_pressure_reading."""
    assert select("BP 139/96", "BP 13,9/96", True, 1) == "BP 13,9/96"
