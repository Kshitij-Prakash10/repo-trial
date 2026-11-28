"""Check that full-page evaluation keeps unmatched lines and numerically significant errors."""
import numpy as np
import pytest

from benchmarks.fields import match_fields, parse_fields
from benchmarks.run_full_pages import align, rotated_bounds, score


def test_extra_and_missing_lines_do_not_shift_later_matches():
    reference = ["One", "Two", "Three"]
    prediction = ["Extra", "One", "Three"]
    assert align(reference, prediction) == [(None, 0), (0, 1), (1, None), (2, 2)]
    assert align([], ["extra"]) == [(None, 0)]
    assert align(["missing"], []) == [(0, None)]


def test_missing_field_and_changed_decimal_count_in_denominators():
    ref = ["Lactulose 2.5 mL", "Metformin 500 mg"]
    pred = ["Lactulose 25 ml"]
    pg = dict(reference=ref, detected=pred, alignment=align(ref, pred))
    metrics = score([pg], "detected")
    assert metrics["numeric_lines"]["den"] == 2
    assert metrics["missing_drug"] == 1
    assert metrics["fields"]["unit"]["num"] == 1
    assert metrics["case_sensitive_unit"]["num"] == 0
    assert metrics["exact_decimal_value"] == {"num": 0, "den": 1, "rate": 0.0}
    assert metrics["line"]["word_errors"] == 5


def test_extra_field_on_an_inserted_line_is_counted():
    ref = ["Metformin 500 mg"]
    pred = ["Metformin 500 mg", "Metformin 500 mg"]  # the second line is inserted output
    pg = dict(reference=ref, detected=pred, alignment=align(ref, pred))
    m = score([pg], "detected")
    c = m["fields"]["counts"]
    assert (c["exact"], c["omissions"], c["extras"]) == (1, 0, 1)
    assert m["fields"]["precision"]["den"] == 2 and m["fields"]["recall"] == {"num": 1, "den": 1, "rate": 1.0}
    assert m["legacy"]["fields"]["exact"]["den"] == 1  # the old metric dropped the inserted line
    # line-level accounting is unchanged by the field fix
    assert m["exact_lines"] == {"num": 1, "den": 1, "rate": 1.0} and m["line"]["n"] == 2


def test_repeated_reference_name_is_not_credited_twice():
    ref = ["Metformin 500 mg and Metformin 500 mg"]
    pred = ["Metformin 500 mg"]
    m = score([dict(reference=ref, detected=pred, alignment=align(ref, pred))], "detected")
    assert m["fields"]["counts"]["omissions"] == 1 and m["missing_drug"] == 1
    assert m["legacy"]["missing_drug"] == 0 and m["legacy"]["fields"]["exact"]["num"] == 2


def test_decimal_comma_and_leading_separator_are_in_the_decimal_denominator():
    ref = ["Lorazepam 0,5 mg", "Metformin ,5 mg", "Review 12/04/2026", "Ramipril 2.5 mg"]
    pred = ["Lorazepam 5 mg", "Metformin 5 mg", "Review 12/04/2026", "Ramipril 2.5 mg"]
    m = score([dict(reference=ref, detected=pred, alignment=align(ref, pred))], "detected")
    assert m["exact_decimal_value"] == {"num": 1, "den": 3, "rate": 0.3333}
    assert m["legacy"]["exact_decimal_value"] == {"num": 1, "den": 1, "rate": 1.0}  # dot-only, comma lost


@pytest.mark.parametrize("pred, unit_case_num", [
    ("Metformin 5 mL and Metformin 5 ml", 2),  # both case-exact occurrences exist, swapped
    ("Metformin 5 ml and Metformin 5 ml", 1),
    ("Metformin 5 ml", 1),
])
def test_repeated_units_are_matched_case_sensitively_for_unit_case(pred, unit_case_num):
    ref = ["Metformin 5 ml and Metformin 5 mL"]
    m = score([dict(reference=ref, detected=[pred], alignment=align(ref, [pred]))], "detected")
    assert m["case_sensitive_unit"] == {"num": unit_case_num, "den": 2, "rate": unit_case_num / 2}
    # the case-insensitive field scores, decimal credit and omissions keep their pairing
    n_pred = len(parse_fields(pred))
    c = m["fields"]["counts"]
    assert (c["exact"], c["omissions"], c["extras"]) == (n_pred, 2 - n_pred, 0)
    assert m["missing_drug"] == 2 - n_pred and m["fields"]["unit"]["num"] == n_pred
    assert m["legacy"]["case_sensitive_unit"]["den"] == 2


def test_case_distinct_swap_in_order_with_case_preserved_keys():
    ref = parse_fields("Metformin 5 ml and Metformin 5 mL", preserve_unit_case=True)
    pred = parse_fields("Metformin 5 mL and Metformin 5 ml", preserve_unit_case=True)
    m = match_fields(pred, ref, unit_case=True)
    assert (m["exact"], m["exact_in_order"], m["unit_correct"]) == (2, 1, 2)
    assert match_fields(pred, ref)["exact_in_order"] == 2  # without case the swap is invisible


@pytest.mark.parametrize("ref, pred, decimal, missing, incomplete, unit_case_num", [
    # Case-exact pairing would give the reference 1.5 mg (value wrong) instead of 0.5 MG (value right).
    ("Lorazepam 0.5 mg", "Lorazepam 1.5 mg and Lorazepam 0.5 MG", 1, 0, 0, 1),
    ("Lorazepam 0.5 mg and Lorazepam 1.5 mg", "Lorazepam 1.5 MG", 1, 1, 0, 0),
    # ... or the occurrence without a unit (incomplete) instead of 0.5 MG.
    ("Lorazepam 0.5 mg", "Lorazepam 0.5 and Lorazepam 0.5 MG", 1, 0, 0, 0),
])
def test_unit_case_pairing_leaves_decimal_credit_and_omissions_alone(ref, pred, decimal, missing, incomplete,
                                                                     unit_case_num):
    """Only case_sensitive_unit uses the case-exact pairing; decimal credit, omissions and incomplete
    fields keep the case-insensitive pairing of the field scores."""
    m = score([dict(reference=[ref], detected=[pred], alignment=align([ref], [pred]))], "detected")
    assert m["exact_decimal_value"]["num"] == decimal
    assert m["missing_drug"] == missing == m["fields"]["counts"]["omissions"]
    assert m["incomplete_fields_with_recognised_drug"] == incomplete
    assert m["case_sensitive_unit"]["num"] == unit_case_num


def test_geometry_tracks_clockwise_orientation_in_edge_coordinates():
    quad = np.array([[2, 1], [6, 1], [6, 4], [2, 4]])
    assert rotated_bounds(quad, 10, 8, 1) == [4, 2, 7, 6]
    assert rotated_bounds(quad, 10, 8, 2) == [4, 4, 8, 7]
    assert rotated_bounds(quad, 10, 8, 3) == [1, 4, 4, 8]


def test_settings_record_the_equivalent_app_request():
    from benchmarks.run_full_pages import APP_SETTINGS, scanner_settings
    assert APP_SETTINGS == {"page_scanner": False, "deskew": False}
    on = scanner_settings(True, False)
    assert on["api_request"] == {"scanner": "1", "deskew": "0"} and on["matches_app_default"] is False
    off = scanner_settings(False, False)
    assert off["api_request"] == {"scanner": "0", "deskew": "0"} and off["matches_app_default"] is True
