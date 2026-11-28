"""Check that full-page evaluation keeps unmatched lines and numerically significant errors."""
import numpy as np

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


def test_geometry_tracks_clockwise_orientation_in_edge_coordinates():
    quad = np.array([[2, 1], [6, 1], [6, 4], [2, 4]])
    assert rotated_bounds(quad, 10, 8, 1) == [4, 2, 7, 6]
    assert rotated_bounds(quad, 10, 8, 2) == [4, 4, 8, 7]
    assert rotated_bounds(quad, 10, 8, 3) == [1, 4, 4, 8]
