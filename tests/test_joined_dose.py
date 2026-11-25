"""Regression cases for a misread quantity touching its unit. All text is invented."""

import pytest

from ocr import common
from ocr.api import _line


@pytest.mark.parametrize("text", [
    "Metformin 5OOmg", "Metformin 5OO mg", "(5OOmg),", "Dose O.5mL",
    "Levothyroxine 5Omcg;", "Insulin 1Ounits", "Dose Z.Smg",
])
def test_lookalike_quantity_before_known_unit(text):
    assert common.has_misread_number(text)
    line = {"text": text, "script": "printed"}
    assert common.line_confirm_numbers(line, script_known=True, flag_misread_digits=True)


@pytest.mark.parametrize("text", [
    "ID5OOmg", "5OOmgRef", "RX-5OOmg", "5OOmg_17", "5OOmetrics",
    "500mg", "0.5mL", "10units", "5OO", "Blog", "Solo", "Soil",
])
def test_valid_quantities_words_and_identifiers_are_quiet(text):
    assert not common.has_misread_number(text)
    assert not common.line_confirm_numbers({"text": text}, script_known=True, flag_misread_digits=True)


@pytest.mark.parametrize("script", ["printed", "handwritten"])
@pytest.mark.parametrize("alternate", ["printed_text", "handwriting_text"])
def test_alternate_joined_reading_keeps_the_displayed_text(script, alternate):
    # No literal digit in either reading: the prominent warning here depends on the option.
    line = {"text": "review tomorrow", alternate: "Dose Z.Smg", "script": script,
            "confidence": 0.97, "box": [0, 0, 30, 20]}
    for enabled in (True, False):
        flag = common.line_confirm_numbers(line, script_known=True, flag_misread_digits=enabled)
        assert flag is enabled
        response = _line(dict(line, confirm_numbers=flag))
        assert response["confirm_numbers"] is enabled
        assert response["text"] == "review tomorrow"


@pytest.mark.parametrize("text,quiet", [("Metformin 5OOmg", True), ("Dose Z.Smg", False)])
def test_api_preserves_joined_readings_and_each_paths_warning_policy(text, quiet):
    line = {"text": text, "confidence": 0.99, "box": [0, 0, 20, 20]}
    # v1 cannot determine script. Its conservative warning on literal digits stays on.
    for enabled in (True, False):
        v1 = _line(line, flag_misread_digits=enabled)
        assert v1["confirm_numbers"] is (enabled or quiet)
        assert v1["has_numbers"] is quiet and v1["text"] == text
        v2flag = common.line_confirm_numbers(line, script_known=True, flag_misread_digits=enabled)
        v2 = _line(dict(line, script="printed", confirm_numbers=v2flag))
        assert v2["confirm_numbers"] is enabled
        assert v2["has_numbers"] is quiet and v2["text"] == text
