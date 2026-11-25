"""Check that experimental image preparation uses the original baseline and size."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from benchmarks.run_handwriting_preprocessing import prepare
from ocr.handwriting import prep_line


def test_default_variant_keeps_the_existing_line_preparation():
    crop = np.full((24, 140, 3), 234, dtype=np.uint8)
    crop[8:16, 31:36] = 46
    np.testing.assert_array_equal(prepare(crop, "baseline", (384, 384)), prep_line(crop))


def test_lanczos_variant_uses_the_processor_target_size():
    crop = np.full((25, 140, 3), 240, dtype=np.uint8)
    crop[8:17, 31:36] = 40
    result = prepare(crop, "lanczos", (128, 96))
    assert result.shape == (96, 128, 3)
    assert result.dtype == np.uint8
    assert result.min() < result.max()


def test_stroke_variant_accentuates_adjacent_ink_pixels():
    crop = np.full((25, 140, 3), 250, dtype=np.uint8)
    crop[8:17, 31:36] = 45
    baseline = prepare(crop, "baseline", (384, 384))
    strengthened = prepare(crop, "stroke", (384, 384))
    assert strengthened.shape == baseline.shape
    assert np.all(strengthened <= baseline)
    assert np.any(strengthened < baseline)
