"""Fast unit tests for scanner.finetune (synthetic in-memory inputs only; no datasets, no model files)."""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="fine-tuning tools: pip install -r requirements-finetune.txt")
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scanner.finetune import evaluate_real as ev
from scanner.finetune import finetune as ft
from scanner.finetune.prep_real import source_of

SQ = np.array([[10, 20], [110, 20], [110, 80], [10, 80]], float)  # TL TR BR BL, 100 x 60 px


# ---------------------------------------------------------------- match_err

def test_match_err_identical_is_zero():
    assert ev.match_err(SQ, SQ) == 0.0


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_match_err_invariant_to_cyclic_shift_and_reversal(shift, reverse):
    t = SQ[::-1] if reverse else SQ
    t = np.roll(t, shift, 0)
    assert ev.match_err(SQ, t) == pytest.approx(0.0, abs=1e-12)


def test_match_err_known_offset():
    pred = SQ + np.array([3.0, 4.0])  # every corner 5 px away
    assert ev.match_err(pred, SQ) == pytest.approx(5.0)
    assert ev.match_err(pred, np.roll(SQ[::-1], 1, 0)) == pytest.approx(5.0)


# ---------------------------------------------------------------- homography

def test_homography_maps_src_to_dst():
    src = np.array([[12, 7], [95, 15], [88, 70], [5, 60]], float)
    dst = np.array([[0, 0], [210, 0], [210, 297], [0, 297]], float)
    H = ev.homography(src, dst)
    assert H.shape == (3, 3)
    ph = np.hstack([src, np.ones((4, 1))]) @ H.T
    np.testing.assert_allclose(ph[:, :2] / ph[:, 2:], dst, atol=1e-6)


def test_homography_identity():
    np.testing.assert_allclose(ev.homography(SQ, SQ), np.eye(3), atol=1e-9)


# ---------------------------------------------------------------- iou_doc_frame

MW, MH = 210.0, 297.0
RECT = np.array([[0, 0], [MW, 0], [MW, MH], [0, MH]], float)


def test_iou_doc_frame_identical_is_one():
    assert ev.iou_doc_frame(RECT, RECT, MW, MH) == pytest.approx(1.0)
    # a perspective-distorted true quad: still 1.0 when the prediction equals it
    q = np.array([[100, 50], [700, 80], [760, 500], [60, 470]], float)
    assert ev.iou_doc_frame(q, q, MW, MH) == pytest.approx(1.0)


def test_iou_doc_frame_half_overlap():
    left_half = np.array([[0, 0], [MW / 2, 0], [MW / 2, MH], [0, MH]], float)
    assert ev.iou_doc_frame(left_half, RECT, MW, MH) == pytest.approx(0.5)
    # same geometry in a scaled image frame: the doc frame removes the scale
    assert ev.iou_doc_frame(left_half * 3, RECT * 3, MW, MH) == pytest.approx(0.5)


def test_iou_doc_frame_disjoint_is_zero():
    far = RECT + np.array([3 * MW, 0])
    assert ev.iou_doc_frame(far, RECT, MW, MH) == 0.0


# ---------------------------------------------------------------- _clip

def test_clip_inside_unchanged():
    out = ev._clip(SQ, 0, 0, 200, 200)
    np.testing.assert_allclose(out, SQ)
    assert ev._area(out) == pytest.approx(100 * 60)


def test_clip_half_outside_area():
    sq = np.array([[-50, 0], [50, 0], [50, 40], [-50, 40]], float)  # 100 x 40, left half outside x >= 0
    out = ev._clip(sq, 0, 0, 100, 100)
    assert ev._area(out) == pytest.approx(50 * 40)
    assert out[:, 0].min() >= 0


def test_clip_fully_outside_is_empty():
    assert len(ev._clip(SQ + 1000, 0, 0, 200, 200)) == 0


# ---------------------------------------------------------------- iou_img

def test_iou_img_identical_is_one():
    assert ev.iou_img(SQ, SQ, 128, 100) == pytest.approx(1.0)


def test_iou_img_disjoint_is_zero():
    a = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float)
    assert ev.iou_img(a, a + 50, 128, 100) == 0.0


# ---------------------------------------------------------------- summarise

def _df(found_pos, found_neg):
    rows = []
    for i, f in enumerate(found_pos):
        rows.append({"negative": False, "found": f, "iou_img": 0.95 - 0.1 * i, "err_px": 10.0 * (i + 1),
                     "err_pct_img": 1.0 * (i + 1)})
    for f in found_neg:
        rows.append({"negative": True, "found": f, "iou_img": np.nan, "err_px": np.nan,
                     "err_pct_img": np.nan})
    return pd.DataFrame(rows)


def test_summarise_counts_and_objectness():
    s = ev.summarise(_df([True, True, True, False], [True, False]))
    assert (s["n"], s["n_pos"], s["n_neg"]) == (6, 4, 2)
    assert s["objectness_recall"] == pytest.approx(0.75)  # 3 of 4 positives found
    assert s["objectness_precision"] == pytest.approx(0.75)  # 3 TP, 1 FP
    assert s["false_alarm_rate_on_negatives"] == pytest.approx(0.5)
    assert s["iou_img_mean"] == pytest.approx(round((0.95 + 0.85 + 0.75 + 0.65) / 4, 4))
    assert s["iou_img_ge_0.9"] == pytest.approx(0.25)
    assert s["err_px_mean"] == pytest.approx(25.0)
    assert s["err_px_median"] == pytest.approx(25.0)


def test_summarise_no_negatives():
    s = ev.summarise(_df([True, False], []))
    assert s["n_neg"] == 0
    assert s["false_alarm_rate_on_negatives"] is None
    assert s["objectness_recall"] == pytest.approx(0.5)
    assert s["objectness_precision"] == pytest.approx(1.0)


def test_summarise_nothing_found():
    s = ev.summarise(_df([False, False], [False]))
    assert s["objectness_recall"] == 0.0
    assert s["objectness_precision"] is None
    assert s["false_alarm_rate_on_negatives"] == 0.0


# ---------------------------------------------------------------- grow (warm start with frozen bins)

def _problem(seed: int, n: int = 2000, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.5, n) > 0.5).astype(int)
    return X, y


def _logloss(clf, X, y):
    return float(np.mean(ft._logloss(y, clf.predict_proba(X)[:, 1])))


@pytest.fixture()
def base_clf():
    """Small seeded classifier, fitted single-threaded (OpenMP threads are slow on a busy machine)."""
    X, y = _problem(0)
    with threadpool_limits(1):
        clf = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
        yield clf


def test_grow_on_shifted_data(base_clf):
    clf = base_clf
    X, y = _problem(1)
    Xs = X + 3.0  # shifted distribution: re-binning would misread the old trees
    n_old = clf.n_iter_
    mapper = clf._bin_mapper
    before_df = clf.decision_function(Xs)
    before_loss = _logloss(clf, Xs, y)
    log: list = []
    out = ft.grow(clf, Xs, y, n_old + 15, 0.1, log)
    assert out is clf
    assert clf.n_iter_ == n_old + 15
    assert clf._bin_mapper is mapper
    assert "_bin_data" not in clf.__dict__  # the patched method was removed again
    assert _logloss(clf, Xs, y) <= before_loss + 1e-12
    assert log and log[-1]["trees"] == n_old + 15
    staged = list(clf.staged_decision_function(Xs))
    assert len(staged) == n_old + 15
    np.testing.assert_allclose(staged[n_old - 1], before_df, rtol=0, atol=1e-10)


def _first_fit_raw_predictions(clf, run):
    """Run run() and return the raw predictions fit() computed for the existing trees on binned training data."""
    seen = []
    orig = clf._raw_predict

    def spy(X, n_threads=None):
        out = orig(X, n_threads=n_threads)
        if getattr(clf, "_in_fit", False) and not seen:
            seen.append(out[:, 0].copy())
        return out

    clf._raw_predict = spy
    try:
        run()
    finally:
        del clf._raw_predict
    assert seen, "fit() did not replay the existing trees on the training data"
    return seen[0]


def test_grow_old_trees_read_training_bins_correctly(base_clf):
    """The bug grow() works around: on warm start fit() replays the old trees on re-binned data. With the bins
    frozen, the replayed raw predictions must equal the old model's decision function on the same X."""
    clf = base_clf
    X, y = _problem(4)
    Xs = X + 3.0
    expected = clf.decision_function(Xs)
    got = _first_fit_raw_predictions(clf, lambda: ft.grow(clf, Xs, y, clf.n_iter_ + 5, 0.1))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_plain_warm_start_misreads_old_trees_on_shifted_data(base_clf):
    """Control: without grow(), scikit-learn's warm start replays the old trees on new bins and gets them wrong.
    If this ever fails, scikit-learn has fixed the behaviour and grow()'s workaround may be unnecessary."""
    clf = base_clf
    X, y = _problem(4)
    Xs = X + 3.0
    expected = clf.decision_function(Xs)
    clf.set_params(warm_start=True, max_iter=clf.n_iter_ + 5)
    got = _first_fit_raw_predictions(clf, lambda: clf.fit(Xs, y))
    assert np.max(np.abs(got - expected)) > 1e-3


def test_grow_raises_when_loss_increases(base_clf, monkeypatch):
    clf = base_clf
    X, y = _problem(3)
    calls = iter([np.zeros(len(y)), np.ones(len(y))])  # before = 0.0, after = 1.0
    monkeypatch.setattr(ft, "_logloss", lambda y_, p_: next(calls))
    with pytest.raises(RuntimeError, match="raised the training loss"):
        ft.grow(clf, X, y, clf.n_iter_ + 2, 0.1)
    assert "_bin_data" not in clf.__dict__  # the finally block restored the class method


# ---------------------------------------------------------------- source_of

@pytest.mark.parametrize("name,src", [
    ("midv500_x.jpg", "midv500"),
    ("_negative_01.jpg", "negative"),
    ("smartdoc-abc.jpg", "smartdoc"),
    ("SmartDoc_bg1_0001.jpg", "smartdoc"),
    ("__negative_7.png", "negative"),
    ("", "unnamed"),
])
def test_source_of(name, src):
    assert source_of(name) == src
