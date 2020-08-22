import pytest
import numpy as np
from reval.utils import kuhn_munkres_algorithm, _build_weight_mat


@pytest.mark.parametrize(
    "true_lab, pred_lab, expected_type, expected_output",
    (
            # standard list
            ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0], np.int64, np.array([0, 0, 0, 1, 1, 1])),
            # true labels > predicted labels
            ([0, 0, 1, 2, 1, 1], [0, 0, 1, 0, 1, 1], np.int64, np.array([0, 0, 1, 0, 1, 1])),
            # true labels < predicted labels
            ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], np.int64, np.array([0, 0, 1, 1, 2, 2])),
            # standard array
            (np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0]), np.int64, np.array([0, 0, 0, 1, 1, 1]))
    )
)
def test_kuhn_munkres_algorithm(true_lab, pred_lab, expected_type, expected_output):
    out = kuhn_munkres_algorithm(true_lab, pred_lab)
    assert out.dtype == expected_type
    np.testing.assert_array_equal(out, expected_output)


@pytest.mark.parametrize(
    "true_lab, pred_lab, exception",
    (
            ([True, True], [False, False], TypeError),
            ([1.5, 1.5, 0, 0], [1, 1, 0, 0], TypeError),
            (["String1"], ["String2"], TypeError),
            ([1], [1], ValueError)
    )
)
def test_kuhn_munkres_exceptions(true_lab, pred_lab, exception):
    with pytest.raises(exception):
        kuhn_munkres_algorithm(true_lab, pred_lab)


@pytest.mark.parametrize(
    "true_lab, pred_lab, expected_type, expected_mat",
    (
            (np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0]), np.float64, np.array([[1., 0.5], [0.5, 1.]])),
    )
)
def test_build_weight_mat(true_lab, pred_lab, expected_type, expected_mat):
    mat_out = _build_weight_mat(true_lab, pred_lab)
    assert mat_out.dtype == expected_type
    np.testing.assert_array_equal(mat_out, expected_mat)


@pytest.mark.parametrize(
    "true_lab, pred_lab, exception",
    (
            ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0], TypeError),
    )
)
def test_build_weight_mat_exception(true_lab, pred_lab, exception):
    with pytest.raises(exception):
        _build_weight_mat(true_lab, pred_lab)