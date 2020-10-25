import pytest
import numpy as np
from reval.internal_baselines import select_best, evaluate_best
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


@pytest.mark.parametrize(
    "data, clustering, internal_measure, clust_range, select, expected_type, expected_output",
    (
            # silhouette
            ([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]] * 10, KMeans(), silhouette_score, [2, 3, 4], 'max',
             (float, int),
             (1, 2)),
            # davies-bouldin
            ([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]] * 10, KMeans(), davies_bouldin_score, [2, 3, 4], 'min',
             (float, int),
             (0, 2))
    )
)
def test_select_best(data, clustering, internal_measure, clust_range, select, expected_type, expected_output):
    out = select_best(data, clustering, internal_measure, clust_range, select)
    assert isinstance(out[0], expected_type)
    assert isinstance(out[1], expected_type)
    np.testing.assert_equal(out, expected_output)


@pytest.mark.parametrize(
    "data, clustering, internal_measure, clust_n, expected_type, expected_output",
    (
            # silhouette
            ([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]] * 10, KMeans(), silhouette_score, 2,
             float,
             1),
            # davies-bouldin
            ([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]] * 10, KMeans(), davies_bouldin_score, 2,
             float,
             0)
    )
)
def test_evaluate_best(data, clustering, internal_measure, clust_n, expected_type, expected_output):
    score = evaluate_best(data, clustering, internal_measure, clust_n)
    assert isinstance(score, expected_type)
    np.testing.assert_array_equal(score, expected_output)
