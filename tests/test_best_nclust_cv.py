import unittest
import numpy as np
from reval.best_nclust_cv import FindBestClustCV, _confint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
import math

# Modify to test other functions and parameters
RNDLABELS_ITER = 10
CLASSIFIER = KNeighborsClassifier(n_neighbors=5)
CLUSTERING = AgglomerativeClustering()
NCLUST_RANGE = list(range(2, 4, 1))
NFOLD = 2


class TestBestNclusterCV(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.s = CLASSIFIER
        cls.c = CLUSTERING
        cls.nrand = RNDLABELS_ITER
        cls.nfold = NFOLD
        cls.nclust_range = NCLUST_RANGE
        cls.findbest = FindBestClustCV(cls.nfold, cls.nclust_range, cls.s, cls.c, cls.nrand)

    def test_best_nclust(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        strat_vect = np.array([0, 1] * 20)
        metrics, best_nclust, _ = self.findbest.best_nclust(data,
                                                            strat_vect=strat_vect)
        m_tr = metrics['train'][best_nclust][0]
        m_val = metrics['val'][best_nclust][0]
        self.assertSequenceEqual([m_tr, m_val, best_nclust], [0, 0, 2])

    def test_evaluate(self):
        data_tr = np.array([[0] * 20,
                            [1] * 20] * 100)
        data_ts = np.array([[0] * 20,
                            [1] * 20] * 50)
        out = self.findbest.evaluate(data_tr, data_ts, 2)
        self.assertSequenceEqual([out.train_cllab.tolist(), out.train_acc,
                                  out.test_cllab.tolist(), out.test_acc], [[1, 0] * 100,
                                                                           1, [1, 0] * 50,
                                                                           1])

    def test_confint(self):
        dist = np.random.normal(0, 1, 1000000)
        m, error = _confint(dist)
        self.assertTrue(abs(round(m, 2)) <= 0.01 and
                        round(error * math.sqrt(len(dist)), 2) == 1.96)


if __name__ == 'main':
    unittest.main()
