import unittest
import numpy as np
from reval.best_nclust_cv import FindBestClustCV, _confint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering


class TestBestNclusterCV(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.s = KNeighborsClassifier(n_neighbors=5)
        cls.c = AgglomerativeClustering()
        cls.nrand = 10
        cls.nfold = 2
        cls.nclust_range = [2, 4]
        cls.findbest = FindBestClustCV(cls.nfold, cls.nclust_range, cls.s, cls.c, cls.nrand)

    def test_best_nclust(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        strat_vect = np.array([0, 1] * 20)
        metrics, best_nclust = self.findbest.best_nclust(data,
                                                         strat_vect=strat_vect)
        m_tr = metrics['train'][best_nclust][0]
        m_val = metrics['val'][best_nclust][0]
        self.assertSequenceEqual([m_tr, m_val, best_nclust], [0, 0, 2])

    def test_confint(self):
        dist = np.random.normal(0, 1, 1000000)
        m, error = _confint(dist)
        self.assertTrue(abs(round(m, 2)) <= 0.01 and round(error * 1000, 2) == 1.96)


if __name__ == 'main':
    unittest.main()
