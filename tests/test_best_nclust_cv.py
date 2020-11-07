import unittest
import numpy as np
from reval.best_nclust_cv import FindBestClustCV, _confint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
import math
import hdbscan
from sklearn.datasets import make_blobs

# Modify to test other functions and parameters
RNDLABELS_ITER = 10
CLASSIFIER = KNeighborsClassifier(n_neighbors=5)
CLUSTERING = AgglomerativeClustering()
NCLUST_RANGE = list(range(2, 4, 1))
NFOLD = 2
N_JOBS = 7
NEW_CLUSTERING = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=200)
NEW_NCLUST_RANGE = None


class TestBestNclusterCV(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.s = CLASSIFIER
        cls.c = CLUSTERING
        cls.nrand = RNDLABELS_ITER
        cls.nfold = NFOLD
        cls.nclust_range = NCLUST_RANGE
        cls.n_jobs = N_JOBS
        cls.findbest = FindBestClustCV(cls.s, cls.c, cls.nrand, cls.nfold,
                                       cls.n_jobs, cls.nclust_range)

    def test_best_nclust(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        strat_vect = np.array([0, 1] * 20)
        metrics, best_nclust = self.findbest.best_nclust(data,
                                                         strat_vect=strat_vect)
        m_tr = metrics['train'][best_nclust][0]
        m_val = metrics['val'][best_nclust][0]
        self.assertSequenceEqual([m_tr, m_val, best_nclust], [0, 0, 2])

    def test_evaluate(self):
        data_tr = np.array([[0] * 20,
                            [1] * 20] * 100)
        data_ts = np.array([[0] * 20,
                            [1] * 20] * 50)
        out = self.findbest.evaluate(data_tr, data_ts, nclust=2)
        self.assertSequenceEqual([out.train_cllab.tolist(), out.train_acc,
                                  out.test_cllab.tolist(), out.test_acc], [[1, 0] * 100,
                                                                           1, [1, 0] * 50,
                                                                           1])

    def test_confint(self):
        dist = np.random.normal(0, 1, 1000000)
        m, error = _confint(dist)
        self.assertTrue(abs(round(m, 2)) <= 0.01 and
                        round(error * math.sqrt(len(dist)), 2) == 1.96)


class NewTestBestNclusterCV(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.s = CLASSIFIER
        cls.c = NEW_CLUSTERING
        cls.c_work = hdbscan.HDBSCAN()
        cls.nrand = RNDLABELS_ITER
        cls.nfold = NFOLD
        cls.nclust_range = NEW_NCLUST_RANGE
        cls.n_jobs = N_JOBS
        cls.findbest = FindBestClustCV(cls.s, cls.c, cls.nrand, cls.nfold,
                                       cls.n_jobs, cls.nclust_range)
        cls.findbest_bis = FindBestClustCV(cls.s, cls.c_work, cls.nrand, cls.nfold,
                                           cls.n_jobs, cls.nclust_range)

    def test_best_nclust(self):
        noise = make_blobs(100, 2, centers=1, random_state=42)[0]
        self.assertEqual(self.findbest.best_nclust(noise), None)

        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        strat_vect = np.array([0, 1] * 20)
        metrics, best_nclust, tr_lab = self.findbest_bis.best_nclust(data,
                                                                     strat_vect=strat_vect)
        m_tr = metrics['train'][best_nclust][0]
        m_val = metrics['val'][best_nclust][0]
        self.assertSequenceEqual([m_tr, m_val, best_nclust], [0, 0, 2])

    def test_evaluate(self):
        data_tr = np.array([[0] * 20,
                            [1] * 20] * 100)
        data_ts = np.array([[0] * 20,
                            [1] * 20] * 50)
        out = self.findbest_bis.evaluate(data_tr, data_ts, nclust=None, tr_lab=[0, 1] * 100)

        self.assertSequenceEqual([out.train_cllab.tolist(), out.train_acc,
                                  out.test_cllab.tolist(), out.test_acc], [[0, 1] * 100,
                                                                           1, [0, 1] * 50,
                                                                           1])

        noise_tr = make_blobs(100, 1, centers=1, random_state=42)[0]
        noise_ts = make_blobs(50, 1, centers=1, random_state=42)[0]
        self.assertEqual(self.findbest.evaluate(noise_tr, noise_ts, nclust=None, tr_lab=[-1] * 100), None)
        self.assertEqual(self.findbest.evaluate(noise_tr, noise_ts, nclust=None, tr_lab=[0, 1] * 50), None)


if __name__ == 'main':
    unittest.main()
