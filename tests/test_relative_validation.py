import unittest
import numpy as np
from reval.relative_validation import RelativeValidation
from reval.utils import kuhn_munkres_algorithm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs

# Modify to test other functions and parameters
RNDLABELS_ITER = 100
CLASSIFIER = KNeighborsClassifier(n_neighbors=15)
CLUSTERING = AgglomerativeClustering(n_clusters=2)
CLUSTERING_NEW = HDBSCAN(min_samples=10, min_cluster_size=200)
NCLUST_RANGE = [2, 20]
NFOLD = 10


class TestReval(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.s = CLASSIFIER
        cls.c = CLUSTERING
        cls.c_new = CLUSTERING_NEW
        cls.nrand = RNDLABELS_ITER
        cls.reval_cls = RelativeValidation(s=cls.s, c=cls.c, nrand=cls.nrand)
        cls.reval_cls_new = RelativeValidation(s=cls.s, c=cls.c_new, nrand=cls.nrand)

    def test_revaltraining(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        misclass, model, labels = self.reval_cls.train(data_tr)
        self.assertSequenceEqual([misclass] + [labels.tolist()],
                                 [0.0, kuhn_munkres_algorithm(labels,
                                                              np.array([1, 0] * 20)).tolist()])
        self.assertEqual(type(model), type(self.s))

        noise = make_blobs(100, 1, centers=1, random_state=42)[0]
        self.assertEqual(self.reval_cls_new.train(noise), None)

    def test_revaltest(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        data_ts = np.array([[0] * 10,
                            [1] * 10] * 10)
        model = self.s.fit(data_tr, [1, 0] * 20)
        misclass, labels = self.reval_cls.test(data_ts, model)
        self.assertSequenceEqual([misclass] + [labels.tolist()],
                                 [0.0, [1, 0] * 10])

        noise = make_blobs(100, 1, centers=1, random_state=42)[0]
        model = self.s.fit(noise, [1, 0] * 50)
        self.assertEqual(self.reval_cls_new.test(noise, model), None)

    def test_rndlabels(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        labels_tr = np.array([0, 1] * 20)
        data_ts = np.array([[0] * 10,
                            [1] * 10] * 10)
        labels_ts = np.array([0, 1] * 10)
        misclass_ts = self.reval_cls.rndlabels_traineval(data_tr,
                                                         data_ts,
                                                         labels_tr,
                                                         labels_ts)
        self.assertTrue(misclass_ts > 0)

    def test_khun_munkres_algorithm(self):
        true_lab = np.array([1, 1, 1, 0, 0, 0])
        pred_lab = np.array([0, 0, 0, 1, 1, 1])
        new_lab = kuhn_munkres_algorithm(true_lab, pred_lab)
        self.assertSequenceEqual(new_lab.tolist(), [1, 1, 1, 0, 0, 0])


if __name__ == 'main':
    unittest.main()
