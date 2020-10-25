import unittest
import numpy as np
from reval.param_selection import SCParamSelection, ParamSelection
from reval.param_selection import _return_best, _return_knownbest, _create_result_table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

# Modify to test other functions and parameters
RNDLABELS_ITER = 10
CLASSIFIER = [KNeighborsClassifier(n_neighbors=5), LogisticRegression(),
              RandomForestClassifier(), SVC()]
CLUSTERING = [AgglomerativeClustering(), KMeans(), SpectralClustering()]
NCLUST_RANGE = list(range(2, 4, 1))
PROCESSES = 4
NFOLD = 2


class TestSCParamSelection(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.sc_params = {'s': CLASSIFIER, 'c': CLUSTERING}
        cls.nrand = RNDLABELS_ITER
        cls.cv = NFOLD
        cls.clust_range = NCLUST_RANGE
        cls.processes = PROCESSES
        cls.scparamselect = SCParamSelection(cls.sc_params,
                                             cls.cv,
                                             cls.nrand,
                                             cls.clust_range,
                                             cls.processes)

    def test_fit(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.scparamselect.fit(data, 2)
        self.assertIsInstance(self.scparamselect.best_param_, list)
        self.assertIsInstance(self.scparamselect.best_index_, list)
        self.assertIsInstance(self.scparamselect.cv_results_, dict)
        self.assertSequenceEqual([(self.scparamselect.cv_results_["best_nclust"][bidx],
                                   self.scparamselect.cv_results_["validation_meanerror"][bidx])
                                  for bidx in self.scparamselect.best_index_], [(2, (0.0, (0.0, 0.0)))] *
                                 len(CLASSIFIER) * len(CLUSTERING))

    def test_run_gridsearchcv(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        perf = self.scparamselect._run_gridsearchcv(data, {'s': CLASSIFIER[0],
                                                           'c': CLUSTERING[0]})
        self.assertEqual(perf, [('s', CLASSIFIER[0]), ('c', CLUSTERING[0]), ('best_nclust', 2),
                                ('mean_train_score', 0.0),
                                ('sd_train_score', 0.0),
                                ('mean_val_score', 0.0),
                                ('sd_val_score', 0.0),
                                ('validation_meanerror', (0.0, (0.0, 0.0)))])


class TestParamSelection(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.params = {'s': {'n_neighbors': [5, 10, 15]},
                      'c': {'affinity': ['euclidean', 'l1', 'l2', 'manhattan'],
                            'linkage': ['ward', 'complete', 'average', 'single']}}
        cls.s = KNeighborsClassifier()
        cls.c = AgglomerativeClustering()
        cls.nrand = RNDLABELS_ITER
        cls.cv = NFOLD
        cls.clust_range = NCLUST_RANGE
        cls.processes = PROCESSES
        cls.paramselect = ParamSelection(cls.params,
                                         cls.cv,
                                         cls.s,
                                         cls.c,
                                         cls.nrand,
                                         cls.clust_range,
                                         cls.processes)

    def test_fit(self):
        # 48 possible parameter combinations - 9 not admissible for
        # agglomerative clustering
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.paramselect.fit(data, 2)
        self.assertIsInstance(self.paramselect.best_param_, list)
        self.assertIsInstance(self.paramselect.best_index_, list)
        self.assertIsInstance(self.paramselect.cv_results_, dict)
        self.assertSequenceEqual([(self.paramselect.cv_results_["best_nclust"][bidx],
                                   self.paramselect.cv_results_["validation_meanerror"][bidx])
                                  for bidx in self.paramselect.best_index_], [(2, (0.0, (0.0, 0.0)))] *
                                 39)

    def test_run_gridsearchcv(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        perf = self.paramselect._run_gridsearchcv(data, {'n_neighbors': 5},
                                                  {'affinity': 'euclidean',
                                                   'linkage': 'ward'})
        self.assertEqual(perf, [('n_neighbors', 5), ('affinity', 'euclidean'),
                                ('linkage', 'ward'),
                                ('best_nclust', 2),
                                ('mean_train_score', 0.0),
                                ('sd_train_score', 0.0),
                                ('mean_val_score', 0.0),
                                ('sd_val_score', 0.0),
                                ('validation_meanerror', (0.0, (0.0, 0.0)))])

    def test_allowed_par(self):
        par_dict = {'affinity': 'l1',
                    'linkage': 'ward'}
        par_dict_a = {'affinity': 'euclidean',
                      'linkage': 'ward'}
        self.assertFalse(self.paramselect._allowed_par(par_dict))
        self.assertTrue(self.paramselect._allowed_par(par_dict_a))


class TestPrivate(unittest.TestCase):

    def test_return_best(self):
        val_scores = [0.0, 0.5, 1.0]
        mult_val_scores = [0.0, 0.0, 0.5, 0.0]
        self.assertEqual(_return_best(val_scores), [0])
        self.assertEqual(_return_best(mult_val_scores), [0, 1, 3])

    def test_return_knownbest(self):
        val_scores = [0.0, 0.5, 1.0]
        idx = [1, 2]
        mult_val_scores = [0.0, 0.0, 0.5, 0.0]
        mult_idx = [0, 1]
        self.assertEqual(_return_knownbest(val_scores, idx), [1])
        self.assertEqual(_return_knownbest(mult_val_scores, mult_idx), [0, 1])

    def test_create_result_table(self):
        out = [(('s', KNeighborsClassifier()), ('s', LogisticRegression())),
               (('c', SpectralClustering(n_clusters=2)), ('c', SpectralClustering(n_clusters=2))),
               (('best_nclust', 2), ('best_nclust', 2)),
               (('mean_train_score', 0.0), ('mean_train_score', 0.0)),
               (('sd_train_score', 0.0), ('sd_train_score', 0.0)),
               (('mean_val_score', 0.0), ('mean_val_score', 0.0)),
               (('sd_val_score', 0.0), ('sd_val_score', 0.0)),
               (('validation_meanerror', (0.0, (0.0, 0.0))), ('validation_meanerror', (0.0, (0.0, 0.0))))]
        self.assertIsInstance(_create_result_table(out), dict)


if __name__ == 'main':
    unittest.main()
