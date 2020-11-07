import unittest
import numpy as np
from reval.param_selection import SCParamSelection, ParamSelection
from reval.param_selection import _return_best, _return_knownbest, _create_result_table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs

# Modify to test other functions and parameters
RNDLABELS_ITER = 10
CLASSIFIER = [KNeighborsClassifier(n_neighbors=5),
              LogisticRegression(solver='liblinear'),
              RandomForestClassifier(),
              SVC()]
CLUSTERING = [AgglomerativeClustering(), HDBSCAN(min_samples=10, min_cluster_size=200)]
CLUSTERING_MOD = [AgglomerativeClustering(), HDBSCAN()]
NCLUST_RANGE = list(range(2, 4, 1))
PROCESSES = 1
NFOLD = 2


class TestSCParamSelection(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.sc_params = {'s': CLASSIFIER, 'c': CLUSTERING}
        cls.sc_hdbscan = {'s': CLASSIFIER, 'c': [CLUSTERING[-1]]}
        cls.sc_hdbscanmod = {'s': CLASSIFIER, 'c': [CLUSTERING_MOD[-1]]}
        cls.nrand = RNDLABELS_ITER
        cls.cv = NFOLD
        cls.clust_range = NCLUST_RANGE
        cls.processes = PROCESSES
        cls.iter_cv = 1
        cls.scparamselect = SCParamSelection(sc_params=cls.sc_params,
                                             cv=cls.cv,
                                             nrand=cls.nrand,
                                             clust_range=cls.clust_range,
                                             n_jobs=cls.processes,
                                             iter_cv=cls.iter_cv)
        cls.scparamselect_noclrange = SCParamSelection(sc_params=cls.sc_hdbscan,
                                                       cv=cls.cv,
                                                       nrand=cls.nrand,
                                                       n_jobs=cls.processes,
                                                       clust_range=None,
                                                       iter_cv=cls.iter_cv)
        cls.scparamselect_range = SCParamSelection(sc_params=cls.sc_hdbscanmod,
                                                   cv=cls.cv,
                                                   nrand=cls.nrand,
                                                   n_jobs=cls.processes,
                                                   clust_range=cls.clust_range,
                                                   iter_cv=cls.iter_cv)

    def test_fit(self):
        """
        HDBSCAN fails to detect clusters.
        """
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.scparamselect.fit(data, 2)
        self.assertIsInstance(self.scparamselect.best_param_, list)
        self.assertIsInstance(self.scparamselect.best_index_, list)
        self.assertIsInstance(self.scparamselect.cv_results_, dict)
        self.assertSequenceEqual([(self.scparamselect.cv_results_["best_nclust"][bidx],
                                   self.scparamselect.cv_results_["validation_meanerror"][bidx])
                                  for bidx in self.scparamselect.best_index_], [(2, (0.0, (0.0, 0.0)))] *
                                 len(CLASSIFIER) * (len(CLUSTERING) - 1))

    def test_fit_hdbscan(self):
        """
        Only HDBSCAN is selected as clustering.
        """
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.scparamselect_noclrange.fit(data, 2)
        self.assertIsInstance(self.scparamselect_noclrange.cv_results_, dict)
        self.assertSequenceEqual([(self.scparamselect_noclrange.cv_results_["best_nclust"],
                                   self.scparamselect_noclrange.cv_results_["validation_meanerror"])],
                                 [([None] *
                                   len(CLASSIFIER), [None] *
                                   len(CLASSIFIER))])

    def test_fit_hdbscan_mod(self):
        """
        Only HDBSCAN is selected as clustering.
        """
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.scparamselect_range.fit(data, 2)
        self.assertIsInstance(self.scparamselect_range.cv_results_, dict)
        self.assertSequenceEqual([(self.scparamselect_range.cv_results_["best_nclust"][bidx],
                                   self.scparamselect_range.cv_results_["validation_meanerror"][bidx])
                                  for bidx in self.scparamselect_range.best_index_], [(2, (0.0, (0.0, 0.0)))] *
                                 len(CLASSIFIER))

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
                                ('validation_meanerror', (0.0, (0.0, 0.0))),
                                ('tr_label', None)])

    def test_fit_noclass(self):
        noise = make_blobs(100, 2, centers=1, random_state=42)[0]
        perf = self.scparamselect_noclrange._run_gridsearchcv(noise, {'s': CLASSIFIER[0],
                                                                      'c': CLUSTERING[-1]})
        self.assertIsInstance(perf, list)
        perf = self.scparamselect_range._run_gridsearchcv(noise, {'s': CLASSIFIER[0],
                                                                  'c': CLUSTERING[-1]})
        self.assertIsInstance(perf, list)


class TestParamSelection(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.params = {'s': {'n_neighbors': [5, 10, 15]},
                      'c': {'affinity': ['euclidean', 'l1', 'l2', 'manhattan'],
                            'linkage': ['ward', 'complete', 'average', 'single']}}
        cls.params_bis = {'s': {'n_neighbors': [5, 10, 15]},
                          'c': {'min_samples': [10, None],
                                'min_cluster_size': [5, 200]}}
        cls.params_nocomb = {'s': {'n_neighbors': [5, 10, 15]},
                             'c': {'min_samples': [10, None],
                                   'min_cluster_size': [200]}}
        cls.s = KNeighborsClassifier()
        cls.c = AgglomerativeClustering()
        cls.c_bis = HDBSCAN()
        cls.nrand = RNDLABELS_ITER
        cls.cv = NFOLD
        cls.clust_range = NCLUST_RANGE
        cls.processes = PROCESSES
        cls.paramselect = ParamSelection(params=cls.params,
                                         cv=cls.cv,
                                         s=cls.s,
                                         c=cls.c,
                                         nrand=cls.nrand,
                                         n_jobs=cls.processes,
                                         iter_cv=1,
                                         clust_range=cls.clust_range)
        cls.paramselect_bis = ParamSelection(params=cls.params_bis,
                                             cv=cls.cv,
                                             s=cls.s,
                                             c=cls.c_bis,
                                             nrand=cls.nrand,
                                             n_jobs=cls.processes,
                                             iter_cv=1)
        cls.paramselect_nocomb = ParamSelection(params=cls.params_nocomb,
                                                cv=cls.cv,
                                                s=cls.s,
                                                c=cls.c_bis,
                                                nrand=cls.nrand,
                                                n_jobs=cls.processes,
                                                iter_cv=1)

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

    def test_bis_fit(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.paramselect_bis.fit(data, 2)
        self.assertIsInstance(self.paramselect_bis.best_param_, list)
        self.assertIsInstance(self.paramselect_bis.best_index_, list)
        self.assertIsInstance(self.paramselect_bis.cv_results_, dict)
        self.assertSequenceEqual([(self.paramselect_bis.cv_results_["best_nclust"][bidx],
                                   self.paramselect_bis.cv_results_["validation_meanerror"][bidx])
                                  for bidx in self.paramselect_bis.best_index_], [(2, (0.0, (0.0, 0.0)))] *
                                 6)

    def test_nocomb_fit(self):
        data = np.array([[0] * 20,
                         [1] * 20] * 20)
        self.paramselect_nocomb.fit(data, 2)
        self.assertIsInstance(self.paramselect_nocomb.cv_results_, dict)
        self.assertSequenceEqual([(self.paramselect_nocomb.cv_results_["best_nclust"],
                                   self.paramselect_nocomb.cv_results_["validation_meanerror"])
                                  ], [([None] * 6, [None] * 6)])

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
                                ('validation_meanerror', (0.0, (0.0, 0.0))),
                                ('tr_label', None)])

        perf_bis = self.paramselect_bis._run_gridsearchcv(data, {'n_neighbors': 5},
                                                          {'min_samples': None,
                                                           'min_cluster_size': 5})
        self.assertEqual(perf_bis, [('n_neighbors', 5), ('min_samples', None),
                                    ('min_cluster_size', 5),
                                    ('best_nclust', 2),
                                    ('mean_train_score', 0.0),
                                    ('sd_train_score', 0.0),
                                    ('mean_val_score', 0.0),
                                    ('sd_val_score', 0.0),
                                    ('validation_meanerror', (0.0, (0.0, 0.0))),
                                    ('tr_label', [1, 0] * 20)])

        perf_nocomb = self.paramselect_bis._run_gridsearchcv(data, {'n_neighbors': 5},
                                                             {'min_samples': None,
                                                              'min_cluster_size': 200})
        self.assertEqual(perf_nocomb, [('n_neighbors', 5), ('min_samples', None),
                                       ('min_cluster_size', 200),
                                       ('best_nclust', None),
                                       ('mean_train_score', None),
                                       ('sd_train_score', None),
                                       ('mean_val_score', None),
                                       ('sd_val_score', None),
                                       ('validation_meanerror', None),
                                       ('tr_label', None)])

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
