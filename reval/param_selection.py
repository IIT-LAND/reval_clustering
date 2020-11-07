from reval.best_nclust_cv import FindBestClustCV
from reval.relative_validation import RelativeValidation
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
import logging
import numpy as np
import itertools

from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class SCParamSelection:
    """
    Class that implements grid search cross-validation in parallel to select the best
    combination of classifier/clustering methods.

    :param sc_params: dictionary of the form {'s': list, 'c': list} including the lists
        of classifiers and clustering methods to fit to the data.
    :type sc_params: dict
    :param cv: cross-validation folds.
    :type cv: int
    :param nrand: number of random label iterations.
    :type nrand: int
    :param n_jobs: number of jobs to run in parallel, default (number of cpus - 1).
    :type n_jobs: int
    :param iter_cv: number of repeated cv, default 1.
    :type iter_cv: int
    :param clust_range: list with number of clusters to investigate, default None.
    :type clust_range: list
    :param strat: stratification vector for cross-validation splits, default None.
    :type strat: numpy array

    :attribute: cv_results_ cross-validation results that can be directly transformed to
        a dataframe. Key names: 's', 'c', 'best_nclust', 'mean_train_score', 'sd_train_score',
        'mean_val_score', 'sd_val_score', 'validation_meanerror'. Dictionary of lists.
    :attribute: best_param_ best solution(s) selected (minimum validation error). List.
    :attribute: best_index_ index/indices of the best solution(s). Values correspond to the
        rows of the `cv_results_` table. List.
    """

    def __init__(self, sc_params, cv, nrand,
                 n_jobs,
                 iter_cv=1,
                 clust_range=None,
                 strat=None):
        self.sc_params = sc_params
        if len(self.sc_params['s']) == 1 and len(self.sc_params['c']) == 1:
            raise AttributeError("Please add at least another classifier/clustering "
                                 "method to run the parameter selection.")
        self.cv = cv
        self.nrand = nrand
        self.clust_range = clust_range
        if abs(n_jobs) > mp.cpu_count():
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = abs(n_jobs)
        self.iter_cv = iter_cv
        self.strat = strat

    def fit(self, data_tr, nclass=None):
        """
        Class method that performs grid search cross-validation on training data. If the number of
        true classes is known, the method returns both the best result with the correct number of
        clusters (and minimum stability), if available, and the overall best result (overall minimum stability).
        The output reports None if the clustering algorithm does not find any cluster (e.g., HDBSCAN label
        all points as -1).

        :param data_tr: training dataset.
        :type data_tr: numpy array
        :param nclass: number of true classes, default None.
        :type nclass: int
        """
        sc_grid = list(ParameterGrid(self.sc_params))
        params = list(zip([data_tr] * len(sc_grid), sc_grid))

        logging.info(f'Running {len(params)} combinations of '
                     f'classification/clustering methods...\n')

        p = mp.Pool(processes=self.n_jobs)
        out = list(zip(*p.starmap(self._run_gridsearchcv, params)))
        p.close()
        p.join()

        # cv_results_
        res_dict = _create_result_table(out)
        SCParamSelection.cv_results_ = res_dict

        # best_param_, best_index_
        val_scores = [vs for vs in res_dict['mean_val_score'] if vs is not None]
        val_idx = [idx for idx, vs in enumerate(res_dict['mean_val_score']) if vs is not None]
        if len(val_scores) > 0:
            idx_best = [val_idx[i] for i in _return_best(val_scores)]
        else:
            logging.info(f"No clustering solutions were found with any parameter combinations.")
            return self

        out_best = []
        if nclass is not None:
            logging.info(f'True number of clusters known: {nclass}\n')
            idx = np.where(np.array(res_dict['best_nclust']) == nclass)[0]
            idx_inter = set(idx).intersection(set(idx_best))
            if len(idx_inter) > 0:
                idx_best = list(idx_inter)
            else:
                if len(idx) > 0:
                    idx_true = _return_knownbest(res_dict['mean_val_score'], idx)
                    logging.info(f'Best solution(s) with true number of clusters:')
                    for bidx in idx_true:
                        logging.info(f'Models S/C: {res_dict["s"][bidx]}/{res_dict["c"][bidx]})')
                        logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                        logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                        out_best.append([res_dict["s"][bidx], res_dict["c"][bidx],
                                         res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        logging.info(f'Best solution(s):')
        for bidx in idx_best:
            logging.info(f'Models: {res_dict["s"][bidx]}/{res_dict["c"][bidx]})')
            logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
            logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
            out_best.append([res_dict["s"][bidx], res_dict["c"][bidx],
                             res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        SCParamSelection.best_param_ = out_best
        SCParamSelection.best_index_ = idx_best

        return self

    def _run_gridsearchcv(self, data, sc):
        """
        Private function with different initializations of
        :class:`reval.best_nclust_cv.FindBestClustCV`.

        :param data: input dataset.
        :type data: numpy array
        :param sc: classifier/clustering of the form {'s':, 'c':}.
        :type sc: dict
        :return: performance list.
        :rtype: list
        """
        findclust = FindBestClustCV(s=sc['s'],
                                    c=sc['c'],
                                    nfold=self.cv,
                                    nrand=self.nrand,
                                    n_jobs=1,
                                    nclust_range=self.clust_range)

        if 'n_clusters' in sc['c'].get_params().keys():
            metric, nclbest = findclust.best_nclust(data, iter_cv=self.iter_cv, strat_vect=self.strat)
            sc['c'].n_clusters = nclbest
            tr_lab = None
        else:
            try:
                metric, nclbest, tr_lab = findclust.best_nclust(data, iter_cv=self.iter_cv, strat_vect=self.strat)
            except TypeError:
                perf = [('s', sc['s']), ('c', sc['c']), ('best_nclust', None),
                        ('mean_train_score', None),
                        ('sd_train_score', None),
                        ('mean_val_score', None),
                        ('sd_val_score', None),
                        ('validation_meanerror', None),
                        ('tr_label', None)]
                return perf

        cv_scores = findclust.cv_results_
        perf = [('s', sc['s']), ('c', sc['c']), ('best_nclust', nclbest),
                ('mean_train_score', np.mean(cv_scores.loc[cv_scores.ncl == nclbest]['ms_tr'])),
                ('sd_train_score', np.std(cv_scores.loc[cv_scores.ncl == nclbest]['ms_tr'])),
                ('mean_val_score', np.mean(cv_scores.loc[cv_scores.ncl == nclbest]['ms_val'])),
                ('sd_val_score', np.std(cv_scores.loc[cv_scores.ncl == nclbest]['ms_val'])),
                ('validation_meanerror', metric['val'][nclbest]),
                ('tr_label', tr_lab)]
        return perf


class ParamSelection(RelativeValidation):
    """
    Class that implements grid search cross-validation in parallel to select
    the best combinations of parameters for fixed classifier/clustering algorithms.

    :param params: dictionary of dictionaries of the form {'s': {classifier parameter grid},
        'c': {clustering parameter grid}}. If one of the two dictionary of parameters is not
        available, initialize key but leave dictionary empty.
    :type params: dict
    :param cv: cross-validation folds.
    :type cv: int
    :param clust_range: list with number of clusters to investigate.
    :type clust_range: list
    :param n_jobs: number of jobs to run in parallel, default (number of cpus - 1).
    :type n_jobs: int
    :param iter_cv: number of repeated cv loops, default 1.
    :type iter_cv: int
    :param strat: stratification vector for cross-validation splits, default None.
    :type strat: numpy array

    :attribute: cv_results_ cross-validation results that can be directly transformed to
        a dataframe. Key names: classifier parameters, clustering parameters,
        'best_nclust', 'mean_train_score', 'sd_train_score',
        'mean_val_score', 'sd_val_score', 'validation_meanerror'. Dictionary of lists.
    :attribute: best_param_ best solution(s) selected (minimum validation error). List.
    :attribute: best_index_ index/indices of the best solution(s). Values correspond to the
        rows of the `cv_results_` table. List.
    """

    def __init__(self, params, cv, s, c, nrand,
                 n_jobs, iter_cv=1, strat=None, clust_range=None):
        super().__init__(s, c, nrand)
        self.params = params
        self.cv = cv
        self.iter_cv = iter_cv
        self.clust_range = clust_range
        if abs(n_jobs) > mp.cpu_count():
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = abs(n_jobs)
        self.strat = strat

    def fit(self, data_tr, nclass=None):
        """
        Class method that performs grid search cross-validation on training data. It
        deals with the error due to wrong parameter combinations (e.g., ward linkage
        with no euclidean affinity). If the true number of classes is know, the method
        selects both the best parameter combination that selects the true number of clusters
        (minimum stability) and the best parameter combination that minimizes
        overall stability.

        :param data_tr: training dataset.
        :type data_tr: numpy array
        :param nclass: number of true classes, default None.
        :type nclass: int
        """
        grid = {'s': ParameterGrid(self.params['s']), 'c': ParameterGrid(self.params['c'])}
        new_grid = list(itertools.product(grid['s'], grid['c']))
        new_params = [(data_tr, ng[0], ng[1]) for ng in new_grid if self._allowed_par(ng[1])]

        if len(new_grid) != len(new_params):
            logging.info(f"Dropped {len(new_grid) - len(new_params)} out of {len(new_grid)} parameter "
                         f"combinations "
                         f"due to {self.clust_method} class requirements.")

        logging.info(f'Running {len(new_params)} combinations of '
                     f'parameters...\n')

        p = mp.Pool(processes=self.n_jobs)
        out = list(zip(*p.starmap(self._run_gridsearchcv, new_params)))
        p.close()
        p.join()

        # cv_results_
        res_dict = _create_result_table(out)
        ParamSelection.cv_results_ = res_dict

        # best_param_, best_index_
        val_scores = [vs for vs in res_dict['mean_val_score'] if vs is not None]
        val_idx = [idx for idx, vs in enumerate(res_dict['mean_val_score']) if vs is not None]
        if len(val_scores) > 0:
            idx_best = [val_idx[i] for i in _return_best(val_scores)]
        else:
            logging.info(f"No clustering solutions were found with any parameter combinations.")
            return self

        out_best = []
        if nclass is not None:
            logging.info(f'True number of clusters known: {nclass}\n')
            idx = np.where(np.array(res_dict['best_nclust']) == nclass)[0]
            idx_inter = set(idx).intersection(set(idx_best))
            if len(idx_inter) > 0:
                idx_best = list(idx_inter)
            else:
                if len(idx) > 0:
                    idx_true = _return_knownbest(res_dict['mean_val_score'], idx)
                    logging.info(f'Best solution(s) with true number of clusters:')
                    for bidx in idx_true:
                        for k in self.params['s'].keys():
                            logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')
                        for k in self.params['c'].keys():
                            logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
                        logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
                        logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
                        out_best.append([res_dict[k][bidx] for k in self.params['s'].keys()] +
                                        [res_dict[k][bidx] for k in self.params['s'].keys()] +
                                        [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        logging.info(f'Best solution(s):')
        for bidx in idx_best:
            for k in self.params['s'].keys():
                logging.info(f'Parameters classifier (S): {k}={res_dict[k][bidx]}')
            for k in self.params['c'].keys():
                logging.info(f'Parameters clustering (C): {k}={res_dict[k][bidx]}')
            logging.info(f'Validation performance: {res_dict["validation_meanerror"][bidx]}')
            logging.info(f'N clusters: {res_dict["best_nclust"][bidx]}\n')
            out_best.append([res_dict[k][bidx] for k in self.params['s'].keys()] +
                            [res_dict[k][bidx] for k in self.params['s'].keys()] +
                            [res_dict["best_nclust"][bidx], res_dict["validation_meanerror"][bidx]])
        ParamSelection.best_param_ = out_best
        ParamSelection.best_index_ = idx_best

        return self

    def _run_gridsearchcv(self, data, param_s, param_c):
        """
        Private method that initializes classifier/clustering with different
        parameter combinations and :class:`reval.best_nclust_cv.FindBestClustCV`.

        :param data: training dataset.
        :type data: numpy array
        :param param_s: dictionary of classifier parameters.
        :type: dict
        :param param_c: dictionary of clustering parameters.
        :type param_c: dict
        :return: performance list.
        :rtype: list
        """
        self.class_method.set_params(**param_s)
        self.clust_method.set_params(**param_c)
        findclust = FindBestClustCV(nfold=self.cv,
                                    s=self.class_method,
                                    c=self.clust_method,
                                    nrand=self.nrand, n_jobs=1,
                                    nclust_range=self.clust_range)
        if self.clust_range is not None:
            metric, nclbest = findclust.best_nclust(data, iter_cv=self.iter_cv, strat_vect=self.strat)
            tr_lab = None
        else:
            try:
                metric, nclbest, tr_lab = findclust.best_nclust(data, iter_cv=self.iter_cv, strat_vect=self.strat)
            except TypeError:
                perf = [(key, val) for key, val in param_s.items()] + \
                       [(key, val) for key, val in param_c.items()] + \
                       [('best_nclust', None),
                        ('mean_train_score', None),
                        ('sd_train_score', None),
                        ('mean_val_score', None),
                        ('sd_val_score', None),
                        ('validation_meanerror', None),
                        ('tr_label', None)]
                return perf

        perf = [(key, val) for key, val in param_s.items()] + \
               [(key, val) for key, val in param_c.items()] + \
               [('best_nclust', nclbest),
                ('mean_train_score', np.mean(
                    findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                ('sd_train_score', np.std(
                    findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_tr'])),
                ('mean_val_score', np.mean(
                    findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                ('sd_val_score', np.std(
                    findclust.cv_results_.loc[findclust.cv_results_.ncl == nclbest]['ms_val'])),
                ('validation_meanerror', metric['val'][nclbest]),
                ('tr_label', tr_lab)]
        return perf

    def _allowed_par(self, par_dict):
        """
        Private method that controls the allowed parameter combinations
        for hierarchical clustering.

        :param par_dict: clustering parameter grid.
        :type par_dict: dict
        :return: whether the parameter combination can be allowed.
        :rtype: bool
        """
        if isinstance(self.clust_method, AgglomerativeClustering):
            try:
                if par_dict['linkage'] == 'ward':
                    return par_dict['affinity'] == 'euclidean'
                else:
                    return True
            except KeyError:
                try:
                    return par_dict['affinity'] == 'euclidean'
                except KeyError:
                    return True
        else:
            return True


"""
Private functions
"""


def _return_best(val_scores):
    """
    Private function that returns indices corresponding to the best solution,
    i.e., those that minimize the validation stability scores.

    :param val_scores: list of validation scores averaged over cross-validation loops.
    :type val_scores: list
    :return: list of indices.
    :rtype: list
    """
    bidx = list(np.where(np.array(val_scores) == min([vs for vs in val_scores]))[0])
    return bidx


def _return_knownbest(val_perf, idx):
    """
    Private function that, given a stability score list and indices, returns the indices corresponding
    to the best solution.

    :param val_perf: list of validation scores averaged over cross-validation loops.
    :type val_perf: list
    :param idx: list of indices.
    :type idx: list
    :return: list of indices.
    :rtype: list
    """
    bidx = _return_best([val_perf[i] for i in idx])
    return [idx[b] for b in bidx]


def _create_result_table(out):
    """
    Private function that builds the performance result dictionary to be transformed to
    dataframe.

    :param out: grid search performance results.
    :type out: list
    :return: dictionary with results.
    :rtype: dict
    """
    dict_obj = {}
    for el in out:
        for key, val in el:
            if key in dict_obj:
                if not isinstance(dict_obj[key], list):
                    dict_obj[key] = [dict_obj[key]]
                dict_obj[key].append(val)
            else:
                dict_obj[key] = val
    return dict_obj
