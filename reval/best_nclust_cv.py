from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from reval.relative_validation import RelativeValidation
from collections import namedtuple
from scipy import stats
import numpy as np
import math
import logging
import multiprocessing as mp
import itertools
import pandas as pd

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class FindBestClustCV(RelativeValidation):
    """
    Child class of :class:`reval.relative_validation.RelativeValidation`.
    It performs (repeated) k-fold cross validation on the training set to
    select the best number of clusters, i.e., the number that minimizes the
    normalized stability (i.e., average misclassification
    error/asymptotic misclassification rate).

    :param nfold: number of CV folds.
    :type nfold: int
    :param nclust_range: list with clusters to look for.
    :type nclust_range: list of int
    :param s: classification object inherited
        from :class:`reval.relative_validation.RelativeValidation`.
    :type s: class
    :param c: clustering object inherited
        from :class:`reval.relative_validation.RelativeValidation`.
    :type c: class
    :param nrand: number of random labeling iterations
        to compute asymptotic misclassification rate, inherited from
        :class:`reval.relative_validation.RelativeValidation` class.
    :type nrand: int
    :param n_jobs: number of processes to be run in parallel, default 1.
    :type n_jobs: int

    :attribute: cv_results_ dataframe with cross validation results. Columns are
        ncl = number of clusters; ms_tr = misclassification training;
        ms_val = misclassification validation.
    """

    def __init__(self, nclust_range, s, c, nrand, nfold=2, n_jobs=1):
        """
        Construct method.
        """
        super().__init__(s, c, nrand)
        self.nfold = nfold
        self.nclust_range = nclust_range
        if abs(n_jobs) > mp.cpu_count():
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = abs(n_jobs)

    def best_nclust(self, data, iter_cv=1, strat_vect=None):
        """
        This method takes as input the training dataset and the
        stratification vector (if available) and performs a
        (repeated) CV procedure to select the best number of clusters that minimizes
        normalized stability.

        :param data: training dataset.
        :type data: ndarray, (n_samples, n_features)
        :param iter_cv: number of iteration for repeated CV, default 1.
        :type iter_cv: integer
        :param strat_vect: vector for stratification, defaults to None.
        :type strat_vect: ndarray, (n_samples,)
        :return: CV metrics for training and validation sets, best number of clusters,
            misclassification errors at each CV iteration.
        :rtype: dictionary, int, dictionary
        """
        data_array = np.array(data)
        reval = RelativeValidation(self.class_method, self.clust_method, self.nrand)

        if strat_vect is not None:
            kfold = RepeatedStratifiedKFold(n_splits=self.nfold, n_repeats=iter_cv, random_state=42)
        else:
            kfold = RepeatedKFold(n_splits=self.nfold, n_repeats=iter_cv, random_state=42)
        fold_gen = kfold.split(data_array, strat_vect)

        params = list(itertools.product([[data_array, reval]],
                                        self.nclust_range, fold_gen))
        if self.n_jobs > 1:
            p = mp.Pool(processes=self.n_jobs)
            miscl = list(zip(*p.starmap(self._fit, params)))
            p.close()
            p.join()
            out = list(zip(*miscl))
        else:
            miscl = []
            for p in params:
                miscl.append(self._fit(p[0], p[1], p[2]))
            out = miscl

        # return dataframe attribute (cv_results_) with cv scores
        cv_results_ = pd.DataFrame(out,
                                   columns=['ncl', 'ms_tr', 'ms_val'])
        FindBestClustCV.cv_results_ = cv_results_

        metrics = {'train': {}, 'val': {}}
        for ncl in cv_results_.ncl.unique():
            norm_stab_tr = cv_results_.loc[cv_results_.ncl == ncl]['ms_tr']
            norm_stab_val = cv_results_.loc[cv_results_.ncl == ncl]['ms_val']
            metrics['train'][ncl] = (np.mean(norm_stab_tr), _confint(norm_stab_tr))
            metrics['val'][ncl] = (np.mean(norm_stab_val), _confint(norm_stab_val))

        val_score = np.array([val[0] for val in metrics['val'].values()])
        bestscore = min(val_score)
        # select the cluster with the minimum misclassification error
        # and the maximum number of clusters
        bestncl = self.nclust_range[np.flatnonzero(val_score == bestscore)[-1]]

        return metrics, bestncl

    def evaluate(self, data_tr, data_ts, nclust):
        """
        Method that applies the selected clustering algorithm with the best number of clusters
        to the test set. It returns clustering labels.

        :param data_tr: training dataset.
        :type data_tr: ndarray, (n_samples, n_features)
        :param data_ts: test dataset.
        :type data_ts: ndarray, (n_samples, n_features)
        :param nclust: best number of clusters.
        :type nclust: int
        :return: labels and accuracy for both training and test sets.
        :rtype: namedtuple, (train_cllab, train_acc, test_cllab, test_acc)
        """
        self.clust_method.n_clusters = nclust
        tr_misc, modelfit, labels_tr = super().train(data_tr)
        ts_misc, labels_ts = super().test(data_ts, modelfit)
        Eval = namedtuple('Eval',
                          ['train_cllab', 'train_acc', 'test_cllab', 'test_acc'])
        out = Eval(labels_tr, 1 - tr_misc, labels_ts, 1 - ts_misc)
        return out

    @staticmethod
    def _fit(data_obj, ncl, idxs):
        data_array, reval = data_obj
        tr_idx, val_idx = idxs
        tr_set, val_set = data_array[tr_idx], data_array[val_idx]
        reval.clust_method.n_clusters = ncl

        # Case in which the number of clusters identified is not sufficient
        # (it happens particularly for SpectralClustering)
        try:
            miscl_tr, modelfit, tr_labels = reval.train(tr_set)
            miscl_val, val_labels = reval.test(val_set, modelfit)
            rndmisc_mean_val = reval.rndlabels_traineval(tr_set, val_set,
                                                         tr_labels,
                                                         val_labels)
            # If random labeling gives perfect prediction, substitute
            # misclassification for validation loop with 1.0
            if rndmisc_mean_val > 0.0:
                ms_val = miscl_val / rndmisc_mean_val
            else:
                ms_val = 1.0
            return ncl, miscl_tr, ms_val
        except ValueError:
            pass


def _confint(vect):
    """
    Private function to compute confidence interval.

    :param vect: performance scores.
    :type vect: array-like
    :return: mean and error.
    :rtype: tuple
    """
    mean = np.mean(vect)
    # interval = 1.96 * math.sqrt((mean * (1 - mean)) / len(vect))
    interval = stats.t.ppf(1 - (0.05 / 2), len(vect) - 1) * (np.std(vect) / math.sqrt(len(vect)))
    return mean, interval
