from sklearn.model_selection import StratifiedKFold, KFold
from reval.relative_validation import RelativeValidation
from collections import namedtuple
from scipy import stats
import numpy as np
import math


class FindBestClustCV(RelativeValidation):
    """
    Child class of :class:`reval.relative_validation.RelativeValidation`.
    It performs k-fold cross validation on the training set to
    select the best number of clusters, i.e., the number that minimizes the
    normalized stability (i.e., average misclassification error/asymptotic misclassification rate).

    :param nfold: number of CV folds
    :type nfold: int
    :param nclust_range: list with minimum and maximum cluster numbers to look for
    :type nclust_range: list ([int, int])
    :param s: classification object inherited from :class:`reval.relative_validation.RelativeValidation`
    :type s: class
    :param c: clustering object inherited from :class:`reval.relative_validation.RelativeValidation`
    :type c: class
    :param nrand: number of random labeling iterations to compute asymptotic misclassification rate, inherited from
        :class:`reval.relative_validation.RelativeValidation` class
    :type nrand: int
    """

    def __init__(self, nfold, nclust_range, s, c, nrand):
        """
        Construct method
        """
        super().__init__(s, c, nrand)
        self.nfold = nfold
        self.nclust_range = nclust_range

    def best_nclust(self, data, strat_vect=None):
        """
        This method takes as input the training dataset and the
        stratification vector (if available) and performs a
        CV to select the best number of clusters that minimizes
        normalized stability.

        :param data: training dataset
        :type data: ndarray, (n_samples, n_features)
        :param strat_vect: vector for stratification, defaults to None
        :type strat_vect: ndarray, (n_samples,)
        :return: CV metrics for training and validation sets, best number of clusters,
            misclassification errors at each CV iteration
        :rtype: dictionary, int, dictionary
        """
        data_array = np.array(data)
        reval = RelativeValidation(self.class_method, self.clust_method, self.nrand)
        metrics = {'train': {}, 'val': {}}
        check_dist = {'train': {}, 'val': {}}
        for ncl in range(self.nclust_range[0], self.nclust_range[1]):
            if strat_vect is not None:
                kfold = StratifiedKFold(n_splits=self.nfold)
                fold_gen = kfold.split(data_array, strat_vect)
            else:
                kfold = KFold(n_splits=self.nfold)
                fold_gen = kfold.split(data_array)
            norm_stab_tr, norm_stab_val = [], []
            for tr_idx, val_idx in fold_gen:
                tr_set, val_set = data_array[tr_idx], data_array[val_idx]
                reval.clust_method.n_clusters = ncl
                miscl_tr, modelfit, tr_labels = reval.train(tr_set)
                miscl_val, val_labels = reval.test(val_set, modelfit)
                rndmisc_mean_val = reval.rndlabels_traineval(tr_set, val_set,
                                                             tr_labels,
                                                             val_labels)
                ms_val = miscl_val / rndmisc_mean_val
                norm_stab_tr.append(miscl_tr)
                norm_stab_val.append(ms_val)
                check_dist['train'].setdefault(ncl, list()).append(miscl_tr)
                check_dist['val'].setdefault(ncl, list()).append(ms_val)
            metrics['train'][ncl] = (np.mean(norm_stab_tr), _confint(norm_stab_tr))
            metrics['val'][ncl] = (np.mean(norm_stab_val), _confint(norm_stab_val))
        val_score = np.array([val[0] for val in metrics['val'].values()])
        bestscore = min(val_score)
        # select the cluster with the minimum misclassification error
        # and the maximum number of clusters
        bestncl = np.flatnonzero(val_score == bestscore)[-1] + self.nclust_range[0]
        return metrics, bestncl, check_dist

    def evaluate(self, data_tr, data_ts, nclust):
        """
        Method that applies the selected clustering algorithm with the best number of clusters
        to the test set. It returns clustering labels.

        :param data_tr: training dataset
        :type data_tr: ndarray, (n_samples, n_features)
        :param data_ts: test dataset
        :type data_ts: ndarray, (n_samples, n_features)
        :param nclust: best number of clusters
        :type nclust: int
        :return: labels and accuracy for both training and test sets
        :rtype: namedtuple, (train_cllab, train_acc, test_cllab, test_acc)
        """
        self.clust_method.n_clusters = nclust
        tr_misc, modelfit, labels_tr = super().train(data_tr)
        ts_misc, labels_ts = super().test(data_ts, modelfit)
        Eval = namedtuple('Eval',
                          ['train_cllab', 'train_acc', 'test_cllab', 'test_acc'])
        out = Eval(labels_tr, 1 - tr_misc, labels_ts, 1 - ts_misc)
        return out


def _confint(vect):
    """
    Private function to compute confidence interval.

    :param vect: performance scores
    :type vect: list
    :return: mean and error
    :rtype: tuple
    """
    error = stats.t.ppf(1 - (0.05 / 2), len(vect) - 1) * (np.std(vect) / math.sqrt(len(vect)))
    mean = np.mean(vect)
    return mean, error
