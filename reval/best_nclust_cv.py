from sklearn.model_selection import StratifiedKFold, KFold
from reval.relative_validation import RelativeValidation
import numpy as np
import math


class FindBestClustCV(RelativeValidation):
    """
    Child class of RelativeValidation

    It performs cross validation on the training set to
    select the best number of clusters, i.e., the number that minimizes the
    misclassification error.

    Initialized with:
    S: inherited classification object
    C: inherited clustering object
    nfold: number of fold for CV (int)
    strat_vect: array, pd Series for stratification (strings, objects or integers)
                (n_samples, )
    n_clust: max number  of clusters to look for (int)
    min_cl: int, minimum number of clusters

    Methods
    -------
    search_best_clust()
    evaluate()
    """

    def __init__(self, nfold, nclust_range, s, c, nrand):
        super().__init__(s, c, nrand)
        self.nfold = nfold
        self.nclust_range = nclust_range

    def best_nclust(self, data, strat_vect=None):
        data_array = np.array(data)
        reval = RelativeValidation(self.class_method, self.clust_method, self.nrand)
        metrics = {'train': {}, 'val': {}}
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
                rndmisc_mean_tr, rndmisc_mean_val = reval.rndlabels_traineval(tr_set, val_set,
                                                                              tr_labels,
                                                                              val_labels)
                norm_stab_tr.append(miscl_tr / rndmisc_mean_tr)
                norm_stab_val.append(miscl_val / rndmisc_mean_val)
            metrics['train'][ncl] = (np.mean(norm_stab_tr), _confint(norm_stab_tr))
            metrics['val'][ncl] = (np.mean(norm_stab_val), _confint(norm_stab_val))
        val_score = np.array([val[0] for val in metrics['val'].values()])
        bestscore = min(val_score)
        # select the cluster with the minimum misclassification error
        # and the maximum number of clusters
        bestncl = max(np.transpose(np.argwhere(val_score == bestscore))[0]) + self.nclust_range[0]
        return metrics, bestncl


def _confint(vect):
    """
    Parameters
    ----------
    vect: list (of performance scores)
    Returns
    ------
    tuple : mean and error

    """
    error = 1.96 * (np.std(vect) / math.sqrt(len(vect)))
    mean = np.mean(vect)
    return mean, error
