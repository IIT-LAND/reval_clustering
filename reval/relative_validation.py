from reval import config
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import zero_one_loss


class RelativeValidation:
    """
    This class is initialized with the supervised algorithm to test cluster stability and
    the clustering algorithm whose labels are used as true labels.

    Methods:
        train() -- it compares the clustering labels on training set (i.e., A(X)) against
        the labels obtained through the classification algorithm (i.e., f(X)).
        It returns the misclassification error and the supervised model fitted to the data.

        test() -- it compares the clustering labels on the test set (i.e., A(X')) against
        the (permuted) labels obtained through the classification algorithm fitted to the training set
        (i.e., f(X')).
        It return the misclassification error.

    """

    def __init__(self, s, c):
        self.class_method = s
        self.clust_method = c

    def train(self, train_data):
        """
        Parameters
        ----------
        train_data: array with the training data
                    (n_samples, n_features)
        Returns
        -------
        misclass: float
        fit_model: fit object
        labels: dictionary of clustering and classification labels
                (array of integers (n_samples,))
        """

        clustlab_tr = self.clust_method.fit_predict(train_data)  # A_k(X)
        fitclass_tr = self.class_method.fit(train_data, clustlab_tr)
        classlab_tr = fitclass_tr.predict(train_data)

        misclass = zero_one_loss(clustlab_tr, classlab_tr)
        labels = {'classification': classlab_tr,
                  'clustering': clustlab_tr}
        return misclass, fitclass_tr, labels

    def test(self, test_data, fit_model):
        """
        Parameters
        ----------
        test_data: array with test data
                   (n_samples, n_features)
        fit_model: object, fitted model

        Returns
        -------
        misclass: float
        labels: dictionary of clustering and classification labels
                (array of integers (n_samples, ))
        """
        clustlab_ts = self.clust_method.fit_predict(test_data)  # A_k(X')
        classlab_ts = fit_model.predict(test_data)
        bestperm = _kuhn_munkres_algorithm(clustlab_ts, classlab_ts)  # array of integers
        misclass = zero_one_loss(clustlab_ts, bestperm)
        labels = {'classification': classlab_ts,
                  'clustering': clustlab_ts}

        return misclass, labels

    def rndlabels_train(self, train_data, train_labels):
        """"
        Method performing random labeling training
        N times
        Parameters
        ----------
        train_data: numpy array
        train_labels: numpy array
        Returns
        -------
        float
        list
            list of fitted classification models
        """
        np.random.seed(0)
        shuf_tr = [list(map(lambda x: np.random.permutation(x),
                            train_labels)) for _ in range(config.RNDLABELS_ITER)]
        model_tr = [self.class_method.fit(train_data, lab) for lab in shuf_tr]
        misclass_tr = [zero_one_loss(x, y.predict(train_data)) for x, y in
                       zip(shuf_tr, model_tr)]
        return np.mean(misclass_tr), model_tr

    @staticmethod
    def rndlabels_test(test_data, test_labels, model_list):
        """"
           Method performing random labeling test
           N times
           Parameters
           ----------
           test_data: numpy array
           test_labels: numpy array (no need to shuffle)
           model_list: list of fitted classification models
           Returns
           -------
           float
        """
        misclass_ts = [zero_one_loss(test_labels,
                                     _kuhn_munkres_algorithm(test_labels,
                                                             mod.predict(test_data))) for mod in model_list]
        return np.mean(misclass_ts)


def _kuhn_munkres_algorithm(true_lab, pred_lab):
    """
    Implementation of the Hungarian method that selects the best label permutation that minimizes the
    misclassification error
    Parameters
    ----------
    true_lab: array as output from the clustering algorithm (n_samples, )
    pred_lab: array as output from the classification algorithm (n_samples, )

    Returns
    -------
    pred_perm: array of permuted labels (n_samples, )
    """
    if isinstance(true_lab, np.ndarray) and isinstance(pred_lab, np.ndarray):
        nclass, nobs = len(set(true_lab)), len(true_lab)
        wmat = np.zeros((nclass, nclass))
        for lab in range(nclass):
            for plab in range(lab, nclass):
                n_intersec = len(set(np.transpose(np.argwhere(true_lab == lab))[0]).intersection(
                    set(np.transpose(np.argwhere(pred_lab == plab))[0])))
                w = (nobs - n_intersec) / nobs
                if lab == plab:
                    wmat[lab, plab] = w
                else:
                    wmat[lab, plab] = w
                    n_intersec = len(set(np.transpose(np.argwhere(true_lab == plab))[0]).intersection(
                        set(np.transpose(np.argwhere(pred_lab == lab))[0])))
                    w = (nobs - n_intersec) / nobs
                    wmat[plab, lab] = w
        new_pred_lab = list(linear_sum_assignment(wmat)[1])
        pred_perm = np.array([new_pred_lab.index(i) for i in pred_lab])
        return pred_perm
    else:
        raise TypeError(f'input variables should be (np.ndarray, np.ndarray)'
                        f' not ({type(true_lab)}, {type(pred_lab)})')
