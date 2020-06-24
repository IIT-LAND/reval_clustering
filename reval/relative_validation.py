import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import zero_one_loss


class RelativeValidation:
    """
    This class allows to perform the relative clustering validation procedure.
    A supervised algorithm is required to test cluster stability.
    Labels output from a clustering algorithm are used as true labels.

    :param s: initialized class for the supervised method
    :type s: class
    :param c: initialized class for clustering algorithm
    :type c: class
    :param nrand: number of iterations to normalize cluster stability
    :type nrand: int
    """

    def __init__(self, s, c, nrand):
        """
        Construct method
        """
        self.class_method = s
        self.clust_method = c
        self.nrand = nrand

    def train(self, train_data):
        """
        Method that performs training. It compares the clustering labels on training set
        (i.e., A(X) computed by :class:`reval.relative_validation.RelativeValidation.clust_method`) against
        the labels obtained from the classification algorithm
        (i.e., f(X), computed by :class:`reval.relative_validation.RelativeValidation.class_method`).
        It returns the misclassification error, the supervised model fitted to the data,
        and both clustering and classification labels.

        :param train_data: training dataset
        :type train_data: ndarray, (n_samples, n_features)
        :return: misclassification error, fitted supervised model object, clustering and classification labels
        :rtype: float, object, dictionary of ndarray (n_samples,)
        """

        clustlab_tr = self.clust_method.fit_predict(train_data)  # A_k(X)
        fitclass_tr = self.class_method.fit(train_data, clustlab_tr)
        classlab_tr = fitclass_tr.predict(train_data)
        misclass = zero_one_loss(clustlab_tr, classlab_tr)
        return misclass, fitclass_tr, clustlab_tr

    def test(self, test_data, fit_model):
        """
        Method that compares test set clustering labels (i.e., A(X'), computed by
        :class:`reval.relative_validation.RelativeValidation.clust_method`) against
        the (permuted) labels obtained through the classification algorithm fitted to the training set
        (i.e., f(X'), computed by
        :class:`reval.relative_validation.RelativeValidation.class_method`).
        It returns the misclassification error, together with
        both clustering and classification labels.

        :param test_data: test dataset
        :type test_data: ndarray, (n_samples, n_features)
        :param fit_model: fitted supervised model
        :type fit_model: class
        :return: misclassification error, clustering and classification labels
        :rtype: float, dictionary of ndarrays (n_samples,)
        """
        clustlab_ts = self.clust_method.fit_predict(test_data)  # A_k(X')
        classlab_ts = fit_model.predict(test_data)
        bestperm = _kuhn_munkres_algorithm(classlab_ts, clustlab_ts)  # array of integers
        misclass = zero_one_loss(classlab_ts, bestperm)
        return misclass, bestperm

    def rndlabels_traineval(self, train_data, test_data, train_labels, test_labels):
        """
        Method that performs random labeling on the training set
        (N times according to
        :class:`reval.relative_validation.RelativeValidation.nrand` instance attribute) and evaluates
        the fitted models on test set.

        :param train_data: training dataset
        :type train_data: ndarray, (n_samples, n_features)
        :param test_data: test dataset
        :type test_data: ndarray, (n_samples, n_features)
        :param train_labels: training set clustering labels
        :type train_labels: ndarray, (n_samples,)
        :param test_labels: test set clustering labels
        :type test_labels: ndarray, (n_samples,)
        :return: averaged misclassification error on the test set
        :rtype: float
        """
        np.random.seed(0)
        shuf_tr = [np.random.permutation(train_labels)
                   for _ in range(self.nrand)]
        misclass_ts = list(map(lambda x: self._rescale_score_(train_data, test_data, x, test_labels), shuf_tr))
        return np.mean(misclass_ts)

    def _rescale_score_(self, xtr, xts, randlabtr, labts):
        """
        Private method that computes the misclassification error when predicting test labels
        with classification model fitted on training set with random labels.

        :param xtr: training dataset
        :type xtr: ndarray, (n_samples, n_features)
        :param xts: test dataset
        :type xts: ndarray, (n_samples, n_features)
        :param randlabtr: random labels
        :type randlabtr: ndarray, (n_samples,)
        :param labts: test set labels
        :type labts: ndarray, (n_samples,)
        :return: misclassification error
        :rtype: float
        """
        self.class_method.fit(xtr, randlabtr)
        me_ts = zero_one_loss(labts, _kuhn_munkres_algorithm(labts, self.class_method.predict(xts)))
        return me_ts


def _kuhn_munkres_algorithm(true_lab, pred_lab):
    """
    Private function that implements the Kuhn-Munkres algorithm method. It selects the best label permutation of the
    classification output that minimizes the
    misclassification error when compared to the clustering labels.

    :param true_lab: clustering algorithm labels
    :type true_lab: ndarray, (n_samples,)
    :param pred_lab: classification algorithm labels
    :type pred_lab: ndarray, (n_samples,)
    :return: permuted labels that minimize the misclassification error
    :rtype: ndarray, (n_samples,)
    """
    if isinstance(true_lab, np.ndarray) and isinstance(pred_lab, np.ndarray):
        nclass, nobs = len(set(true_lab)), len(true_lab)
        wmat = np.zeros((nclass, nclass))
        for lab in range(nclass):
            for plab in range(lab, nclass):
                n_intersec = len(set(np.flatnonzero(true_lab == lab)).intersection(
                    set(np.flatnonzero(pred_lab == plab))))
                w = (nobs - n_intersec) / nobs
                if lab == plab:
                    wmat[lab, plab] = w
                else:
                    wmat[lab, plab] = w
                    n_intersec = len(set(np.flatnonzero(true_lab == plab)).intersection(
                        set(np.flatnonzero(pred_lab == lab))))
                    w = (nobs - n_intersec) / nobs
                    wmat[plab, lab] = w
        new_pred_lab = list(linear_sum_assignment(wmat)[1])
        try:
            pred_perm = np.array([new_pred_lab.index(i) for i in pred_lab])
        except ValueError:
            pred_perm = np.array([], dtype=int)
            for i in pred_lab:
                if len(new_pred_lab) <= i:
                    pred_perm = np.append(pred_perm, i)
                else:
                    pred_perm = np.append(pred_perm, new_pred_lab.index(i))
        return pred_perm
    else:
        raise TypeError(f'input variables should be (np.ndarray, np.ndarray)'
                        f' not ({type(true_lab)}, {type(pred_lab)})')
