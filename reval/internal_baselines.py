import numpy as np


def select_best(data, c, int_measure, select='max', nclust_range=None):
    """
    Select the best number of clusters that minimizes/maximizes
    the internal measure selected.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param select: it can be 'min', if the internal measure is to be minimized
        or 'max' if the internal measure should be macimized.
    :type select: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :return: internal score and best number of clusters.
    :rtype: float, int
    """
    if nclust_range is not None:
        scores = []
        label_vect = []
        for nc in nclust_range:
            c.n_clusters = nc
            label = c.fit_predict(data)
            scores.append(int_measure(data, label))
            label_vect.append(label)
    else:
        label = c.fit_predict(data)
        best = int_measure(data, label)
        return best, len([lab for lab in np.unique(label) if lab >= 0]), label
    if select == 'max':
        best = np.where(np.array(scores) == max(scores))[0]
    elif select == 'min':
        best = np.where(np.array(scores) == min(scores))[0]
    if len(set(label_vect[int(max(best))])) == nclust_range[int(max(best))]:
        return scores[int(max(best))], nclust_range[int(max(best))], label_vect[int(max(best))]
    else:
        return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def evaluate_best(data, c, int_measure, ncl=None):
    """
    Function that, given a number of clusters, returns the corresponding internal measure
    for a dataset.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param ncl: number of clusters.
    :type ncl: int
    :return: internal score.
    :rtype: float
    """
    if 'n_clusters' in c.get_params().keys():
        c.n_clusters = ncl
    label = c.fit_predict(data)
    return int_measure(data, label)
