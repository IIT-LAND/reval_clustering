import sys

sys.path.append('../../reval_clustering/')

from reval.best_nclust_cv import FindBestClustCV
from reval.internal_baselines import select_best, evaluate_best
from reval.visualization import plot_metrics
from reval.utils import kuhn_munkres_algorithm, compute_metrics
from reval.param_selection import SCParamSelection
from datasets.manuscript_builddatasets import build_ucidatasets

from sklearn.datasets import make_blobs, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import hdbscan
from umap import UMAP
import numpy as np

import warnings
import logging
import matplotlib.pyplot as plt

# Modify this variable for parallelization
N_JOBS = 7

warnings.filterwarnings('ignore')

"""
Three example functions that can also be run from shell (see manuscript_examples.py file). 

Example 1: blobs dataset;
Example 2: hand-written digits dataset from UCI repository;
Example 3: real-world dataset MNIST;
Example 4: best clussifier-clustering combinations for 18 datasets from UCI Machine Learning Repository.
"""


# EXAMPLE 1: Isotropic Gaussian blobs
def example1():
    # Generate dataset
    data = make_blobs(1000, 2, centers=5,
                      center_box=(-20, 20),
                      random_state=42)

    # Visualize dataset
    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.scatter(data[0][data[1] == i][:, 0],
                    data[0][data[1] == i][:, 1],
                    label=i, cmap='tab20')
    plt.title("Blobs dataset")
    # plt.savefig('./blobs.png', format='png')
    plt.show()

    # Create training and test sets
    X_tr, X_ts, y_tr, y_ts = train_test_split(data[0],
                                              data[1],
                                              test_size=0.30,
                                              random_state=42,
                                              stratify=data[1])

    # Initialize clustering and classifier
    classifier = KNeighborsClassifier(n_neighbors=15)
    clustering = KMeans()

    # Run relatve validation (repeated CV and testing)
    findbestclust = FindBestClustCV(nfold=2,
                                    nclust_range=list(range(2, 7, 1)),
                                    s=classifier,
                                    c=clustering,
                                    nrand=10,
                                    n_jobs=N_JOBS)
    metrics, nbest = findbestclust.best_nclust(X_tr, iter_cv=10, strat_vect=y_tr)
    out = findbestclust.evaluate(X_tr, X_ts, nclust=nbest)

    # Plot CV metrics
    plot_metrics(metrics, prob_lines=False)
    logging.info(f"Validation stability: {metrics['val'][nbest]}")
    perm_lab = kuhn_munkres_algorithm(y_ts, out.test_cllab)

    logging.info(f"Best number of clusters: {nbest}")
    logging.info(f'AMI (true labels vs predicted labels) for test set = '
                 f'{adjusted_mutual_info_score(y_ts, out.test_cllab)}')
    logging.info('\n\n')

    # Compute metrics
    logging.info("Metrics from true label comparisons on test set:")
    class_scores = compute_metrics(y_ts, perm_lab, perm=False)
    for k, val in class_scores.items():
        if k in ['F1', 'MCC']:
            logging.info(f"{k}, {val}")
    logging.info("\n\n")

    # Internal measures
    # SILHOUETTE
    logging.info("Silhouette score based selection")
    sil_score_tr, sil_best_tr, sil_label_tr = select_best(X_tr, clustering, silhouette_score,
                                                          select='max',
                                                          nclust_range=list(range(2, 7, 1)))
    sil_score_ts, sil_best_ts, sil_label_ts = select_best(X_ts, clustering, silhouette_score,
                                                          select='max',
                                                          nclust_range=list(range(2, 7, 1)))

    sil_eval = evaluate_best(X_ts, clustering, silhouette_score, sil_best_tr)

    logging.info(f"Best number of clusters (and scores) for tr/ts independent runs: "
                 f"{sil_best_tr}({sil_score_tr})/{sil_best_ts}({sil_score_ts})")
    logging.info(f"Test set evaluation {sil_eval}")
    logging.info(f'AMI (true labels vs clustering labels) training = '
                 f'{adjusted_mutual_info_score(y_tr, kuhn_munkres_algorithm(y_tr, sil_label_tr))}')
    logging.info(f'AMI (true labels vs clustering labels) test = '
                 f'{adjusted_mutual_info_score(y_ts, kuhn_munkres_algorithm(y_ts, sil_label_ts))}')
    logging.info('\n\n')

    # DAVIES-BOULDIN
    logging.info("Davies-Bouldin score based selection")
    db_score_tr, db_best_tr, db_label_tr = select_best(X_tr, clustering, davies_bouldin_score,
                                                       select='min', nclust_range=list(range(2, 7, 1)))
    db_score_ts, db_best_ts, db_label_ts = select_best(X_ts, clustering, davies_bouldin_score,
                                                       select='min', nclust_range=list(range(2, 7, 1)))

    db_eval = evaluate_best(X_ts, clustering, davies_bouldin_score, db_best_tr)

    logging.info(f"Best number of clusters (and scores) for tr/ts independent runs: "
                 f"{db_best_tr}({db_score_tr})/{db_best_ts}({db_score_ts})")
    logging.info(f"Test set evaluation {db_eval}")
    logging.info(f'AMI (true labels vs clustering labels) training = '
                 f'{adjusted_mutual_info_score(y_tr, kuhn_munkres_algorithm(y_tr, db_label_tr))}')
    logging.info(f'AMI (true labels vs clustering labels) test = '
                 f'{adjusted_mutual_info_score(y_ts, kuhn_munkres_algorithm(y_ts, db_label_ts))}')
    logging.info('\n\n')

    # Plot true vs predicted labels for test sets
    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.scatter(X_ts[y_ts == i][:, 0],
                    X_ts[y_ts == i][:, 1],
                    label=str(i),
                    cmap='tab20')
    plt.legend(loc=3)
    plt.title("Test set true labels")
    # plt.savefig('./blobs_true.png', format='png')
    plt.show()

    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.scatter(X_ts[perm_lab == i][:, 0],
                    X_ts[perm_lab == i][:, 1],
                    label=str(i),
                    cmap='tab20')
    plt.legend(loc=3)
    plt.title("Test set clustering labels")
    # plt.savefig('./blobs_clustering.png', format='png')
    plt.show()


# Example 2: MNIST dataset
def example2():
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(int)

    X_tr, y_tr = mnist['data'][:60000], mnist.target[:60000]
    X_ts, y_ts = mnist['data'][60000::], mnist.target[60000::]
    transform = UMAP(n_components=2,
                     random_state=42,
                     n_neighbors=30,
                     min_dist=0.0)
    X_tr = transform.fit_transform(X_tr)
    X_ts = transform.transform(X_ts)

    s = KNeighborsClassifier(n_neighbors=30)
    c = hdbscan.HDBSCAN(min_samples=10,
                        min_cluster_size=200)

    reval = FindBestClustCV(s=s,
                            c=c,
                            nfold=2,
                            nrand=10,
                            n_jobs=N_JOBS)

    metrics, nclustbest, tr_lab = reval.best_nclust(X_tr, iter_cv=10, strat_vect=y_tr)

    plot_metrics(metrics)

    out = reval.evaluate(X_tr, X_ts, nclust=nclustbest, tr_lab=tr_lab)
    perm_lab = kuhn_munkres_algorithm(y_ts, out.test_cllab)
    logging.info(f"Validation stability: {metrics['val'][nclustbest]}")

    logging.info(f"Best number of clusters during CV: {nclustbest}")
    logging.info(f"Best number of clusters on test set: "
                 f"{len([lab for lab in np.unique(out.test_cllab) if lab >= 0])}")
    logging.info(f'AMI (true labels vs predicted labels) = '
                 f'{adjusted_mutual_info_score(y_ts, out.test_cllab)}')
    logging.info('\n\n')

    logging.info("Metrics from true label comparisons on test set:")
    class_scores = compute_metrics(y_ts, perm_lab)
    for k, val in class_scores.items():
        logging.info(f'{k}, {val}')
    logging.info('\n\n')

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tr[:, 0],
                         X_tr[:, 1],
                         c=y_tr, cmap='rainbow_r',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Train set true labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tr[:, 0],
                         X_tr[:, 1],
                         c=kuhn_munkres_algorithm(y_tr, tr_lab),
                         cmap='tab20',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Train set predicted labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_ts[:, 0],
                         X_ts[:, 1],
                         c=y_ts, cmap='tab20',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Test set true labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_ts[:, 0],
                         X_ts[:, 1],
                         s=0.1,
                         c=perm_lab, cmap='tab20')
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Test set clustering labels (digits dataset)")
    plt.show()

    # Internal measures
    # SILHOUETTE
    logging.info("Silhouette score based selection")
    sil_score_tr, sil_best_tr, sil_label_tr = select_best(X_tr, c, silhouette_score, select='max')
    sil_score_ts, sil_best_ts, sil_label_ts = select_best(X_ts, c, silhouette_score, select='max')
    logging.info(
        f"Best number of clusters (and scores) for tr/ts independent runs: "
        f"{sil_best_tr}({sil_score_tr})/{sil_best_ts}({sil_score_ts})")
    logging.info(f'AMI (true labels vs clustering labels) training = '
                 f'{adjusted_mutual_info_score(y_tr, kuhn_munkres_algorithm(y_tr, sil_label_tr))}')
    logging.info(f'AMI (true labels vs clustering labels) test = '
                 f'{adjusted_mutual_info_score(y_ts, kuhn_munkres_algorithm(y_ts, sil_label_ts))}')
    logging.info('\n\n')

    # DAVIES-BOULDIN
    logging.info("Davies-Bouldin score based selection")
    db_score_tr, db_best_tr, db_label_tr = select_best(X_tr, c, davies_bouldin_score,
                                                       select='min')
    db_score_ts, db_best_ts, db_label_ts = select_best(X_ts, c, davies_bouldin_score,
                                                       select='min')

    logging.info(
        f"Best number of clusters (and scores) for tr/ts independent runs: "
        f"{db_best_tr}({db_score_tr})/{db_best_ts}({db_score_ts})")
    logging.info(f'AMI (true labels vs clustering labels) training = '
                 f'{adjusted_mutual_info_score(y_tr, kuhn_munkres_algorithm(y_tr, db_label_tr))}')
    logging.info(f'AMI (true labels vs clustering labels) test = '
                 f'{adjusted_mutual_info_score(y_ts, kuhn_munkres_algorithm(y_ts, db_label_ts))}')
    logging.info('\n\n')

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tr[:, 0],
                         X_tr[:, 1],
                         c=sil_label_tr, cmap='tab20',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Train set silhouette labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_ts[:, 0],
                         X_ts[:, 1],
                         c=sil_label_ts, cmap='tab20',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Test set silhouette labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tr[:, 0],
                         X_tr[:, 1],
                         c=db_label_tr, cmap='tab20',
                         s=0.1)
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Train set Davies-Bouldin labels (digits dataset)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_ts[:, 0],
                         X_ts[:, 1],
                         s=0.1,
                         c=db_label_ts, cmap='tab20')
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    plt.title("Test set Davies-Bouldin labels (digits dataset)")
    plt.show()


# Example 3: best classifier/clustering combination for UCI dataset
def example3(n_jobs, preprocess=None):
    """
    :param preprocess: it can be 'scaled',
        'umap', 'scaled+umap', default None for raw processing.
    :type preprocess: str
    :return:
    """
    # Example 4: best clussifier/clustering for UCI dataset

    # Classifiers
    s = [LogisticRegression(solver='liblinear',
                            random_state=42),
         RandomForestClassifier(n_estimators=100,
                                random_state=42),
         KNeighborsClassifier(n_neighbors=1,
                              metric='euclidean'),
         SVC(C=1,
             random_state=42)]

    # Clustering
    c = [AgglomerativeClustering(),
         KMeans(random_state=42),
         hdbscan.HDBSCAN()]

    scparam = {'s': s,
               'c': c}

    transform = UMAP(n_neighbors=30, min_dist=0.0, random_state=42)
    scale = StandardScaler()

    # Import benchmark datasets
    uci_data = build_ucidatasets()
    # Run ensemble learning algorithm
    best_results = {}
    for data, name in zip(uci_data, uci_data._fields):
        scparam['s'][-1].gamma = (1 / data['data'].shape[0])
        nclass = len(np.unique(data['target']))
        logging.info(f"Processing dataset {name}")
        logging.info(f"True number of classes: {nclass}\n")
        X_tr, X_ts, y_tr, y_ts = train_test_split(data['data'],
                                                  data['target'],
                                                  test_size=0.40,
                                                  random_state=42,
                                                  stratify=data['target'])
        if preprocess == 'umap+scaled':
            X_tr = transform.fit_transform(scale.fit_transform(X_tr))
        elif preprocess == 'umap':
            X_tr = transform.fit_transform(X_tr)
        elif preprocess == 'scaled':
            X_tr = scale.fit_transform(X_tr)

        scparam_select = SCParamSelection(sc_params=scparam,
                                          cv=2,
                                          nrand=10,
                                          clust_range=list(range(2, nclass + 3, 1)),
                                          n_jobs=n_jobs,
                                          iter_cv=10,
                                          strat=y_tr)
        scparam_select.fit(X_tr, nclass=nclass)
        best_results[name] = scparam_select.best_param_
        # Uncomment to save the results
        #     pkl.dump(best_results, open('./best_resultUCI_scaledumap.pkl', 'wb'))
        logging.info('*' * 100)
        logging.info('\n\n')


if __name__ == "__main__":
    # Format logging output and save
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO,
                        filename='example_out.log',
                        filemode='w')
    example1()
    example2()
    example3(n_jobs=7, preprocess='scaled+umap')
