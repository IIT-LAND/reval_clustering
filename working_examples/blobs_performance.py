import time
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def blobs_performance():
    """
    Function performing multiple iterations of reval on simulated 5-blob datasets
    with varying number of samples and features and 10 repetitions of 10-fold CVs.
    The function plots the performance (in seconds) of the algorithm for each
    parameter configuration.
    """
    feat = [10, 100, 500, 1000]
    samples = [100, 500, 1000, 1500, 2000]

    perftime = []
    for s in samples:
        perf = []
        for f in feat:
            start = time.time()
            data = make_blobs(s, f, 5, center_box=(-20, 20),
                              random_state=42)

            X_tr, X_ts, y_tr, y_ts = train_test_split(data[0],
                                                      data[1],
                                                      test_size=0.30,
                                                      random_state=42,
                                                      stratify=data[1])

            classifier = KNeighborsClassifier(n_neighbors=5)
            clustering = KMeans()

            findbestclust = FindBestClustCV(nfold=10,
                                            nclust_range=[2, 7],
                                            s=classifier,
                                            c=clustering,
                                            nrand=100)
            metrics, nbest, _ = findbestclust.best_nclust(X_tr, iter_cv=10, strat_vect=y_tr)
            tmp_time = time.time() - start
            perf.append(tmp_time)
            print(f'Feat {f}, samples {s}: N cluster {nbest}, time: {tmp_time}')
        perftime.append(perf)

    perftime = np.array(perftime)
    fig, ax = plt.subplots()
    ax.plot(samples, perftime[:, 0], label='10 features', linestyle='--', color='black')
    ax.plot(samples, perftime[:, 1], label='100 features', color='black')
    ax.plot(samples, perftime[:, 2], label='500 features', linestyle='-.', color='black')
    ax.plot(samples, perftime[:, 3], label='1000 features', linestyle=':', color='black')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Execution time (s)')
    ax.set_title("")
    ax.legend()
    plt.savefig('./performance_blobs.png', dpi=300)


if __name__ == "__main__":
    blobs_performance()
