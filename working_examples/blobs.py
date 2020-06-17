from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import zero_one_loss
import time
import logging

# Generate 1,000 samples for 5 blobs
classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()
for n in [3000]:
    for feat in [500]:
        start = time.time()
        data = make_blobs(n, feat, 5, random_state=42)
        # Split them into training and test set (30% of data)
        X_tr, X_ts, y_tr, y_ts = train_test_split(data[0], data[1],
                                                  test_size=0.30, random_state=42,
                                                  stratify=data[1])

        # Apply relative clustering validation with KNN and Hierarchical clustering

        findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 7],
                                        s=classifier, c=clustering, nrand=100)
        metrics, nbest, chk_dist = findbestclust.best_nclust(X_tr, y_tr)
        out = findbestclust.evaluate(X_tr, X_ts, nbest)

        # logging.info(f"Results accuracy (on test set): {1 - zero_one_loss(out.test_cllab, y_ts)}")
        print(f'Execution time with {n} samples and {feat} feat: {time.time() - start}.')

# # Create a noisy dataset with 6 clusters
# data_noisy = make_blobs(1000, 100, 6, random_state=42, cluster_std=5)
# Xnoise_tr, Xnoise_ts, ynoise_tr, ynoise_ts = train_test_split(data_noisy[0], data_noisy[1],
#                                                               test_size=0.30, random_state=42,
#                                                               stratify=data_noisy[1])
#
# metrics_noise, nbest_noise, chk_dist_noise = findbestclust.best_nclust(Xnoise_tr, ynoise_tr)
# out_noise = findbestclust.evaluate(Xnoise_tr, Xnoise_ts, nbest_noise)
#
# logging.info(f"Results accuracy (on test set): {1 - zero_one_loss(out_noise.test_cllab, ynoise_ts)}")
