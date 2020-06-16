from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import logging
from umap import UMAP
from working_examples.plot_outputs import plot_metrics


# MNIST dataset with 10 classes
mnist, label = fetch_openml('mnist_784', version=1, return_X_y=True)
transform = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)

classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 16],
                                s=classifier, c=clustering, nrand=100)

# Stratified subsets of 7000 elements for training set and ~1000 for test set
mnist_tr, mnist_ts, label_tr, label_ts = train_test_split(mnist, label,
                                                          train_size=0.20,
                                                          test_size=0.20,
                                                          random_state=42,
                                                          stratify=label)
# Dimensionality reduction with UMAP as pre-processing step
mnist_tr = transform.fit_transform(mnist_tr)
metrics, nbest, chk_dist = findbestclust.best_nclust(mnist_tr, label_tr)
plot_metrics(metrics, "Relative clustering validation performance on MNIST dataset")

mnist_ts = transform.transform(mnist_ts)
out = findbestclust.evaluate(mnist_tr, mnist_ts, nbest)

plt.scatter(mnist_tr[:, 0], mnist_tr[:, 1],
            c=label_tr.astype(int), s=0.1, cmap='Spectral')
plt.scatter(mnist_tr[:, 0], mnist_tr[:, 1],
            c=out.train_cllab, s=0.1, cmap='Spectral')

plt.scatter(mnist_ts[:, 0], mnist_ts[:, 1],
            c=label_ts.astype(int), s=0.1, cmap='Spectral')
plt.scatter(mnist_ts[:, 0], mnist_ts[:, 1],
            c=out.test_cllab, s=0.1, cmap='Spectral')

logging.info(f"Results accuracy (on test set): {1 - zero_one_loss(out.test_cllab, label_ts)}")
