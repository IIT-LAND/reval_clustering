from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from umap import UMAP
from reval.visualization import plot_metrics
from reval.relative_validation import _kuhn_munkres_algorithm

# MNIST dataset with 10 classes
mnist, label = fetch_openml('mnist_784', version=1, return_X_y=True)
transform = UMAP(n_neighbors=30, min_dist=0.0, n_components=10, random_state=42)

# Stratified subsets of 7000 elements for both training and test set
mnist_tr, mnist_ts, label_tr, label_ts = train_test_split(mnist, label,
                                                          train_size=0.1,
                                                          test_size=0.1,
                                                          random_state=42,
                                                          stratify=label)

# Dimensionality reduction with UMAP as pre-processing step
mnist_tr = transform.fit_transform(mnist_tr)
mnist_ts = transform.transform(mnist_ts)

plt.scatter(mnist_tr[:, 0],
            mnist_tr[:, 1],
            c=label_tr.astype(int),
            s=0.1,
            cmap='rainbow_r')
plt.title('UMAP-transformed training subsample of MNIST dataset (N=7,000)')

plt.scatter(mnist_ts[:, 0], mnist_ts[:, 1],
            c=label_ts.astype(int), s=0.1, cmap='rainbow_r')
plt.title('UMAP-transformed test subsample of MNIST dataset (N=7,000)')

# Run relative clustering validation
classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 12],
                                s=classifier, c=clustering, nrand=100)

metrics, nbest, _ = findbestclust.best_nclust(mnist_tr, label_tr)
out = findbestclust.evaluate(mnist_tr, mnist_ts, nbest)

plot_metrics(metrics, "Relative clustering validation performance on MNIST dataset")

perm_lab = _kuhn_munkres_algorithm(label_ts.astype(int), out.test_cllab)

plt.scatter(mnist_ts[:, 0], mnist_ts[:, 1],
            c=perm_lab, s=0.1, cmap='rainbow_r')
plt.title("Predicted labels for MNIST test set")

print(f"Best number of clusters: {nbest}")
print(f"Test set external ACC: "
      f"{1 - zero_one_loss(label_ts.astype(int), perm_lab)}")
print(f'AMI = {adjusted_mutual_info_score(label_ts.astype(int), perm_lab)}')
print(f"Validation set normalized stability (misclassification): {metrics['val'][nbest]}")
print(f"Result accuracy (on test set): "
      f"{out.test_acc}")
