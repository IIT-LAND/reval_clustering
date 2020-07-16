from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score
from reval.visualization import plot_metrics
import matplotlib.pyplot as plt
from reval.utils import kuhn_munkres_algorithm

# Generate 1,000 samples for 5 blobs
# ----------------------------------
data = make_blobs(1000, 2, 5, random_state=42)
plt.scatter(data[0][:, 0],
            data[0][:, 1],
            c=data[1], cmap='rainbow_r')

# Split them into training and test set (30% of data)
X_tr, X_ts, y_tr, y_ts = train_test_split(data[0],
                                          data[1],
                                          test_size=0.30,
                                          random_state=42,
                                          stratify=data[1])

# Apply relative clustering validation with KNN and Hierarchical clustering
classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

findbestclust = FindBestClustCV(nfold=10,
                                nclust_range=[2, 7],
                                s=classifier,
                                c=clustering,
                                nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

perm_lab = kuhn_munkres_algorithm(y_ts, out.test_cllab)

print(f"Best number of clusters: {nbest}")
print(f"Test set external ACC: "
      f"{1 - zero_one_loss(y_ts, perm_lab)}")
print(f'AMI = {adjusted_mutual_info_score(y_ts, out.test_cllab)}')
print(f"Validation set normalized stability (misclassification): {metrics['val'][nbest]}")
print(f'Test set ACC = {out.test_acc}')

plot_metrics(metrics, title="Reval performance")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=y_ts, cmap='rainbow_r')
plt.title("True labels for test set")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=perm_lab, cmap='rainbow_r')
plt.title("Clustering labels for test set")

# Create a noisy dataset with 5 clusters
# ----------------------------------------
data_noisy = make_blobs(1000, 10, 5, random_state=42, cluster_std=3)
plt.scatter(data_noisy[0][:, 0],
            data_noisy[0][:, 1],
            c=data_noisy[1],
            cmap='rainbow_r')

Xnoise_tr, Xnoise_ts, ynoise_tr, ynoise_ts = train_test_split(data_noisy[0],
                                                              data_noisy[1],
                                                              test_size=0.30,
                                                              random_state=42,
                                                              stratify=data_noisy[1])

metrics_noise, nbest_noise, _ = findbestclust.best_nclust(Xnoise_tr, ynoise_tr)
out_noise = findbestclust.evaluate(Xnoise_tr, Xnoise_ts, nbest_noise)

plot_metrics(metrics_noise, title="Reval performance")

perm_lab_noise = kuhn_munkres_algorithm(ynoise_ts, out_noise.test_cllab)

print(f"Best number of clusters: {nbest_noise}")
print(f"Test set external ACC: "
      f"{1 - zero_one_loss(ynoise_ts, perm_lab_noise)}")
print(f'AMI = {adjusted_mutual_info_score(ynoise_ts, out_noise.test_cllab)}')
print(f"Validation set normalized stability (misclassification): {metrics_noise['val'][nbest_noise]}")
print(f"Result accuracy (on test set): "
      f"{out_noise.test_acc}")

plt.scatter(Xnoise_ts[:, 0], Xnoise_ts[:, 1],
            c=ynoise_ts, cmap='rainbow_r')
plt.title("True labels")

plt.scatter(Xnoise_ts[:, 0], Xnoise_ts[:, 1],
            c=perm_lab_noise, cmap='rainbow_r')
plt.title("Clustering labels for test set")

# Pre-processing with UMAP
from umap import UMAP

transform = UMAP(n_components=10, n_neighbors=30, min_dist=0.0)

Xtr_umap = transform.fit_transform(Xnoise_tr)
Xts_umap = transform.transform(Xnoise_ts)

plt.scatter(Xtr_umap[:, 0], Xtr_umap[:, 1],
            c=ynoise_tr, cmap='rainbow_r')
plt.title("UMAP-transformed training set with true labels")

plt.scatter(Xts_umap[:, 0], Xts_umap[:, 1],
            c=ynoise_ts, cmap='rainbow_r')
plt.title("UMAP-transformed test set with true labels")

metrics, nbest, _ = findbestclust.best_nclust(Xtr_umap, ynoise_tr)
out = findbestclust.evaluate(Xtr_umap, Xts_umap, nbest)

plot_metrics(metrics, title='Reval performance of UMAP-transformed dataset')

perm_noise = kuhn_munkres_algorithm(ynoise_ts, out.test_cllab)

print(f"Best number of clusters: {nbest}")
print(f"Test set external ACC: "
      f"{1 - zero_one_loss(ynoise_ts, perm_noise)}")
print(f'AMI = {adjusted_mutual_info_score(ynoise_ts, out.test_cllab)}')
print(f"Validation set normalized stability (misclassification): {metrics['val'][nbest]}")
print(f"Result accuracy (on test set): "
      f"{out.test_acc}")

plt.scatter(Xts_umap[:, 0], Xts_umap[:, 1],
            c=perm_noise, cmap='rainbow_r')
plt.title("Predicted labels for UMAP-preprocessed test set")
