from sklearn.datasets import make_blobs, load_digits
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score
from reval.visualization import plot_metrics
import matplotlib.pyplot as plt
from reval.utils import kuhn_munkres_algorithm
from umap import UMAP

# EXAMPLE 1: Isotropic Gaussian blobs
data = make_blobs(1000, 2, 5, center_box=(-20, 20),
                  random_state=42)
plt.figure(figsize=(6, 4))
plt.scatter(data[0][:, 0],
            data[0][:, 1],
            c=data[1], cmap='rainbow_r')
plt.title("Blobs dataset (N=1000)")
plt.show()

X_tr, X_ts, y_tr, y_ts = train_test_split(data[0],
                                          data[1],
                                          test_size=0.30,
                                          random_state=42,
                                          stratify=data[1])

classifier = KNeighborsClassifier(n_neighbors=5)
clustering = AgglomerativeClustering(affinity='euclidean',
                                     linkage='ward')

findbestclust = FindBestClustCV(nfold=10,
                                nclust_range=[2, 7],
                                s=classifier,
                                c=clustering,
                                nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

perm_lab = kuhn_munkres_algorithm(y_ts, out.test_cllab)

print(f"Best number of clusters: {nbest}")
print(f"Test set prediction ACC: "
      f"{1 - zero_one_loss(y_ts, perm_lab)}")
print(f'AMI (true labels vs predicted labels) = '
      f'{adjusted_mutual_info_score(y_ts, out.test_cllab)}')
print(f"Validation set normalized stability (misclassification):"
      f"{metrics['val'][nbest]}")
print(f'Test set ACC = {out.test_acc} '
      f'(true labels vs predicted labels)')

plot_metrics(metrics, title="Reval performance blobs dataset",
             legend_loc=2)

plt.figure(figsize=(6, 4))
plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=y_ts, cmap='rainbow_r')
plt.title("Test set true labels (blobs dataset)")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=perm_lab, cmap='rainbow_r')
plt.title("Test set clustering labels (blobs dataset)")
plt.show()

# EXAMPLE 2: Handwritten digits dataset example
digits_dataset = load_digits()

digits_data = digits_dataset['data']
digits_target = digits_dataset['target']

X_tr, X_ts, y_tr, y_ts = train_test_split(digits_data,
                                          digits_target,
                                          test_size=0.40,
                                          random_state=42,
                                          stratify=digits_target)

transform = UMAP(n_components=2,
                 random_state=42,
                 n_neighbors=30,
                 min_dist=0.0)
X_tr = transform.fit_transform(X_tr)
X_ts = transform.transform(X_ts)

s = KNeighborsClassifier(n_neighbors=30)
c = AgglomerativeClustering()

reval = FindBestClustCV(s=s,
                        c=c,
                        nfold=5,
                        nclust_range=[2, 15],
                        nrand=100)

metrics, nclustbest, _ = reval.best_nclust(X_tr, y_tr)

plot_metrics(metrics, title='Reval performance digits dataset')

out = reval.evaluate(X_tr, X_ts, nclust=nclustbest)
perm_lab = kuhn_munkres_algorithm(y_ts, out.test_cllab)

print(f"Best number of clusters: {nclustbest}")
print(f"Test set prediction ACC: "
      f"{1 - zero_one_loss(y_ts, perm_lab)}")
print(f'AMI (true labels vs predicted labels) = '
      f'{adjusted_mutual_info_score(y_ts, out.test_cllab)}')
print(f"Validation set normalized stability (misclassification):"
      f"{metrics['val'][nclustbest]}")
print(f'Test set ACC = {out.test_acc} '
      f'(true labels vs predicted labels)')

plt.figure(figsize=(6, 4))
plt.scatter(X_ts[:, 0],
            X_ts[:, 1],
            c=y_ts, cmap='rainbow_r')
plt.title("Test set true labels (digits dataset)")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(X_ts[:, 0],
            X_ts[:, 1],
            c=perm_lab, cmap='rainbow_r')
plt.title("Test set clustering labels (digits dataset)")
plt.show()
