from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from reval.visualization import plot_metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, zero_one_loss
from reval.relative_validation import _kuhn_munkres_algorithm
import matplotlib.pyplot as plt
import numpy as np

# NUMBER OF FEATURES
# ------------------
# The first example illustrates how results can be influenced by the number of dataset features.
# Increasing the number of features can fix the problem.

data1 = make_blobs(1000, 10, 5, cluster_std=5, random_state=42)

# Plot synthetic dataset
plt.scatter(data1[0][:, 0], data1[0][:, 1],
            c=data1[1], cmap='rainbow_r')
plt.title('True labels for 10-feature dataset')

X_tr, X_ts, y_tr, y_ts = train_test_split(data1[0],
                                          data1[1],
                                          test_size=0.30, random_state=42,
                                          stratify=data1[1])
# Apply relative clustering validation with KNN and Hierarchical clustering
classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 7],
                                s=classifier, c=clustering, nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

plot_metrics(metrics, "Reval performance for synthetic dataset with 10 features")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=out.test_cllab, cmap='rainbow_r')
plt.title("Predicted labels for 10-feature dataset")

# Compare Reval solution to true labels
print(f'AMI test set = {adjusted_mutual_info_score(y_ts, out.test_cllab)}')
relabeling = _kuhn_munkres_algorithm(y_ts, out.test_cllab)
print(f'ACC test set = {1 - zero_one_loss(y_ts, relabeling)}')

# Increase the number of features from 10 to 20
data2 = make_blobs(1000, 20, 5, cluster_std=5, random_state=42)

# Plot synthetic dataset
plt.scatter(data2[0][:, 0],
            data2[0][:, 1],
            c=data2[1],
            cmap='rainbow_r')
plt.title('True labels for 20-feature dataset')

X_tr, X_ts, y_tr, y_ts = train_test_split(data2[0],
                                          data2[1],
                                          test_size=0.30, random_state=42,
                                          stratify=data2[1])

findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 7],
                                s=classifier, c=clustering, nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

plot_metrics(metrics, "Reval performance for synthetic dataset with 20 features")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=out.test_cllab, cmap='rainbow_r')
plt.title("Predicted labels for 20-feature dataset")

# Compare predicted labels with true labels (Adjusted Mutual Information, accuracy)
print(f'AMI test set = {adjusted_mutual_info_score(y_ts, out.test_cllab)}')
relabeling = _kuhn_munkres_algorithm(y_ts, out.test_cllab)
print(f'ACC test set = {1 - zero_one_loss(y_ts, relabeling)}')

# NUMBER OF SAMPLES
# -----------------
np.random.seed(42)

# We generate three random samples from normal distributions
data1 = np.random.normal(-5, size=(100, 2))
data2 = np.random.normal(12, 2.5, size=(50, 2))
data3 = np.random.normal(6, 2.5, size=(50, 2))
data = np.append(data1, data2, axis=0)
data = np.append(data, data3, axis=0)

label = [0] * 100 + [1] * 50 + [2] * 50

plt.scatter(data[:, 0], data[:, 1],
            c=label, cmap='rainbow_r')
plt.title('Random samples from normal distribution Ns=(100, 50, 50)')

classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

X_tr, X_ts, y_tr, y_ts = train_test_split(data, label,
                                          test_size=0.30,
                                          random_state=42,
                                          stratify=label)

# Apply relative clustering validation with KNN and Hierarchical clustering
findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 7],
                                s=classifier, c=clustering, nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)
plot_metrics(metrics, "Reval performance for synthetic dataset with Ns=(100, 50, 50)")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=_kuhn_munkres_algorithm(np.array(y_ts),
                                      out.test_cllab),
            cmap='rainbow_r')
plt.title(f'Predicted labels for classes with Ns=(100, 50, 50)')

# We now increase the number of samples in groups 2 and 3 to 500
data1 = np.random.normal(-5, size=(100, 2))
data2 = np.random.normal(12, 2.5, size=(500, 2))
data3 = np.random.normal(6, 2.5, size=(500, 2))
data = np.append(data1, data2, axis=0)
data = np.append(data, data3, axis=0)

label = [0] * 100 + [1] * 500 + [2] * 500

plt.scatter(data[:, 0], data[:, 1],
            c=label, cmap='rainbow_r')
plt.title('Random samples from normal distribution Ns=(100, 500, 500)')

classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()

X_tr, X_ts, y_tr, y_ts = train_test_split(data, label,
                                          test_size=0.30,
                                          random_state=42,
                                          stratify=label)

# Apply relative clustering validation with KNN and Hierarchical clustering
findbestclust = FindBestClustCV(nfold=10, nclust_range=[2, 7],
                                s=classifier, c=clustering, nrand=100)
metrics, nbest, _ = findbestclust.best_nclust(X_tr, y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)
plot_metrics(metrics, "Reval performance for synthetic dataset with Ns=(100, 500, 500)")

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=y_ts,
            cmap='rainbow_r')
plt.title(f'Test set true labels for classes with Ns=(100, 500, 500)')

plt.scatter(X_ts[:, 0], X_ts[:, 1],
            c=_kuhn_munkres_algorithm(np.array(y_ts),
                                      out.test_cllab),
            cmap='rainbow_r')
plt.title(f'Predicted labels for classes with Ns=(100, 500, 500)')

# Performance scores
# Test set ACC
print(f'Test set external '
      f'ACC = {1 - zero_one_loss(y_ts, _kuhn_munkres_algorithm(np.array(y_ts), out.test_cllab))}')
print(f'AMI = {adjusted_mutual_info_score(y_ts, out.test_cllab)}')
print(f"Validation stability metrics: {metrics['val'][nbest]}")
print(f"Test set ACC = {out.test_acc}")
