from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering


RNDLABELS_ITER = 100
CLASSIFIER = KNeighborsClassifier(n_neighbors=15)
CLUSTERING = AgglomerativeClustering(n_clusters=2)