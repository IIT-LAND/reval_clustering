import timeit
import numpy as np

setup_code = """
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from reval.best_nclust_cv import FindBestClustCV

data = make_blobs(1000, 10, centers=2)
data_tr, data_ts, y_tr, y_ts = train_test_split(data[0],
                                                data[1],
                                                test_size=0.5, 
                                                stratify=data[1],
                                                random_state=42)
s = KNeighborsClassifier()
c = KMeans()

findbest = FindBestClustCV(s=s,
                           c=c,
                           nrand=10,
                           nfold=2,
                           n_jobs=1,
                           nclust_range=list(range(2, 7, 1)))
"""

main_code = """
findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
"""

main_code_evaluate = """
findbest.evaluate(data_tr, data_ts, nclust=2)
"""

print("Find best number of clusters execution time:")
out_best = timeit.repeat(stmt=main_code,
                         setup=setup_code,
                         repeat=100,
                         number=1)
print(f"Average time: {np.mean(out_best)} ({np.std(out_best)})")

print("Evaluate best clustering solution execution time:")
out_evaluate = timeit.repeat(stmt=main_code_evaluate,
                             setup=setup_code,
                             repeat=100,
                             number=1)
print(f"Average time: {np.mean(out_evaluate)} ({np.std(out_evaluate)})")

