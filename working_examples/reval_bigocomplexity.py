import cProfile
import pstats
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from reval.best_nclust_cv import FindBestClustCV

s = KNeighborsClassifier()
# c = KMeans()
c = AgglomerativeClustering()

findbest = FindBestClustCV(s=s,
                           c=c,
                           nrand=10,
                           nfold=2,
                           n_jobs=1,
                           nclust_range=list(range(2, 7, 1)))

data = make_blobs(100, 10, centers=2)
data_tr, data_ts, y_tr, y_ts = train_test_split(data[0],
                                                data[1],
                                                test_size=0.5,
                                                stratify=data[1],
                                                random_state=42)

print("Profiling of algorithm that finds the best number of clusters.")
profiler = cProfile.Profile()
profiler.enable()
findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
stats.print_stats()

print("Profiling of algorithm that evaluates the best number of clusters.")
profiler.enable()
findbest.evaluate(data_tr, data_ts, nclust=2)
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
stats.print_stats()
