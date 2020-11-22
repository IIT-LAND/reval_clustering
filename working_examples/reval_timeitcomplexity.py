import time
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from reval.best_nclust_cv import FindBestClustCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import itertools
from hdbscan import HDBSCAN
import pickle as pkl
import matplotlib.pyplot as plt


def time_cmplx(n_jobs=1):
    data = make_blobs(100, 10, centers=2)
    data_tr, data_ts, y_tr, y_ts = train_test_split(data[0],
                                                    data[1],
                                                    test_size=0.5,
                                                    stratify=data[1],
                                                    random_state=42)
    s = [KNeighborsClassifier(), SVC(), LogisticRegression(), RandomForestClassifier()]
    c = [HDBSCAN(), AgglomerativeClustering(), KMeans(), SpectralClustering()]
    param = itertools.product(s, c)

    labels = ['KNN'] * 4 + ['SVM'] * 4 + ["LR"] * 4 + ['RF'] * 4
    time_cv = {'LR': [],
               'KNN': [],
               'RF': [],
               'SVM': []}
    time_ev = {'LR': [],
               'KNN': [],
               'RF': [],
               'SVM': []}
    for idx, mod in enumerate(param):
        classifier, clustering = mod[0], mod[1]
        findbest = FindBestClustCV(s=classifier,
                                   c=clustering,
                                   nrand=10,
                                   nfold=2,
                                   n_jobs=n_jobs,
                                   nclust_range=list(range(2, 7, 1)))
        if isinstance(clustering, HDBSCAN):
            start = time.time()
            _, _, tr_lab = findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
            time_cv[labels[idx]].append(time.time() - start)

            start = time.time()
            findbest.evaluate(data_tr, data_ts, nclust=2, tr_lab=tr_lab)
            time_ev[labels[idx]].append(time.time() - start)
        else:
            start = time.time()
            _, _ = findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
            time_cv[labels[idx]].append(time.time() - start)

            start = time.time()
            findbest.evaluate(data_tr, data_ts, nclust=2)
            time_ev[labels[idx]].append(time.time() - start)

    pkl.dump(time_cv, open(f'time_cv_njobs{n_jobs}.pkl', 'wb'))
    pkl.dump(time_ev, open(f'time_ev_njobs{n_jobs}.pkl', 'wb'))

    clustering = KMeans()
    classifier = KNeighborsClassifier()
    time_knnkmeans = {10: [],
                      100: [],
                      1000: []}
    for nsamples, nfeatures in itertools.product([100, 500, 1000, 1500, 2000, 2500, 3000],
                                                 [10, 100, 1000]):
        data = make_blobs(nsamples, nfeatures, centers=2)
        data_tr, data_ts, y_tr, y_ts = train_test_split(data[0],
                                                        data[1],
                                                        test_size=0.5,
                                                        stratify=data[1],
                                                        random_state=42)
        findbest = FindBestClustCV(s=classifier,
                                   c=clustering,
                                   nrand=10,
                                   nfold=2,
                                   n_jobs=n_jobs,
                                   nclust_range=list(range(2, 7, 1)))
        start = time.time()
        _, _ = findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
        findbest.evaluate(data_tr, data_ts, nclust=2)
        time_knnkmeans[nfeatures].append(time.time() - start)

    pkl.dump(time_knnkmeans, open(f'time_knnkmeans{n_jobs}.pkl', 'wb'))


def plot(file, save_name=None):
    time_dict = pkl.load(open(file, 'rb'))
    style = ['-', '-.', '--', ':']
    fig, ax = plt.subplots(figsize=(5, 5))
    cl_list = ['hdbscan', 'hierarchical', 'k-means', 'spectral']
    i = 0
    for s, t in time_dict.items():
        ax.plot(cl_list,
                t,
                linewidth=2,
                linestyle=style[i],
                label=s,
                color='black')
        i += 1
    ax.legend(fontsize=10, loc=2)
    plt.xticks(cl_list, fontsize=10)
    plt.yticks(fontsize=10)
    # plt.xlabel('Clustering algorithms', fontsize=10)
    plt.ylabel('Execution time (s)', fontsize=10, labelpad=15)
    if save_name is not None:
        plt.savefig(f'./{save_name}.pdf', format='pdf')
    else:
        plt.show()


def plot_one(file, save_name=None):
    time_one = pkl.load(open(file, 'rb'))
    style = ['-', '-.', '--']
    fig, ax = plt.subplots(figsize=(5, 5))
    cl_list = [100, 500, 1000, 1500, 2000, 2500, 3000]
    i = 0
    for s, t in time_one.items():
        ax.plot(cl_list,
                t,
                linewidth=2,
                linestyle=style[i],
                label=f'{s} features',
                color='black')
        i += 1
    ax.legend(fontsize=10, loc=2)
    plt.xticks(cl_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Number of samples', fontsize=10, labelpad=8)
    # plt.ylabel('Execution time (s)', fontsize=10)
    if save_name is not None:
        plt.savefig(f'./{save_name}.pdf', format='pdf')
    else:
        plt.show()


#
#
#
# setup = """
#
#
#
# s = KNeighborsClassifier()
# c = AgglomerativeClustering()
#
#
# # time_dict = {}
# # for classifier, clustering in param:
# findbest = FindBestClustCV(s=s,
#                            c=c,
#                            nrand=10,
#                            nfold=2,
#                            n_jobs=7,
#                            nclust_range=list(range(2, 7, 1)))
# """
#
# statement_1 = """
# # train_time, evalu_time = [], []
# # for it in range(100):
# # train_start = time.time()
# findbest.best_nclust(data_tr, iter_cv=10, strat_vect=y_tr)
# # train_time.append(time.time() - train_start)
# """
#
# statement_2 = """
# # evalu_start = time.time()
# findbest.evaluate(data_tr, data_ts, nclust=2)
# # evalu_time.append(time.time() - evalu_start)
# # time_dict[(classifier, clustering)] = [(np.mean(train_time), np.std(train_time)),
# #                                        (np.mean(evalu_time), np.std(evalu_time))]
#
# # for k, v in time_dict.items():
# #     print(f"S/C: {k} -- Execution time (tr/ts): {v}")
# """

if __name__ == "__main__":
    # time_cmplx(n_jobs=1)
    plot('time_cv_njobs1.pkl', save_name='time_cv_njobs1')
    plot('time_ev_njobs1.pkl', save_name='time_ev_njobs1')
    plot_one('time_knnkmeans1.pkl', save_name='time_knnkmeans_njobs1')

    # time_cmplx(n_jobs=7)
    plot('time_cv_njobs7.pkl', save_name='time_cv_njobs7')
    plot('time_ev_njobs7.pkl', save_name='time_ev_njobs7')
    plot_one('time_knnkmeans7.pkl', save_name='time_knnkmeans_njobs7')

# out_ev = timeit.repeat(stmt=statement_2, setup=setup, repeat=1)
# print(np.mean(out_cv), np.std(out_cv))
# print(np.mean(out_ev, np.std(out_ev)))

# time_cv = {'LR': ([2.59, 7.69, 8.60, 8.96], [0.10, 0.36, 0.66, 0.32]),
#            'KNN': ([2.05, 2.44, 3.22, 3.39], [0.19, 0.13, 0.20, 0.16]),
#            'RF': ([9.47, 30.63, 30.73, 41.49], [0.13, 1.43, 1.03, 6.09]),
#            'SVM': ([1.78, 2.00, 2.80, 2.92], [0.03, 0.06, 0.17, 0.07])}
# time_ev = {'LR': ([0.009, 0.006, 0.02, 0.05], [0.002, 0.0008, 0.003, 0.003]),
#            'KNN': ([0.008, 0.005, 0.02, 0.05], [0.0009, 0.0007, 0.002, 0.003]),
#            'RF': ([0.13, 0.1, 0.11, 0.19], [0.008, 0.006, 0.003, 0.02]),
#            'SVM': ([0.004, 0.002, 0.02, 0.04], [0.0005, 0.0003, 0.002, 0.002])}
# time_s = time_ev
# style = ['-', '-.', '--', ':']
# fig, ax = plt.subplots(figsize=(5, 5))
# cl_list = ['hdbscan', 'hierarchical', 'k-means', 'spectral']
# i = 0
# for s, time in time_s.items():
#     ax.plot(cl_list,
#             time[0],
#             linewidth=2,
#             linestyle=style[i],
#             label=s,
#             color='black')
#     i += 1
#     ax.errorbar(cl_list,
#                 time[0],
#                 [1.96 * (tval / 10) for tval in time[1]],
#                 linewidth=1,
#                 linestyle='-',
#                 label='',
#                 color='black')
# ax.legend(fontsize=10, loc=2)
# plt.xticks(cl_list, fontsize=10)
# plt.yticks(fontsize=10)
# plt.xlabel('Clustering algorithms', fontsize=10)
# plt.ylabel('Execution time (s)', fontsize=10)
# # if save_fig is not None:
# #     plt.savefig(f'./{save_fig}', format='png')
# # else:
# plt.show()
