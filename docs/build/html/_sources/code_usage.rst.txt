How to Use ``reval``
--------------------

In the following, we are going to simulate N = 1,000 samples of dataset with two groups and two features
(for visualization purposes), then we will show how to apply the ``reval`` package and investigate
the result types. We will use hierarchical clustering and KNN classification algorithms.

First, let us import a bunch of useful libraries and our class ``reval.best_nclust_cv.FindBestClustCV``:

.. code:: python3

    from reval.best_nclust_cv import FindBestClustCV
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

Then we simulate the toy dataset and visualize it:

.. code:: python3

    data = make_blobs(1000, 2, 2, random_state=42)
    plt.scatter(data[0][:, 0], data[0][:, 1],
                c=data[1], cmap='rainbow_r')
    plt.show()

.. image:: images/useblobs.png
   :align: center

Then, we split the dataset into training and test sets:

.. code:: python3

    X_tr, X_ts, y_tr, y_ts = train_test_split(data[0], data[1],
                                              test_size=0.30,
                                              random_state=42,
                                              stratify=data[1])

We apply the stability-based relative clustering validation approach with 10-fold cross-validation,
100 iterations of random labeling, and number of clusters ranging from 2 to 10.

.. code:: python3

    classifier = KNeighborsClassifier()
    clustering = AgglomerativeClustering()
    findbestclust = FindBestClustCV(nfold=10,
                                    nclust_range=[2, 11],
                                    s=classifier,
                                    c=clustering,
                                    nrand=100)
    metrics, nbest, chk_dist = findbestclust.best_nclust(X_tr, y_tr)
    out = findbestclust.evaluate(X_tr, X_ts, nbest)

To obtain the training stability and the normalized validation stability for the
selected number of clusters we need to call:

.. code:: python3

    nbest
    # 2
    metrics['train'][nbest]
    # (0.0, (0.0, 0.0)) (stab, CI)
    metrics['val'][nbest]
    # (0.0, (0.0, 0.0)) (stab, CI)

In ``chk_dist`` we have access to the misclassification errors during
cross-validation. ``out`` returns train/test accuracies and test set clustering labels.

.. code:: python3

    out.train_cllab
    # array([0, 1, 0, 1, 0, 0, 1...
    out.test_cllab
    # array([0, 0, 0, 0, 1...
    out.train_acc
    # 1.0
    out.test_acc
    # 1.0

To visualize performance metrics:

.. code:: python3

    from reval.visualization import plot_metrics
    plot_metrics(metrics, title="Reval metrics")

.. image:: images/performanceexample.png
    :align: center














