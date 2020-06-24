.. Relative clustering validation documentation master file, created by
   sphinx-quickstart on Mon May 11 18:17:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stability-based relative clustering validation to determine the best number of cluster
======================================================================================

``reval`` allows to determine the best number of clusters to partition datasets without a priori knowledge.
It leverages a stability-based relative clustering validation method (Lange et al., 2004) that transforms
a clustering algorithm into a supervised classification problem and selects the number of clusters
that leads to the minimum expected misclassification error, i.e., stability.

This library allows to:

1. Select any classification algorithm from ``sklearn`` library;
2. Select a clustering algorithm with ``n_clusters`` parameter, i.e., choose among ``sklearn.cluster.KMeans``,
   ``sklearn.cluster.AgglomerativeClustering``, ``sklearn.cluster.SpectralClustering``;
3. Perform *k*-fold cross-validation to determine the best number of clusters;
4. Test the final model on an held-out dataset.

Underlying mathematics can be found in (Lange et al., 2004), whereas code can be found on `github
<https://github.com/IIT-LAND/reval_clustering>`__.

The analysis steps performed by ``reval`` package are displayed below.

.. image:: images/revalpipeline.png
   :align: center

Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004).
Stability-based validation of clustering solutions. *Neural computation*, 16(6), 1299-1323.

.. toctree::
   :maxdepth: 2
   :caption: User guide / Tutorial

   installing
   code_usage
   experiments
   datadimension

.. toctree::
   :maxdepth: 2
   :caption: Code guide

   code_description

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
