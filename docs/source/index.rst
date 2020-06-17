.. Relative clustering validation documentation master file, created by
   sphinx-quickstart on Mon May 11 18:17:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stability-based relative clustering validation to determine the best number of cluster
======================================================================================

``reval`` allows to determine the best number of clusters to partition datasets without a priori knowledge.
It leverages a stability-based relative clustering validation method (Lange et al., 2004) that transforms
a clustering algorithm into a supervised classification problem and selects the number of clusters
that lead to the minimum expected misclassification error, i.e., stability.

This library allows to:

1. Select any classification algorithm, provided that ``fit()`` and ``transform()`` methods are available;
2. Select any clustering algorithm with ``n_clusters`` parameter;
3. Perform *k*-fold cross-validation to determine the best number of clusters;
4. Test the final model on an held-out dataset.

Underlying mathematics can be found in (Lange et al., 2004), whereas code can be found on `github
<https://github.com>`_.

Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004).
Stability-based validation of clustering solutions. *Neural computation*, 16(6), 1299-1323.

.. toctree::
   :maxdepth: 2
   :caption: User Guide / Tutorial

   installing
   code_usage
   experiments

.. toctree::
   :maxdepth: 2
   :caption: Code Guide

   code_description

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
