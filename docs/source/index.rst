.. Relative clustering validation documentation master file, created by
sphinx-quickstart on Mon May 11 18:17:16 2020.

Stability-based relative clustering validation to determine the best number of cluster
======================================================================================

``reval`` allows to determine the best clustering solution without a priori knowledge.
It leverages a stability-based relative clustering validation method (Lange et al., 2004) that transforms
a clustering algorithm into a supervised classification problem and selects the number of clusters
that leads to the minimum expected misclassification error, i.e., stability.

This library allows to:

1. Select any classification algorithm from ``sklearn`` library;
2. Select a clustering algorithm with ``n_clusters`` parameter or HDBSCAN density-based algorithm,
    i.e., choose among ``sklearn.cluster.KMeans``,
   ``sklearn.cluster.AgglomerativeClustering``, ``sklearn.cluster.SpectralClustering``, ``hdbscan.HDBSCAN``;
3. Perform (repeated) *k*-fold cross-validation to determine the best number of clusters;
4. Test the final model on an held-out dataset.

Theoretical background can be found in (Lange et al., 2004), whereas code can be found on `github
<https://github.com/IIT-LAND/reval_clustering>`__.

The analysis steps performed by ``reval`` package are displayed below.

.. image:: images/revalv0.0.2pipeline.png
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

Cite as
=======

.. parsed-literal::

   Landi I, Mandelli V, Lombardo MV.
   reval: A Python package to determine best clustering solutions with stability-based relative clustering validation.
   Patterns (N Y). 2021 Apr 2;2(4):100228.
   doi: 10.1016/j.patter.2021.100228. PMID: 33982023; PMCID: PMC8085609.

BibTeX alternative

.. parsed-literal::

   @article{landi2021100228,
            title = {reval: A Python package to determine best clustering solutions with stability-based relative clustering validation},
            journal = {Patterns},
            volume = {2},
            number = {4},
            pages = {100228},
            year = {2021},
            issn = {2666-3899},
            doi = {https://doi.org/10.1016/j.patter.2021.100228},
            url = {https://www.sciencedirect.com/science/article/pii/S2666389921000428},
            author = {Isotta Landi and Veronica Mandelli and Michael V. Lombardo},
            keywords = {stability-based relative validation, clustering, unsupervised learning, clustering replicability}
            }

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
