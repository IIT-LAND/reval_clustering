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

   Isotta Landi, Veronica Mandelli, & Michael Vincent Lombardo. (2020, June 29). reval: stability-based relative
   clustering validation method to determine the best number of clusters (Version v1.0.0). Zenodo.
   http://doi.org/10.5281/zenodo.3922334

BibTeX alternative

.. parsed-literal::

   @software{isotta_landi_2020_3922334,
              author       = {Isotta Landi and
                              Veronica Mandelli and
                              Michael Vincent Lombardo},
              title        = {{reval: stability-based relative clustering
                               validation method to determine the best number of
                               clusters}},
              month        = jun,
              year         = 2020,
              publisher    = {Zenodo},
              version      = {v1.0.0},
              doi          = {10.5281/zenodo.3922334},
              url          = {https://doi.org/10.5281/zenodo.3922334}
            }

Pre-print manuscript

.. parsed-literal::

    @misc{l2020reval,
          title={reval: a Python package to determine the best number of clusters with stability-based relative clustering validation},
          author={Isotta Landi and Veronica Mandelli and Michael V. Lombardo},
          year={2020},
          eprint={2009.01077},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
