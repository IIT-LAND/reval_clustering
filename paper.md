---
title: '`reval`: determine the best number of clusters with stability-based relative clustering validation.'

tags:
- Python
- Machine learning
- Clustering

authors: 
- name: Isotta Landi
- affiliation: 1
- name: Michael V. Lombardo
- affiliation: 1

affiliations:
- name: LAND laboratory, Italian Institute of Technology
  index: 1

date: 15 June 2020

bibliography: paper.bib

---

# Summary

`reval` Python library implements stability-based validation of clustering solutions to determine the number of
 clusters that best partitions the data, as described by Lange and colleagues [@lange2004]. 
 Clustering algorithms identify intrinsic subgroups in a dataset arranging together elements that show smaller 
 pairwise dissimilarities with respect to other subgroups [@stat]. These methods are comprised in 
 unsupervised learning methods, which refer to a set of machine learning methods that aim 
 at identifying patterns in the data in the absence of previous knowledge.
 
 Because of the lack of a priori information, and in the absence of a unique clustering validation approach 
 that evaluates a clustering solution, the process of determining the number of clusters in a dataset 
 is a challenging task. Attempts to solve this leverage both relative and internal cluster validity methods. 
 In particular, relative criteria aim at comparing clustering solutions with different parameters, whereas 
 internal criteria focus on quantities and features inherent of a grouping solution [@vazirgiannis2009].  
 
 Methods to determine the number of clusters based on internal criteria are, for example, the elbow method that 
 focuses on within-cluster variability and selects the number of clusters for which the dispersion decrement 
 is minimum. Tree-cut technique for hierarchical clustering cuts the dendrogram obtained merging groups of elements 
 consecutively at a fixed height. Silhouette-based approach maximizes clusters cohesion and separation, 
 i.e., how similar an object is to other elements of the same cluster compared to elements of other clusters. 
 Differently, relative validation techniques consider the estimation of the number of clusters as a model selection 
 problem via the minimization of prediction error. A dataset is thus split into training and test sets that are 
 independently partitioned into clusters. The performance of a classification algorithm in predicting labels 
 on the test set is evaluated with varying numbers of clusters via cross-validation. Finally, the number of clusters 
 that corresponds to the minimum classification error is selected. Prediction performance can be defined in different 
 ways, e.g., Tibshirani and colleagues [@tibshirani2005] used prediction strenght, i.e., the proportion of observation 
 pairs that are assigned to the same cluster in both training and test sets. Other approaches consider stability 
 measures [@lange2004], which are based on misclassification error.
 
 In general, internal indices have similar behaviors to relative indices with respect to clustering errors, with the 
 advantage of being less computationally expensive [@brun2007]. However, in the case of complex models and clusters,
 an approach based on prediction measures is more reliable, because internal indices tend to fail to correlate well
 with errors [@brun2007]. Furthermore, a model selection approach within a cross-validation framework allows to select 
 the solution that best generalizes to new data and allows to investigate replicability of the results in an 
 automated fashion. On the contrary, internal indices are highly tied to the characteristics of the data at hand, 
 by construction. Hence, they fail to assess clustering generalizability and replicability.
 
 `reval` library has three modules:
 - `relative_validation`: it includes the training and test methods that return the misclassification errors 
 obtained comparing classification labels, permuted according the Kuhn-Munkres algorithm [@kuhn1955; @munkres1957], 
 and the clustering labels. The `relative_validation.RelativeValidation.rndlabel_traineval` method computes 
 the normalized stability score via random labeling;
 - `best_nclust_cv`: it implements the cross-validation procedure and returns the best number of clusters along with 
 the normalized stability scores. The `nest_nclust_cv.FindBestClustCV.evaluate` method applies the fitted model with 
 the returned number of clusters to the hold-out dataset.
 - `visualization`: module that includes functions to plot cross-validated performance metrics with 95% confidence 
 interval and scatter plots displaying the clustering solution (it first requires the application of a 
 dimensionality reduction algorithm). 
 
 Method details can be found in [@lange2004], code and documentation with working performance examples can be found 
 [here](https://github.com/landiisotta/relative_validation_clustering).
  
# Statement of needs

`reval` package enables stability-based relative clustering validation to select the best number of clusters
 for a selected clustering algorithm. It needs as input: 1) classification algorithm 
 (with `fit()` and `transform()` methods); 2) clustering algorithm (with `n_clusters` parameter);
 3) dataset already partitioned into training and test sets. At each cross-validation iteration, two clustering 
 algorithms are applied, to both training and test sets, and a classifier is fitted on the training set and then
 evaluated on the test set. The stability measure obtained at each iteration is then normalized for the stability 
 obtained from $100$ iterations of the algorithm with random labels. For this reason the
 computational cost tends to increase with dataset size (see \autoref{fig:1} for an example of
 execution time performances).
 
 | Number of samples      | Number of features | Execution time (s) |
 | ---------------------- | ------------------ | ------------------ |
 | $1000$                 | $10$               | $29.50$            |
 | $1000$                 | $100$              | $55.85$            |
 | $1000$                 | $1000$             | $387.55$           |
 | $5000$                 | $10$               | $158.94$           |
 | $5000$                 | $100$              | $613.63$           |
 | $5000$                 | $1000$             | $$           |
 | $10000$                | $10$               | $$           |
 | $10000$                | $100$              | $$           |
 | $10000$                | $1000$             | $$           |
 

 ![`best_nclust_cv` module applied to simulation data with 5 blobs and varying number of samples and 
  features. Execution time in seconds is reported for algorithm performance. \label{fig:1}](figure_perf.png)
 
# Key references
 Libraries and methods for the automated selection of the best number of clusters are 
 available in both Python and R. However, they mainly focus on internal validation criteria (e.g., Python 
 `yellowbrick.cluster.KElbowVisualizer` [@yellowbrick:2018]; R `NbClust` [@nbclust]). In R, `clValid` [@clvalid] and 
 `cStability`[^1] apply stability-based relative validation models (`clValid` also enables validation with internal 
 measures, whereas `cStability` includes a model-free approach, see [@haslbeck2016]) with leave-one-out 
 cross-validation and bootstrapping, respectively. Users cannot select the preferred classifier nor the desired 
 _k_ parameter for the _k_-fold cross-validation procedure.
 
 [^1]: https://CRAN.R-project.org/package=cstab
 
 # Acknowledgments
 Grant?
 
 # References