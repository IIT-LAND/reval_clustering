---
title: '`reval`: a Python package to determine the best number of clusters with stability-based relative clustering 
validation.'
tags:
- Python
- Machine learning
- Clustering
authors:
- name: Isotta Landi
- affiliation: 1
- name: Veronica Mandelli
- affiliation: 1, 2
- name: Michael V. Lombardo
- affiliation: 1, 3
affiliations:
- name: Laboratory for Autism and Neurodevelopmental Disorders, Center for Neuroscience and Cognitive Systems @UniTn, 
Istituto Italiano di Tecnologia, Rovereto, Italy
 index: 1
- name: Center for Mind/Brain Sciences, University of Trento, Rovereto, Italy
 index: 2
- name: Autism Research Centre, Department of Psychiatry, University of Cambridge, Cambridge, United Kingdom
 index: 3
date: 15 June 2020
bibliography: paper.bib
---

# Summary

Clustering algorithms identify intrinsic subgroups in a dataset arranging together elements that show smaller pairwise 
dissimilarities with respect to other subgroups [@stat]. They are one of a number of machine learning methods that do 
unsupervised learning with the aim of identifying patterns in the data in the absence of supervised/external knowledge. 
The process of determining the number of clusters in a dataset is a challenging task, because of the lack of a priori 
information and without a unique clustering validation approach to evaluate clustering solutions. Attempts to address 
this task leverage both relative and internal cluster validity methods. In particular, relative criteria aim at 
comparing clustering solutions obtained with different parameter settings, whereas internal criteria focus on 
quantities and features inherent to a grouping solution [@vazirgiannis2009]. While a variety of software packages 
contain internal cluster validity methods and metrics, open-source software solutions to easily implement relative 
clustering techniques are lacking. Here we present the `reval` Python library (pronounced “reh-val”, like the word 
“revel”), that implements stability-based validation of clustering solutions to determine the number of clusters that 
best partitions the data, as described by Lange and colleagues [@lange2004]. 

Several methods exist to determine the number of clusters based on internal criteria. For example, the elbow method 
[@thorndike1953] selects the number of clusters for which the within-cluster variability decrement is minimum. Another 
popular method using internal criteria is the silhouette-based approach [@rousseeuw1987]. This method maximizes cluster 
cohesion and separation, i.e., how similar an object is to other elements of the same cluster compared to elements of 
other clusters. Silhouette is one of many other statistics that have been developed using internal criteria to suggest 
what the optimal number of clusters is. NbClust package [@nbclust] compiles 30 different metrics and users have the 
option to compute all or a subset of these metrics and use a majority vote rule to select the optimal number of 
clusters.

In contrast to internal criteria, relative validation techniques transform the problem of selecting the best number of 
clusters into a model selection problem which minimizes a prediction error. A dataset is first split into training and 
test sets and then independently partitioned into clusters. Second, training set labels are used within supervised 
classification methods to learn how to best predict labels for the test set. These predicted labels are then compared 
to the actual clustering labels identified from the test set. This procedure is repeated using cross-validation and the 
optimal number of clusters corresponds to the number of clusters that minimizes the prediction error chosen. Prediction 
performance can be defined in different ways. For example, Tibshirani and colleagues [@tibshirani2005] used prediction 
strength - that is, the proportion of observation pairs that are assigned to the same cluster in both training and test 
sets. Other approaches consider stability measures [@lange2004], which are based on misclassification error.

Internal and relative indices can exhibit similar behavior with respect to clustering errors, with the advantage of the 
former being less computationally expensive [@brun2007]. However, in the case of complex models and clusters, an 
approach based on prediction error is more indicative of clustering performance, because internal indices tend to fail 
to correlate well with errors [@brun2007]. Another advantage of the model selection approach within a cross-validation 
framework is that it allows for selection of the solution that best generalizes to new data and to investigate 
replicability of the results in an automated fashion. On the contrary, internal indices are highly tied to the 
characteristics of the data at hand. Hence, they are not suited to assess clustering generalizability and 
replicability.

The `reval` library has three modules:
- `relative_validation`: This module includes training and test methods that return the misclassification errors 
obtained by comparing classification labels, permuted according to the Kuhn-Munkres algorithm 
[@kuhn1955; @munkres1957], and the clustering labels. Within this module, the 
`relative_validation.RelativeValidation.rndlabel_traineval` method allows users to compute the asymptotic 
misclassification rate via random labeling.
- `best_nclust_cv`: This module implements the cross-validation procedure and returns the best number of clusters along 
with the normalized stability scores, obtained from the average of the misclassification scores divided by the 
asymptotic misclassification rate. The `nest_nclust_cv.FindBestClustCV.evaluate` method applies the fitted model with 
the returned number of clusters to the held-out dataset.
- `visualization`: This module includes functions to plot cross-validated performance metrics with 95% confidence 
intervals for varying number of cluster solutions and scatterplots that display the clustering solution. Note that 
scatterplots first require the application of a dimensionality reduction algorithm to the dataset 
(e.g., PCA, UMAP, t-SNE).
Method details can be found in [@lange2004], code and documentation with working examples can be found 
[here](https://github.com/landiisotta/relative_validation_clustering).

# Statement of needs

The `reval` package allows users the ability to apply unsupervised clustering techniques to their data and then offer a 
principled method for selecting the optimal number of clusters by embedding the clustering approach within a supervised 
classification framework. `reval` works with *scikit-learn* Python library for machine learning. In particular, among 
clustering methods, because it requires the `n_clusters` parameter, users can select `KMeans`, `SpectralClustering`, 
and `AgglomerativeClustering` from `sklearn.cluster`. Moreover, all classifiers can be selected (e.g., 
`KNeighborsClassifier` from `sklearn.neighbors`). `reval` returns the best number of clusters based on 
cross-validation procedure and it can also evaluate the model with the parameter selected on an held-out dataset, 
if available. The code has been optimized to speed up computations. However, it is worth acknowledging that at 
each cross-validation iteration we:
1) apply two clustering algorithms, one to each fold;
2) fit a classifier to the training set and evaluate it on the test set;
3) normalize the stability measure after estimating the stability from $N$ iterations of random labeling.
For these reasons, the computational cost tends to increase with dataset size (see \autoref{fig:1} for an example of 
execution time performances).

![`best_nclust_cv` module applied to simulation blob data with 5 clusters and varying number of samples and features. 
Number of clusters ranges from 2 to 6. We report execution time in seconds for algorithm performance. 
\label{fig:1}](makeblobs_performance.png)

# Key references

Libraries and methods for the automated selection of the best number of clusters are available in both Python and R. 
However, they mainly focus on internal validation criteria (e.g., Python
`yellowbrick.cluster.KElbowVisualizer` [@yellowbrick:2018]; R `NbClust` [@nbclust]). In R, `clValid` [@clvalid] and 
`cStability`[^1] apply stability-based relative validation models (`clValid` also enables validation with internal 
measures, whereas `cStability` also includes a model-free approach, see [@haslbeck2016]) with leave-one-out 
cross-validation and bootstrapping, respectively. Nevertheless, users cannot select the preferred classifier nor the 
desired number of folds for the *k*-fold cross-validation procedure.

[^1]: https://CRAN.R-project.org/package=cstab

# Acknowledgments
This work was supported by an ERC Starting Grant (ERC-2017-STG; 755816) to MVL. The authors declare no competing 
financial interests.

# References
