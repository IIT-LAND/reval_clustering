---
title: '`reval`: a Python package to determine the best number of clusters with stability-based relative clustering 
        validation.'
tags:
- Python
- Machine learning
- Clustering
authors:
- name: Isotta Landi
  affiliation: 1
- name: Veronica Mandelli
  affiliation: "1, 2"
- name: Michael V. Lombardo
  affiliation: "1, 3"
affiliations:
- name: Laboratory for Autism and Neurodevelopmental Disorders, Center for Neuroscience and Cognitive Systems @UniTn, 
        Italian Institute of Technology, Rovereto, Italy
  index: 1
- name: Center for Mind/Brain Sciences, University of Trento, Corso Bettini 84, 38068 Rovereto (TN), Italy
  index: 2
- name: Autism Research Centre, Department of Psychiatry, University of Cambridge, Cambridge, United Kingdom
  index: 3
date: 24 June 2020
bibliography: paper.bib
---

# Summary

Clustering algorithms identify intrinsic subgroups in a dataset by arranging together elements that show smaller 
pairwise dissimilarities relative to other subgroups [@stat]. They are one of a number of machine learning methods that 
do unsupervised learning with the aim of identifying patterns in the data in the absence of supervised/external 
knowledge. The process of determining the number of clusters in a dataset is challenging because of the lack of a 
priori information and of a unique clustering validation approach to evaluate clustering solutions. Attempts to address 
this challenge leverage both relative and internal cluster validity methods. In particular, relative criteria aim to 
compare clustering solutions obtained with different parameter settings, whereas internal criteria focus on quantities 
and features inherent to a grouping solution [@vazirgiannis2009]. While a variety of software packages contain 
internal cluster validity methods and metrics, open-source software solutions to easily implement relative clustering 
techniques are lacking. Here we present the `reval` Python package (pronounced “reh-val”, like the word “revel”), 
that implements stability-based validation of clustering solutions to determine the number of clusters that best 
partitions the data, as described by Lange and colleagues [@lange2004]. 

Several methods help determine the best number of clusters based on internal criteria, e.g., the elbow method 
[@thorndike1953] selects the number of clusters for which the within-cluster variability decrement is minimum. 
Another popular method that leverages internal criteria is the silhouette-based approach [@rousseeuw1987], 
 which returns the number of clusters that maximizes 
cluster cohesion and separation - that is, how similar an object is to other elements of the same cluster compared to 
elements of other clusters. Many other statistics have been developed using internal criteria 
to suggest what the optimal number of clusters is, e.g., the NbClust library [@nbclust] compiles 30 different 
internal metrics and users have the option to compute all or a subset of these metrics and use a majority vote rule to 
select the optimal number of clusters.

In contrast to internal criteria, relative validation techniques transform the problem of selecting the best number of 
clusters into a model selection problem, whereby selection is guided by minimization of prediction error. First, a 
dataset is split into training and test sets and then independently partitioned into clusters. Second, training set 
labels are used within supervised classification methods to learn how to best predict the labels. Applying the 
classification model to the test set, the model’s predicted labels are then compared to the actual clustering labels 
derived from the test set. This procedure is repeated using cross validation and the optimal number of clusters 
corresponds to the number of clusters that minimizes the prediction error. Prediction performance can be defined in 
different ways. For example, Tibshirani and colleagues [@tibshirani2005] used prediction strength - that is, the 
proportion of observation pairs that are assigned to the same cluster in both training and test sets. Other approaches 
consider stability measures [@lange2004], which are based on misclassification error.

Internal and relative indices can exhibit similar behavior with respect to clustering errors, with the advantage of 
the former being less computationally expensive [@brun2007]. However, in the case of complex models and clusters, an 
approach based on minimization of prediction error is more indicative of clustering performance, because internal 
indices tend to fail to correlate well with errors [@brun2007]. Another advantage of the model selection approach 
within a cross-validation framework is that it allows for selection of the solution that best generalizes to new 
unseen data and allows for investigation of replicability of the clustering results in an automated fashion. On the 
contrary, internal indices are highly tied to the characteristics of the data at hand and thus can run the risk of 
overfitting. Hence, they are not well suited when the aim is to assess clustering generalizability and replicability.

The `reval` library has three modules:

- `relative_validation`: This module includes training and test methods that return the misclassification errors 
obtained by comparing classification labels, permuted according to the Kuhn-Munkres algorithm 
[@kuhn1955; @munkres1957], and the clustering labels. Within this module, the 
`relative_validation.RelativeValidation.rndlabel_traineval` method allows users to compute the asymptotic 
misclassification rate via random labeling.

- `best_nclust_cv`: This module implements the cross-validation procedure and returns the best number of clusters 
along with the normalized stability scores, obtained from the average of the misclassification scores divided by 
the asymptotic misclassification rate. The `best_nclust_cv.FindBestClustCV.evaluate` method applies the fitted 
model with the returned number of clusters to the held-out dataset.

- `visualization`: This module includes the `plot_metrics` function to plot cross-validated performance 
metrics with 95% confidence intervals for varying number of clustering solutions.

Method details can be found in [@lange2004], [code](https://github.com/IIT-LAND/reval_clustering) and 
[documentation](https://reval.readthedocs.io/en/latest/) are available.
An overview of the `reval` framework is reported in Figure \autoref{fig:framework}.

![`reval` framework overview. \label{fig:framework}](revalpipeline.png)

# Statement of needs

The `reval` package allows users the ability to apply unsupervised clustering techniques to their data and then 
offer a principled method for selecting the optimal number of clusters by embedding the clustering approach within a 
supervised classification framework. `reval` works with the *scikit-learn* Python library for machine learning. 
In particular, among clustering methods, because we need the `n_clusters` parameter, users can 
select `KMeans`, `SpectralClustering`, and `AgglomerativeClustering` from `sklearn.cluster`. Moreover, any classifier 
from *scikit-learn* can be selected (e.g., `KNeighborsClassifier` from `sklearn.neighbors`). `reval` returns the best 
number of clusters based on a cross-validation procedure and can also evaluate the model with the parameter selected 
on an held-out dataset, if available. The code has been optimized to speed up computations. However, it is worth 
acknowledging that at each cross-validation iteration we:

1) Apply two clustering algorithms, one to each fold;
2) Fit a classifier to the training set and evaluate it on the test set;
3) Normalize the stability measure with the stability computed from $N$ iterations of random labeling.

For these reasons, the computational cost tends to increase with dataset size (see Figure \autoref{fig:1} 
for an example of execution time performances).

![Execution time for `best_nclust_cv` module applied to synthetic data. 5 isotropic Gaussian blobs are simulated with 
varying number of samples and features (``sklearn.datasets.make_blobs``). Number of clusters ranges from 2 to 6. 
We report execution time in seconds for algorithm performance. 
\label{fig:1}](makeblobs_performance.png)

# Key references

Libraries and methods for the automated selection of the best number of clusters are available in both Python and R. 
However, they mainly focus on internal validation criteria (e.g., Python
`yellowbrick.cluster.KElbowVisualizer` [@yellowbrick:2018]; R `NbClust` [@nbclust]). In R, `clValid` [@clvalid] and 
`cStability`[^1] apply stability-based relative validation models directly comparing clustering solutions with 
leave-one-out cross validation and bootstrapping, respectively. Nevertheless, users cannot select the preferred 
classifier nor the desired number of folds for the *k*-fold cross-validation procedure. `clValid` also enables 
validation with internal measures, whereas `cStability` also includes a model-free approach, see [@haslbeck2016].

[^1]: https://CRAN.R-project.org/package=cstab

# Acknowledgments
This work was supported by an ERC Starting Grant (ERC-2017-STG; 755816) to MVL. 
The authors declare no competing financial interests.

# References

