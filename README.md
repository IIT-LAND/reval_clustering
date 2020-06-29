# `reval`: stability-based relative clustering validation method to determine the best number of clusters

Determining the number of clusters that best partitions a dataset can be a challenging task because of 1) the lack of a 
priori information within an unsupervised learning framework; and 2) the absence of a unique clustering validation 
approach to evaluate clustering solutions. Here we present `reval`: a Python package that leverages 
stability-based relative clustering validation methods to determine best clustering solutions, as described in [1]. 
    
Statistical software, both in R and Python, usually compute internal validation metrics that can be leveraged
to select the number of clusters that best fit the data and open-source software solutions that easily implement 
relative clustering techniques are lacking. The advantage of a relative approach over internal validation methods 
lies in the fact that internal metrics exploit characteristics of the data itself to produce a result, 
whereas relative validation converts an unsupervised clustering algorithm into a supervised classification problem, 
hence enabling generalizability and replicability of the results.

## Requirements

    python>=3.6
    
## Installing

From github:

    git clone https://github.com/IIT-LAND/reval_clustering
    pip install -r requirements.txt

PyPI alternative:

    pip install reval

## Documentation

Code documentation can be found [here](https://reval.readthedocs.io/en/latest/). Documents include Python code 
descriptions, `reval` usage examples, 
performance on benchmark datasets, and common issues that can be encountered related to a dataset number of features 
and samples.

## Refrences

[1] Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions. 
*Neural computation*, 16(6), 1299-1323.

## Cite as

    Isotta Landi, Veronica Mandelli, & Michael Vincent Lombardo. (2020, June 29). 
    reval: stability-based relative clustering validation method to determine the best number of clusters 
    (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3922334
    
BibTeX alternative

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