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

PyPI alternative (latest version v0.1.0):

    pip install reval

## Documentation

Code documentation can be found [here](https://reval.readthedocs.io/en/latest/). Documents include Python code 
descriptions, `reval` usage examples, 
performance on benchmark datasets, and common issues that can be encountered related to a dataset number of features 
and samples.

## Manuscript

`reval` package functionalities are presented in our recent work that, as of now, can be found as a 
[preprint](https://arxiv.org/abs/2009.01077). The experiments presented in the manuscript are in 
the Python file `./working_examples/manuscript_examples.py` of the github folder. For reproducibility, all experiments 
were run with `reval v0.1.0`.

## Refrences

[1] Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions. 
*Neural computation*, 16(6), 1299-1323.

## Cite as

   Landi, I., Mandelli, V., & Lombardo, M. V. (2021). reval: A Python package to determine best clustering solutions with stability-based relative clustering validation. Patterns, 2(4), 100228.
 
BibTeX alternative

```
@article{LANDI2021100228,
title = {reval: A Python package to determine best clustering solutions 
         with stability-based relative clustering validation},
journal = {Patterns},
volume = {2},
number = {4},
pages = {100228},
year = {2021},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2021.100228},
url = {https://www.sciencedirect.com/science/article/pii/S2666389921000428},
author = {Isotta Landi and Veronica Mandelli and Michael V. Lombardo},
keywords = {stability-based relative validation, 
            clustering, 
            unsupervised learning, 
            clustering replicability}
}
```