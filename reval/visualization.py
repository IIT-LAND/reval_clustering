import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import export_png
from matplotlib import pyplot as plt


def plot_metrics(cv_score, title, figsize=(20, 10), save_fig=None):
    """
    Function that plots the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets.

    :param cv_score: collection of cv scores as output by `reval.best_nclust_cv.FindBestCLustCV.best_nclust`
    :type cv_score: dictionary
    :param title: figure title
    :type title: str
    :param figsize: (width, height)
    :type figsize: tuple
    :param save_fig: file name for saving figure in png format, default None
    :type save_fig: str
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(cv_score['train'].keys()),
            [me[0] for me in cv_score['train'].values()],
            linewidth=5,
            label='training set')
    ax.errorbar(list(cv_score['val'].keys()),
                [me[0] for me in cv_score['val'].values()],
                [me[1][1] for me in cv_score['val'].values()],
                linewidth=5,
                label='validation set')
    ax.legend(fontsize=18, loc=1)
    plt.xticks([lab for lab in cv_score['train'].keys()], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of clusters', fontsize=18)
    plt.ylabel('Normalized stability', fontsize=18)
    plt.title(title, fontsize=18)
    if save_fig is not None:
        plt.savefig(f'./{save_fig}', format='png')
    else:
        plt.show()

