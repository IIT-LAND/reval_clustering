from matplotlib import pyplot as plt


def plot_metrics(cv_score,
                 figsize=(8, 5),
                 linewidth=1,
                 color=('black', 'black'),
                 legend_loc=2,
                 fontsize=12,
                 title="",
                 prob_lines=False,
                 save_fig=None):
    """
    Function that plots the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets. The horizontal lines represent the random performance error
    for the correspondent number of clusters.

    :param cv_score: collection of cv scores as output by `reval.best_nclust_cv.FindBestCLustCV.best_nclust`.
    :type cv_score: dictionary
    :param figsize: (width, height), default (8, 5).
    :type figsize: tuple
    :param linewidth: width of the lines to draw.
    :type linewidth: int
    :param color: line colors for train and validation sets, default ('black', 'black').
    :type color: tuple
    :param legend_loc: legend location, default 2.
    :type legend_loc: int
    :param fontsize: size of fonts, default 12.
    :type fontsize: int
    :param title: figure title, default "".
    :type title: str
    :param prob_lines: plot the normalized stability of random labeling as thresholds, default False.
    :type prob_lines: bool
    :param save_fig: file name for saving figure in png format, default None.
    :type save_fig: str
    """
    fig, ax = plt.subplots(figsize=figsize)
    cl_list = list(cv_score['train'].keys())
    ax.plot(list(cv_score['train'].keys()),
            [me[0] for me in cv_score['train'].values()],
            linewidth=linewidth,
            linestyle='-.',
            label='training set',
            color=color[0])
    ax.errorbar(list(cv_score['val'].keys()),
                [me[0] for me in cv_score['val'].values()],
                [me[1][1] for me in cv_score['val'].values()],
                linewidth=linewidth,
                linestyle='-',
                label='validation set',
                color=color[1])
    if prob_lines:
        plt.hlines([(1 - (1 / k)) for k in cl_list], xmin=[k - 0.1 for k in cl_list],
                   xmax=[k + 0.1 for k in cl_list], linewidth=linewidth,
                   color=color[1])
    ax.legend(fontsize=fontsize, loc=legend_loc)
    plt.xticks([lab for lab in cv_score['train'].keys()], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Number of clusters', fontsize=fontsize)
    plt.ylabel('Normalized stability', fontsize=fontsize)
    plt.title(title)
    if save_fig is not None:
        plt.savefig(f'./{save_fig}', format=str.split(save_fig, '.')[1])
    else:
        plt.show()
