import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import export_png
from matplotlib import pyplot as plt


def plot_metrics(cv_score, title, figsize=(20, 10), save_fig=None):
    """
    Function that plots the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets.

    :param cv_score: collection of cv scores
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
    ax.legend(fontsize=18, loc=2)
    plt.xticks([lab for lab in cv_score['train'].keys()], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of clusters', fontsize=18)
    plt.ylabel('Normalized stability', fontsize=18)
    plt.title(title, fontsize=18)
    if save_fig is not None:
        plt.savefig(f'./plot/{save_fig}', format='png')
    else:
        plt.show()


def scatter_plot(mtx,
                 el_cl_list,
                 colors,
                 fig_height,
                 fig_width,
                 label,
                 title='',
                 save_fig=None):
    """
    Bokeh scatterplot to visualize clusters and row info.

    :param mtx: Array with dataset (after dimensionality reduction)
    :type mtx: numpy array
    :param el_cl_list: list of data elements (i.e., rows) ordered as in mtx and cluster labels
    :type el_cl_list: list of tuples
    :param colors: Color list
    :type colors: list
    :param fig_height: figure height
    :type fig_height: int
    :param fig_width: figure width
    :type fig_width: int
    :param label: dictionary of class numbers and subtype labels, e.g. {1: 'cluster 1', 2: 'cluster 2'}
    :type label: dict
    :param title: figure title
    :type title: str
    :param save_fig: flag to enable figure saving, defaults to None
    :type save_fig: str
    """

    el_list = list(map(lambda x: x[0], el_cl_list))
    subc_list = list(map(lambda x: x[1], el_cl_list))
    df_dict = {'x': mtx[:, 0].tolist(),
               'y': mtx[:, 1].tolist(),
               'pid_list': el_list,
               'subc_list': subc_list}

    df = pd.DataFrame(df_dict).sort_values('subc_list')

    source = ColumnDataSource(dict(
        x=df['x'].tolist(),
        y=df['y'].tolist(),
        el=df['el_list'].tolist(),
        subc=list(map(lambda x: label[str(x)], df['subc_list'].tolist())),
        col_class=[str(i) for i in df['subc_list'].tolist()]))

    labels = [str(i) for i in df['subc_list']]
    cmap = CategoricalColorMapper(factors=sorted(pd.unique(labels)),
                                  palette=colors)
    TOOLTIPS = [('el', '@el'),
                ('subc', '@subc')]

    plotTools = 'box_zoom, wheel_zoom, pan,  crosshair, reset, save'

    output_notebook()
    p = figure(plot_width=fig_width * 80, plot_height=fig_height * 80,
               tools=plotTools, title=title)
    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    p.circle('x', 'y', legend_group='subc', source=source,
             color={'field': 'col_class',
                    "transform": cmap}, size=12)
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_color = None
    p.yaxis.major_label_text_color = None
    p.grid.grid_line_color = None
    p.title.text_font_size = '13pt'
    p.legend.label_text_font_size = '18pt'
    p.legend.location = 'top_left'
    if save_fig is not None:
        export_png(p, filename=f'./plot/{save_fig}.png')
    else:
        show(p)
