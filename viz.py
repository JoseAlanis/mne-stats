# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from mne.utils import _check_pandas_installed


def plot_design_matrix(inst, predictors, scale=True, contrast='effect',
                       cmap='binary_r'):
    """
    Parameters
    ----------
    inst : instance of Epochs
        The data that contains infromation about the design.
    predictors : array-like
        Names of the regressors.
    contrast : str
        Type of contrast to use. Can be 'effect' (1 vs -1) or
        'dummy' (1 vs. 0). Defaults to effect.
    scale : bool
        Whether to scale predictors.
    cmap : str
        colormap to use. Default to binary_r.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----

    """
    import matplotlib.pyplot as plt

    pd = _check_pandas_installed()

    # possible formula
    # '~face + phase_coherence'  # could be parsed with split.

    epochs = inst.copy()
    design = epochs.metadata.copy()

    samples = list(range(0, design.shape[0]))
    names = predictors.copy()

    data = design[names].copy()
    # scale = True
    cbar_kw={}
    # main = ['Face_Condition', 'Noise']

    # preset
    data = data.assign(intercept=1)  # add intercept
    data['face a - face b'] = np.where(design['face'] == 'A', 1, -1)

    # dynamic
    # predictors_recoded = dict(predictors=[])
    # for predictor in predictors:
    #     if data[predictor].dtype == 'float64':
    #         continue
    #     else:
    #         predictors_recoded[] = {'%_recoded': bool(data[predictor])} % predictor

    predictors = ['intercept', 'face a - face b', 'phase-coherence']

    # scale
    if scale:
        for predictor in predictors:
            # data['Intercept'] = design.Intercept.copy()
            data[predictor] = np.interp(data[predictor], (data[predictor].min(), data[predictor].max()), (0, +1))
        cbarlabel = "Weight [scaled 0 - 1]"
    else:
        cbarlabel = "Weight [uncaled]"

    # interactions
    # data['Face_x_Noise'] = np.zeros(data.shape[0], int)
    # data['Face_x_Noise'] = data['Face_Condition'] * data['Noise']

    data_to_show = np.array(data[predictors])

    vmax = data_to_show.max()
    vmin = data_to_show.min()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data_to_show, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # create color bar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(predictors)))
    # ax.set_yticks(np.arange(len(samples)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(predictors)
    # ax.set_yticklabels(np.arange(len(samples)))
    ax.set_xlabel('Predictors', fontsize=12)
    ax.xaxis.set_label_coords(0.5, -0.20)
    ax.set_ylabel('Epochs', fontsize=12)

    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    rotate = [name for name in predictors if len(name) > 10]
    if rotate:
        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data_to_show.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.array([0, data_to_show.shape[0]]), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Design Matrix")
    fig.tight_layout()
    plt.show()

    return fig
