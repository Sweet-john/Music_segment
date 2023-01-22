import libfmp.b
import matplotlib
from matplotlib import pyplot as plt


def plot_feature_ssm(S, Fs_S, title='', label='Time (seconds)',
                     figsize=(5, 5), clim=None):
    cmap = matplotlib.colormaps['viridis']
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[0], ax[1]], cmap=cmap, clim=clim,
                         title=title, xlabel=label, ylabel='', colorbar=True)

    return fig, ax
