import numpy as np
import os, sys, librosa
from scipy import signal
import matplotlib
from matplotlib import pyplot as plt
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6


def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S


def plot_feature_ssm(X, Fs_X, S, Fs_S, duration, color_ann=None,
                     title='', label='Time (seconds)', time=True,
                     figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with feature representation and annotations (standard setting is time in seconds)

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X: Feature representation
        Fs_X: Feature rate of ``X``
        S: Similarity matrix (SM)
        Fs_S: Feature rate of ``S``
        ann: Annotaions
        duration: Duration
        color_ann: Color annotations (see :func:`libfmp.b.b_plot.plot_segments`) (Default value = None)
        title: Figure title (Default value = '')
        label: Label for time axes (Default value = 'Time (seconds)')
        time: Display time axis ticks or not (Default value = True)
        figsize: Figure size (Default value = (5, 6))
        fontsize: Font size (Default value = 10)
        clim_X: Color limits for matrix X (Default value = None)
        clim: Color limits for matrix ``S`` (Default value = None)

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    cmap = matplotlib.colormaps['viridis']#libfmp.b.compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [0.3, 1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 0], ax[0, 1]], cmap=cmap, clim=clim_X,
                         xlabel='', ylabel='', title=title)
    #ax[0, 0].axis('off')
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 0], ax[1, 1]], cmap=cmap, clim=clim,
                         title='', xlabel=label, ylabel='', colorbar=True)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    #libfmp.b.plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
    #                       colors=color_ann,
    #                       time_label=label, time_max=duration * Fs_X)
    # ax[2, 2].axis('off'), ax[2, 0].axis('off')
    #libfmp.b.plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
    #                       direction='vertical', colors=color_ann,
    #                       time_label=label, time_max=duration * Fs_X)
    return fig, ax


# Waveform
x_duration = 120
y, Fs = librosa.load('../songs/Duvet.mp3', duration=x_duration)

# Chroma Feature Sequence
N, H = 4096, 512
chromagram = librosa.feature.chroma_stft(y=y, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chromagram, Fs / H, filt_len=41, down_sampling=10)

# Annotation
# filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
# fn_ann = os.path.join('..', 'data', 'C4', filename)
# ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)
# ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)

# SSM
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X, X)
fig, ax = plot_feature_ssm(X, 1, S, 1, x_duration,
                           clim_X=[0, 1], clim=[0, 1], label='Time (frames)',
                           title='Chroma feature (Fs=%0.2f)' % Fs_X)
plt.show()
