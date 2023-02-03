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
                     figsize=(5, 5), fontsize=10, clim_X=None, clim=None):
    cmap = matplotlib.colormaps['viridis']
    #fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
    #                                          'wspace': 0.2,
    #                                          'height_ratios': [0.3, 1]},
    #                       figsize=figsize)
    #libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 0], ax[0, 1]], cmap=cmap, clim=clim_X,
    #                     xlabel='', ylabel='', title=title)
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [1]},
                                              figsize=figsize)
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[0], ax[1]], cmap=cmap, clim=clim,
                         title=title, xlabel=label, ylabel='', colorbar=True)
    #ax[0, 1].set_xticks([])
    #ax[0, 1].set_yticks([])

    return fig, ax


# Waveform
x_duration = 120
y, Fs = librosa.load('../SMT_dataset/EN/Phoenix.mp3', duration=x_duration)

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
                           title='SSM for Chroma Feature')
plt.show()
