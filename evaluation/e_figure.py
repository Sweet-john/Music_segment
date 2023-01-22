import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import lib
import libfmp.b
import libfmp.c4


def plot_boundary_measures(B_ref, B_est, tau, figsize=(8, 2.5)):
    """Plot B_ref and B_est (see :func:`libfmp.c4.c4s5_evaluation.evaluate_boundary`)

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        B_ref: Reference boundary annotations
        B_est: Estimated boundary annotations
        tau: Tolerance parameter
        figsize: Figure size (Default value = (8, 2.5))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = libfmp.c4.evaluate_boundary(B_ref, B_est, tau)

    colorList = np.array([[1., 1., 1., 1.], [0., 0., 0., 1.], [0.7, 0.7, 0.7, 1.]])
    cmap_tol = ListedColormap(colorList)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap_measures = ListedColormap(colorList)

    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.02], 'hspace': 0.8,
                                              'wspace': 0.1, 'height_ratios': [1, 1, 1]},
                           figsize=figsize)

    im = ax[0, 0].imshow(B_tol, cmap=cmap_tol, interpolation='nearest', aspect='auto')
    ax[0, 0].set_title('Reference boundaries (with tolerance)')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[0, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[1, 0].imshow(np.array([B_est]), cmap=cmap_tol, interpolation='nearest', aspect='auto')
    ax[1, 0].set_title('Estimated boundaries')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[1, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[2, 0].imshow(B_eval, cmap=cmap_measures, interpolation='nearest', aspect='auto')
    ax[2, 0].set_title('Evaluation')
    im.set_clim(vmin=-0.5, vmax=3.5)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[2, 1])
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    plt.tight_layout()
    plt.show()
    return fig, ax


def scaling(b, duration):
    result = []
    for i in b:
        r = np.floor(i)
        r = r / duration * 20
        result.append(r)


audio_data = '../SMT_dataset/CH/热爱105度的你.mp3'
songs = lib.excel_read('../SMT.xls', 0)

x, sample_rate = librosa.load(audio_data)
duration = librosa.get_duration(y=x, sr=sample_rate)
print(duration)

# time_segment = lib.chroma_segment(x, sample_rate)
# time_segment = lib.mfcc_segment(x, sample_rate)
time_segment = lib.rhythm_segment(x, sample_rate)

excel_data = songs['热爱105度的你.mp3']
excel_segment = lib.get_excel_data_times(excel_data)
print(time_segment)
print(excel_segment)

e_ref = np.zeros(int(np.floor(duration)) + 1)
e_est = np.zeros(int(np.floor(duration)) + 1)
for i in excel_segment:
    e_ref[i] = 1

for i in time_segment:
    e_est[int(np.floor(i))] = 1

tau_list = [3]

for tau in tau_list:
    print('====== Evaluation using tau = %d ======' % tau)
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = libfmp.c4.evaluate_boundary(e_ref, e_est, tau)
    print('#TP = ', num_TP, ';  #FN = ', num_FN, ';  #FP = ', num_FP)
    print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))
    fig, ax = plot_boundary_measures(e_ref, e_est, tau=tau)
