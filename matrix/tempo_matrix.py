import librosa
import librosa.feature as feature
import librosa.display
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6


def compute_sm_dot(X, Y):
    S = np.dot(np.transpose(X), Y)
    return S


def plot_feature_ssm(X, Fs_X, S, Fs_S, duration, color_ann=None,
                     title='', label='Time (seconds)', time=True,
                     figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
    cmap = matplotlib.colormaps['viridis']  # libfmp.b.compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [0.3, 1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 0], ax[0, 1]], cmap=cmap, clim=clim_X,
                         xlabel='', ylabel='', title=title)

    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 0], ax[1, 1]], cmap=cmap, clim=clim,
                         title='', xlabel=label, ylabel='', colorbar=True)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    return fig, ax


def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N)
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = libfmp.c6.compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature


def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    return X_norm


# Compute local onset autocorrelation
x_duration = 120
y, Fs = librosa.load('../songs/Duvet.mp3', duration=x_duration)
N, H = 4096, 512
nov, Fs_nov = compute_novelty_spectrum(y, Fs=Fs, N=N, H=H, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=1102)

X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=N, H=H)
tempogram = np.abs(X)

print(tempogram.shape)

Fs_X = Fs_nov / H
# X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(tempogram, Fs / H)
X = normalize_feature_sequence(tempogram, norm='2', threshold=0.001)
S = compute_sm_dot(X, X)
fig, ax = plot_feature_ssm(X, 1, S, 1, x_duration * Fs_X, title='Tempogram (Fs=%0.2f)' % Fs_X,
                           label='Time (frames)')
plt.show()
