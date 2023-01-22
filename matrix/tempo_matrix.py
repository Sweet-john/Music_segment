import librosa
import librosa.display
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6
from matrix import plot


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


# Compute local onset autocorrelation
x_duration = 120
y, Fs = librosa.load('../songs/Duvet.mp3', duration=x_duration)
N, H = 4096, 512
nov, Fs_nov = compute_novelty_spectrum(y, Fs=Fs, N=N, H=H, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=1102)

X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=N, H=H)
tempogram = np.abs(X)
X = libfmp.c3.normalize_feature_sequence(tempogram, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X, X)
fig, ax = plot.plot_feature_ssm(S, 1, title='SSM for Tempogram', label='Time (frames)')
plt.show()
