import numpy as np
import librosa
import matplotlib
from matplotlib import pyplot as plt
import libfmp.c3
import libfmp.c4
import libfmp.c6
import plot

# Waveform
x_duration = 120
y, Fs = librosa.load('../songs/Duvet.mp3', duration=x_duration)

# Chroma Feature Sequence
N, H = 4096, 512
chromagram = librosa.feature.chroma_stft(y=y, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chromagram, Fs / H, filt_len=41, down_sampling=10)

# SSM
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X, X)
fig, ax = plot.plot_feature_ssm(S, 1, clim=[0, 1], label='Time (frames)',
                                title='SSM for Chroma Feature')
plt.show()
