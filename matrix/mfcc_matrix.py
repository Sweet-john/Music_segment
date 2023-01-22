from libfmp.b import FloatingBox
import numpy as np
import librosa
import matplotlib
from matplotlib import pyplot as plt
import libfmp.c3
import libfmp.c4
import libfmp.c6
from matrix import plot

float_box = libfmp.b.FloatingBox()

# MFCC-based feature sequence
x_duration = 120
y, Fs = librosa.load('../songs/Duvet.mp3', duration=x_duration)
N, H = 4096, 512
X_MFCC = librosa.feature.mfcc(y=y, sr=Fs, hop_length=H, n_fft=N)
'''coef = np.arange(0, 20)
X_MFCC_upper = X_MFCC[coef, :]
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X_MFCC_upper, Fs/H, filt_len=41, down_sampling=10)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X,X)
fig, ax = plot_feature_ssm(X, 1, S, 1, x_duration*Fs_X,
    title='SSM for MFCC  (20 coefficients)', label='Time (frames)')'''
# float_box.add_fig(fig)


# MFCC-based feature sequence only using coefficients 4 to 14
coef = np.arange(4, 15)
X_MFCC_upper = X_MFCC[coef, :]
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X_MFCC_upper, Fs / H, filt_len=41, down_sampling=10)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X, X)
fig, ax = plot.plot_feature_ssm(S, 1, label='Time (frames)',
                                title='SSM for MFCC (coefficients 4 to 14)')
# float_box.add_fig(fig)
# float_box.show()
plt.show()
