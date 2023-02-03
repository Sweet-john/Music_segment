import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import librosa
import librosa.display
import numpy as np
import lib


#file path
audio_data = 'SMT_dataset/CH/热爱105度的你.mp3'

y, sr = librosa.load(audio_data)
S = np.abs(librosa.stft(y))

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()