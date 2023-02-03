<<<<<<< Updated upstream
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

#file path
audio_data = 'test dataset/6.mp3'


x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

mfccs = librosa.feature.mfcc(y=x, sr=sample_rate)

fig, ax = plt.subplots(nrows=2, sharex=True)

S = np.abs(librosa.stft(x))

img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])

ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

=======
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

#file path
audio_data = 'SMT_dataset/CH/热爱105度的你.mp3'


x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

mfccs = librosa.feature.mfcc(y=x, sr=sample_rate)

fig, ax = plt.subplots(nrows=2, sharex=True)

S = np.abs(librosa.stft(x))

img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])

ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

>>>>>>> Stashed changes
plt.show()