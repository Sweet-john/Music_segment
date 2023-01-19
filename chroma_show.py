import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

#file path
audio_data = 'test dataset/6.mp3'


x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

#X = librosa.feature.chroma_cqt(y=x, sr=sample_rate)
X = librosa.feature.chroma_stft(y=x, sr=sample_rate)

plt.figure(figsize=(20, 5))
librosa.display.specshow(X, sr=sample_rate, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.show()
