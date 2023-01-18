import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import lib

#file path
audio_data = 'test dataset/6.mp3'
#audio_data = 'songs/Duvet.mp3'

x , sample_rate = librosa.load(audio_data)

chroma_ref = librosa.feature.chroma_cqt(y=x, sr=sample_rate)
#chroma_ref = librosa.feature.mfcc(y=x, sr=sample_rate)
lib.cal_collerate

m = round(5*sample_rate/512)
tau = 1
w = (m-1)*tau
chroma_ref = np.concatenate((np.zeros((chroma_ref.shape[0], w)), chroma_ref), axis=1)

#R = librosa.segment.recurrence_matrix(chroma_ref, mode='affinity', k=chroma_ref.shape[1])
#R_aff = librosa.segment.recurrence_matrix(chroma_ref, metric='cosine',mode='affinity')


recurrence = librosa.segment.recurrence_matrix(chroma_ref, mode='affinity', k=chroma_ref.shape[1])
plt.figure(figsize=(7, 7))
plt.title('Recurrence matrix from chroma vector from LIBROSA')
plt.imshow(recurrence, cmap='gray')
plt.show()

'''

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
imgsim = librosa.display.specshow(R, x_axis='s', y_axis='s',
                          cmap='jet', ax=ax[0])
ax[0].set(title='Binary cross-similarity (symmetric)')
imgaff = librosa.display.specshow(R_aff, x_axis='s', y_axis='s',
                         cmap='jet', ax=ax[1])
ax[1].set(title='Cross-affinity')
ax[1].label_outer()
fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')

plt.show()
'''