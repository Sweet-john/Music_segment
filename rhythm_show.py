import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import librosa
import librosa.display
import numpy as np
import lib


#file path
#audio_data = 'test dataset/6.mp3'
audio_data = 'SMT dataset/CH/安平之光.mp3'
x , sample_rate = librosa.load(audio_data)


#x = librosa.stft(y=x)

o_env = librosa.onset.onset_strength(y=x, sr=sample_rate)
times = librosa.times_like(o_env, sr=sample_rate)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sample_rate)


collerate_list = lib.cal_collerate(o_env)

collerate_list_trans = list(map(list,zip(*collerate_list)))

fig = plt.figure(figsize=(10,8))

plt.xlabel("Time (min.)",size = 14)
plt.ylabel("Rhythm interval (sec.)",size = 14)

plt.imshow(collerate_list_trans,cmap='OrRd', aspect = 100 , 
            extent = [ 0 , librosa.get_duration(y=x, sr=sample_rate) , 0 , 2 ] , norm = 'linear', origin = 'lower')
plt.colorbar()
plt.show()

'''
# division of onset_tgram
print(len(o_env),len(times),len(onset_frames))
times = times[0:100]
o_env = o_env[0:100]
onset_frames = onset_frames[0:7] #could change upperbound

#plt
fig, ax = plt.subplots(nrows=1)
ax.set(title='onset gram')
ax.plot(times, o_env, label='Onset strength')
ax.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')
ax.legend()
plt.show()

'''

