<<<<<<< Updated upstream
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import lib


#file path
audio_data = 'SMT dataset/CH/天真有邪.mp3'
#audio_data = 'songs/Duvet.mp3'

x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

time_segment_r = lib.rhythm_segment(x,sample_rate) #rhythm
time_segment_h = lib.chroma_segment(x,sample_rate) #harmony
time_segment_t = lib.mfcc_segment(x,sample_rate) #timbre
print(time_segment_r)
print(time_segment_h)
print(time_segment_t)
=======
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import lib
import datetime
import string

#file path
audio_data = 'SMT_dataset/KR/청춘.mp3'
#audio_data = 'songs/Duvet.mp3'

x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

time_segment_r = lib.rhythm_segment(x,sample_rate) #rhythm
time_segment_h = lib.chroma_segment(x,sample_rate) #harmony
time_segment_t = lib.mfcc_segment(x,sample_rate) #timbre

time_segment_r = lib.second_2_min(time_segment_r)
time_segment_h = lib.second_2_min(time_segment_h)
time_segment_t = lib.second_2_min(time_segment_t)

# print in 'minute : second' type
print([i[2:i.find('.')] for i in time_segment_r])
print([i[2:i.find('.')] for i in time_segment_h])
print([i[2:i.find('.')] for i in time_segment_t])
>>>>>>> Stashed changes
