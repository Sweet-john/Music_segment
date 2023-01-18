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
