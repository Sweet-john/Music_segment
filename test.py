import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import lib

SEGMENT_NUM = 10

#file path
audio_data = 'test dataset/6.mp3'

x , sample_rate = librosa.load(audio_data)
print(x.shape, sample_rate)
print (librosa.get_duration(x))

time_segment = lib.mfcc_segment(x,sample_rate)
print(time_segment)
