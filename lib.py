import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

SEGMENT_NUM = 10

def chroma_segment(audio,sample_rate):
    X = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    bounds = librosa.segment.agglomerative(X, SEGMENT_NUM)
    bound_times = librosa.frames_to_time(bounds, sr=sample_rate)
    return bound_times

def mfcc_segment(audio,sample_rate):
    X = librosa.feature.mfcc(y=audio, sr=sample_rate)
    bounds = librosa.segment.agglomerative(X, SEGMENT_NUM)
    bound_times = librosa.frames_to_time(bounds, sr=sample_rate)
    return bound_times


def rhythm_segment(audio,sample_rate):
    return

