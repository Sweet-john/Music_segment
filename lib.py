import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import datetime

SEGMENT_NUM = 14

def cal_collerate(onsetS):
    # 8 seconds = 344   2 seconds = 86  10 seconds = 430  7.5 seconds = 322
    result = []
    init_index = 0
    while (init_index + 430) < len(onsetS):

        origin = onsetS[init_index : init_index + 344]
        similarity_list = []

        for i in range(86):
            target = onsetS[init_index + i : init_index + 344 + i]
            similarity = np.correlate(origin, target)
            similarity_list.append(similarity[0])
        
        result.append(similarity_list)
        init_index = init_index + 322

    return result

def chroma_segment(audio,sample_rate):
    X = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    print('x:',len(X),'y:',len(X[0]))
    bounds = librosa.segment.agglomerative(X, SEGMENT_NUM)
    bound_times = librosa.frames_to_time(bounds, sr=sample_rate)
    return bound_times

def mfcc_segment(audio,sample_rate):
    X = librosa.feature.mfcc(y=audio, sr=sample_rate)
    print('x:',len(X),'y:',len(X[0]))
    bounds = librosa.segment.agglomerative(X, SEGMENT_NUM)
    bound_times = librosa.frames_to_time(bounds, sr=sample_rate)
    return bound_times


def rhythm_segment(audio,sample_rate):
    o_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)

    collerate_list = cal_collerate(o_env)
    
    collerate_list_trans = list(map(list,zip(*collerate_list)))

    #print('x:',len(collerate_list),'y:',len(collerate_list[0]))

    bounds = librosa.segment.agglomerative(collerate_list_trans, SEGMENT_NUM)
    new_bounds = [i * 322 for i in bounds]
    bound_times = librosa.frames_to_time(new_bounds, sr=sample_rate)
    return bound_times

def second_2_min(list_second):
    min_list = []
    for i in list_second:
        tranform_i = datetime.timedelta(seconds=i)
        #tranform_i.microseconds = 0

        min_list.append(str(tranform_i))
        #m, s = divmod(i, 60)
        #min_list.append((round(m),round(s)))
    return min_list

