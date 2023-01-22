import xlrd
import os
import librosa
from datetime import time
import lib
import libfmp.c4


def evaluate(language='CH'):
    # file path
    database = "../SMT_dataset/" + language
    excel_path = '../SMT.xls'
    songs = {}
    results = []

    if language == 'CH':
        songs = lib.excel_read(excel_path, 0)
    elif language == 'JP':
        songs = lib.excel_read(excel_path, 1)
    elif language == 'KR':
        songs = lib.excel_read(excel_path, 2)
    elif language == 'EN':
        songs = lib.excel_read(excel_path, 3)

    head = ['Name', 'Feature1', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure',
            'Feature2', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure',
            'Feature3', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure']
    wb, ws = lib.create_table('evaluation_' + language + '.xlsx', labels=head)

    for audio_data in lib.find_all_song_file(database):
        excel_data = songs[audio_data]
        excel_segment = lib.get_excel_data_times(excel_data)

        x, sample_rate = librosa.load(database + '/' + audio_data)
        duration = librosa.get_duration(y=x, sr=sample_rate)
        time_segment = [lib.rhythm_segment(x, sample_rate),
                        lib.chroma_segment(x, sample_rate),
                        lib.mfcc_segment(x, sample_rate)]
        r = []
        for f in range(3):
            feature = " "
            if f == 0:
                feature = "rhythm"
                r.append(audio_data)
            elif f == 1:
                feature = "chroma"
            elif f == 2:
                feature = "timbre"

            ref = lib.get_result_sequence(excel_segment, duration)
            est = lib.get_result_sequence(time_segment[f], duration)
            tau = 3

            P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = libfmp.c4.evaluate_boundary(ref, est, tau)
            r.extend([feature, num_TP, num_FN, num_FP, P, R, F])
        results.append(r)
        print(audio_data)
    lib.write_data(results, ws, start_row=1)
    wb.close()


evaluate(language='CH')
