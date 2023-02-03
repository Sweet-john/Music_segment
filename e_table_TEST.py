import xlrd
import os
import librosa
from datetime import time
import lib
import libfmp.c4
import xlwt
import xlsxwriter

def evaluate(seg_num):
    # file path

    excel_path = './SMT_v2.xls'
    
    head = ['Name', 'Feature1', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure',
            'Feature2', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure',
            'Feature3', 'TP', 'FN', 'FP', 'Precision', 'Recall', 'F-measure']

    workbook = xlsxwriter.Workbook('evaluation_'+ str(seg_num) + '.xlsx')

    for i in range(4):

        if i == 0:
            songs = lib.excel_read(excel_path, 0)
            language = "CH"
        elif i == 1:
            songs = lib.excel_read(excel_path, 1)
            language = 'JP'
        elif i == 2:
            songs = lib.excel_read(excel_path, 2)
            language = 'KR'
        elif i == 3:
            songs = lib.excel_read(excel_path, 3)
            language = 'EN'
        #print(songs)
        worksheet = workbook.add_worksheet(language)
        worksheet.write_row('A1', head)

        database = "./SMT_dataset/" + language    
        results = []

        for audio_data in lib.find_all_song_file(database):
            excel_data = songs[audio_data]
            excel_segment = lib.get_excel_data_times(excel_data)

            x, sample_rate = librosa.load(database + '/' + audio_data)
            duration = librosa.get_duration(y=x, sr=sample_rate)
            time_segment = [lib.rhythm_segment(x, sample_rate, seg_num),
                            lib.chroma_segment(x, sample_rate, seg_num),
                            lib.mfcc_segment(x, sample_rate, seg_num)]
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
            #print(audio_data)
        lib.write_data(results, worksheet, start_row=1)

    workbook.close()



for seg_num in range(18,21):
    evaluate(seg_num)
    print("segment over: ", seg_num)
    
    


