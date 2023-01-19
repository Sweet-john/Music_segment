import xlrd
import os
import librosa
from datetime import time
import lib

# file path
database = "./SMT_dataset/CH"
excel_path = 'SMT.xls'

CH_songs = lib.excel_read(excel_path, 0)

for audio_data in lib.find_all_song_file(database):

    # Find segment data in excel
    try:
        print(audio_data, CH_songs[audio_data])
    except:
        print('Cannot find segment data in excel: ', audio_data)

    # load mp3 file
    try:
        x , sample_rate = librosa.load(database + '/' + audio_data)
    except:
        print('Mp3 error: ', audio_data)
