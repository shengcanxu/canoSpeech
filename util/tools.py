import os
from glob import glob

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def display_melspectrogram(filepath:str, name = 'Mel spectrogram'):
    y, sr = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    plt.tight_layout()
    plt.show()

def remove_backup_dataset_pt_file(path:str):
    files = glob(f"{path}/**/*.wav.pt", recursive=True)
    for file in files:
        os.remove(file)
        print(f"remove file: {file}")

if __name__ == '__main__':
    # filepath = 'D:\\dataset\\VCTK\\wav48_silence_trimmed\\p233\\p233_001_mic2.flac.wav'
    # display_melspectrogram(filepath, "test")

    remove_backup_dataset_pt_file('D:\\dataset\\kokoro\\wavs')
    remove_backup_dataset_pt_file('D:\\dataset\\CMLTTS\\train\\audio')
    remove_backup_dataset_pt_file('D:\\dataset\\VCTK\\wav48_silence_trimmed')
    remove_backup_dataset_pt_file('D:\\dataset\\baker')

    # remove_backup_dataset_pt_file('/home/cano/dataset/kokoro/wavs')
    # remove_backup_dataset_pt_file('/home/cano/dataset/CMLTTS/train/audio')
    # remove_backup_dataset_pt_file('/home/cano/dataset/VCTK/wav48_silence_trimmed')
    # remove_backup_dataset_pt_file('/home/cano/dataset/baker')