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


if __name__ == '__main__':
    filepath = 'D:\\dataset\\VCTK\\wav48_silence_trimmed\\p233\\p233_001_mic2.flac.wav'
    display_melspectrogram(filepath, "test")