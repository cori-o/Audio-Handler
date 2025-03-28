import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import argparse 
import os 


def visualize_all(y, sr, title_prefix="", file_name=None):
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # STFT Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

    # 시각화
    plt.figure(figsize=(14, 10))

    # 1. Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{title_prefix} - Waveform")

    # 2. STFT Spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title_prefix} - STFT Spectrogram")

    # 3. Mel Spectrogram
    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title_prefix} - Mel Spectrogram")

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300)
    plt.close()


def main(args): 
    # y, sr = librosa.load(args.file_name, sr=None)
    # visualize_all(y, sr, file_name='origin.png')
    y2, sr2 = librosa.load(args.file_name2, sr=None)
    visualize_all(y2, sr2, file_name='after.png')
    '''
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)    
    librosa.display.waveshow(y, sr=sr)
    plt.title("Before Noise Reduction") 

    y2, sr2 = librosa.load(args.file_name2, sr=None) 
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y2, sr=sr)
    plt.title("After Noise Reduction")
    plt.tight_layout()
    plt.show()
    '''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_name', type=str, required=True)
    argparser.add_argument('--file_name2', type=str, required=True)
    cli_args = argparser.parse_args()
    main(cli_args)