from src import AudioVisualizer
import librosa.display
import librosa
import matplotlib
matplotlib.use("Agg")

import noisereduce as nr
import numpy as np
import argparse 
import os 


def main(args): 
    audio_visualizer = AudioVisualizer()
    y, sr = librosa.load(args.file_name, sr=None)

    # 개별 시각화
    audio_visualizer.get_waveform(y, sr, title_prefix="My Audio", file_name="waveform.png")
    audio_visualizer.get_spectrogram(y, sr, title_prefix="My Audio", file_name="stft.png")
    audio_visualizer.get_melspectrogram(y, sr, title_prefix="My Audio", file_name="mel.png")
    audio_visualizer.visualize_all(y, sr, title_prefix="My Audio", file_name="full_visual.png")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_name', type=str, required=True)
    cli_args = argparser.parse_args()
    main(cli_args)