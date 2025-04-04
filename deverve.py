import numpy as np
import soundfile as sf
from nara_wpe.wpe import wpe
import argparse 
import os 

def main(args):
    audio, sr = sf.read(os.path.join(args.data_path, 'denoised', args.file_name))  # 입력 오디오 파일 (멀티 채널 지원)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]     # (samples, 1)
    audio = audio.T     # (1, samples)
    print(audio.shape)
    clean_audio = wpe(audio, iterations=5, taps=10, delay=3)

    print(clean_audio.shape)    # (channels, samples) 일 경우
    sf.write(os.path.join(args.data_path, "deverve_" + args.file_name), clean_audio.T, sr)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='dataset')
    cli_parser.add_argument("--file_name", type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)