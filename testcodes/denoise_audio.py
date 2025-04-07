from src import NoiseHandler, AudioFileProcessor
import argparse 
import time 
import os 

def main(args):
    noise_handler = NoiseHandler()
    audio_file_p = AudioFileProcessor()
    save_path = './dataset/outputs'
    file_name = 'new_denoised_' + args.file_name.split('/')[-1]

    time_s = time.time()      
    denoised_audio = noise_handler.denoise_audio(args.file_name, model_type=args.model_type)
    audio_file_p.save_audio(denoised_audio, save_path, file_name)
    time_e = time.time() 
    print(f"{args.file_name.split('/')[-1]} 잡음 제거 소요 시간: {round(time_e - time_s, 2)}초")


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--model_type', type=str, default='nsnet')
    cli_args = cli_parser.parse_args()
    main(cli_args)