from src import NoiseHandler
import argparse 
import time 
import os 

def main(args):
    noise_handler = NoiseHandler()
    time_s = time.time()    
    noise_handler.denoise_audio(args.file_name, model_type=args.model_type, data_path='./dataset/')
    time_e = time.time() 
    print(f"{args.file_name.split('/')[-1]} 잡음 제거 소요 시간: {round(time_e - time_s, 2)}초")


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=600)
    cli_parser.add_argument('--model_type', type=str, default='nsnet')
    cli_args = cli_parser.parse_args()
    main(cli_args)