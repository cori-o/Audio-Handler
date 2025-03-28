from src import NoiseHandler
import argparse 
import os 

def main(args):
    noise_handler = NoiseHandler()
    noise_handler.denoise_audio(args.file_name, data_path='./dataset/')

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)