from src import NoiseHandler
import argparse 
import os 

def main(args):
    noise_handler = NoiseHandler()
    noise_handler.denoise_audio(args.file_name, model_type=args.model_type, data_path='./dataset/')

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--model_type', type=str, default='nsnet')
    cli_args = cli_parser.parse_args()
    main(cli_args)