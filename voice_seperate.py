from dotenv import load_dotenv
from src import NoiseHandler
import argparse 
import json 
import os 

def main(args):
    load_dotenv()
    noise_handler = NoiseHandler()
    noise_handler.separate_vocals_with_demucs(audio_file=args.file_name)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True) 
    cli_args = cli_parser.parse_args()
    main(cli_args)