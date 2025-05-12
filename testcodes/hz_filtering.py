from src import NoiseHandler 
import argparse 
import os 

def main(args):
    noise_handler = NoiseHandler()
    noise_handler.filter_audio_with_ffmpeg(args.file_name, output_file= 'filtered_' + args.file_name)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)