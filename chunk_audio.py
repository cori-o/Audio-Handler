from src import DataProcessor, AudioFileProcessor
import argparse
import os

def main(args):
    file_name = args.file_name
    audio_file_path = os.path.join(args.data_path, file_name)    

    audio_p = AudioFileProcessor()
    audio_p.chunk_audio(audio_file_path, chunk_length=args.chunk_length, chunk_file_path=args.output_path, chunk_file_name='chunk_' + args.file_name.split('.')[0])

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./dataset')
    cli_parser.add_argument('--output_path', type=str, default='./dataset/chunk')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=270)
    cli_args = cli_parser.parse_args()
    main(cli_args)