from src import FrontendProcessor
import argparse 
import os 

def main(args):
    frontend_processor = FrontendProcessor()
    frontend_processor.set_env()
    clean_audio = frontend_processor.process_audio(args.file_name, deverve=True)
    frontend_processor.save_audio(clean_audio, './dataset/denoised/deverved-pipe-test.wav')


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)