from src import PyannotVAD
from pathlib import Path
import argparse
import torch
import os 


def main(args):
    pyannot_vad = PyannotVAD()
    pipeline = pyannot_vad.load_pipeline_from_pretrained(os.path.join(args.model_config_path, args.model_config_file))  
    print(pipeline)
    vad_result = pyannot_vad.get_vad_timestamp(pipeline, args.audio_file)
    print(vad_result)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--model_config_file', type=str, default='pyannote_vad_config.yaml')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)