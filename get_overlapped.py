from pyannote.audio import Pipeline
from src import PyannotOSD
import argparse
import torch
import os 


def main(args):
    pyannot_overlap = PyannotOLP()
    pipeline = pyannot_overlap.load_pipeline_from_pretrained(os.path.join(args.model_config_path, args.model_config_file))  
    print(pipeline)
    overlap_result = pyannot_overlap.get_overlapped_result(pipeline, args.audio_file)
    print(overlap_result)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--model_config_file', type=str, default='pyannote_diarization_config.yaml')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)