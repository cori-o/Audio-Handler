from src import PyannotDIAR
from pyannote.audio import Pipeline
import argparse
import torch
import os 


def main(args):
    pyannot_diar = PyannotDIAR()
    pipeline = pyannot_diar.load_pipeline_from_pretrained(os.path.join(args.model_config_path, args.model_config_file))  
    print(pipeline)
    diar_result = pyannot_diar.get_diar_result(pipeline, args.file_name)
    print(diar_result)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--model_config_file', type=str, default='pyannote_diarization_config.yaml')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)