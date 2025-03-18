from src import PyannoteDIARP
from dotenv import load_dotenv
import pandas as pd
import argparse
import torch
import time
import json
import os 


def main(args):
    load_dotenv()
    hf_api_key = os.getenv('HF_API')
    with open(os.path.join(args.config_path, args.config_file)) as f:
        audio_config = json.load(f)
    audio_config['hf_key'] = hf_api_key
    
    start = time.time()
    diar_pipe = PyannoteDIARP(audio_config)
    diar_result = diar_pipe.get_diar(args.audio_file)
    df = pd.DataFrame(diar_result, columns=["timestamp", "speaker"])
    df.to_csv("./dataset/diarization_20250220.csv", index=False)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config')
    cli_parser.add_argument('--config_file', type=str, default='audio_config.json')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)