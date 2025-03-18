from dotenv import load_dotenv
from src import PyannoteVADP
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
    vad_pipe = PyannoteVADP(audio_config)
    vad_result = vad_pipe.get_vad(args.audio_file)
    df = pd.DataFrame(vad_result, columns=["time_s", "time_e"])
    csv_filename = 'vad_' + args.audio_file.split('.')[0] + '.csv'
    df.to_csv(os.path.join(args.data_path, csv_filename), index=False)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config')
    cli_parser.add_argument('--data_path', type=str, default='./dataset')
    cli_parser.add_argument('--config_file', type=str, default='audio_config.json')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)