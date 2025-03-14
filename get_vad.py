from pyannote.audio import Inference, Model
from dotenv import load_dotenv
from src import PyannotVAD
import argparse
import torch
import json
import os 

def main(args): 
    load_dotenv()
    hf_api_key = os.getenv('HF_API')
    with open(os.path.join(args.config_path, args.config_file)) as f:
        audio_config = json.load(f)

    audio_config['hf_key'] = hf_api_key
    pyannot_emb = PyannotVAD(audio_config)
    pyannot_emb.set_inference()
    vad = pyannot_emb.get_vad(pyannot_emb.inference, audio_file=args.audio_file)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='config')
    cli_parser.add_argument('--config_file', type=str, default='audio_config.json')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)