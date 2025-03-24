from pyannote.audio import Pipeline
from src import SBEMB
import numpy as np
import argparse
import torch
import json
import os 


def main(args):
    with open(os.path.join(args.model_config_path, args.model_config_file)) as f:
        model_config = json.load(f) 

    speaker_emb = SBEMB(model_config)
    speaker_emb.set_classifier()
    speaker_emb.set_srmodel()
    audio_emb = speaker_emb.get_emb(speaker_emb.classifier, args.audio_file)
    print(np.shape(audio_emb))
    print(audio_emb[:3])
    print(speaker_emb.calc_emb_similarity(speaker_emb.srmodel, args.audio_file, './dataset/chunk/chunk_20250311_1.wav'))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--model_config_file', type=str, default='speechbrain_config.json')
    cli_parser.add_argument('--audio_file', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)