from pyannote.audio import Pipeline
from dotenv import load_dotenv
import pandas as pd
import torch
import time
import json
import os 

load_dotenv()
hf_api_key = os.getenv('HF_API')

start = time.time()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_api_key)
start = time.time()

pipeline.to(torch.device("cuda"))

output = pipeline("./meeting_records/20250220.wav")

diar_result = [] 
for segment, _, speaker in output.itertracks(yield_label=True):
    start_time = segment.start 
    end_time = segment.end
    duration = end_time - start_time 

    if duration >= 0.7:
        diar_result.append([(start_time, end_time), speaker])

df = pd.DataFrame(diar_result, columns=["timestamp", "speaker"])
df.to_csv("./meeting_records/diarization_20250220.csv", index=False)