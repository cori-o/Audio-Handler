from pyannote.audio import Pipeline
from dotenv import load_dotenv
import pandas as pd
import torch
import time
import json
import os 

load_dotenv()
hf_api_key = os.getenv('HF_API')

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_api_key)
start = time.time()
pipeline.to(torch.device("cuda"))
pipeline.instantiate({
    "onset": 0.3,  # 감지 민감도 증가 (기본값 0.5)
    "offset": 0.3,  # 감지 지속 시간 조정 (기본값 0.5)
})
output = pipeline("./meeting_records/20250220.wav")

vad_timestamp = [] 
for speech in output.get_timeline().support():
    vad_timestamp.append((speech.start, speech.end))

print(vad_timestamp)
print(f'end: {round(time.time() - start, 2)}')
pd.DataFrame(zip(vad_timestamp), columns=['timestamp']).to_csv(os.path.join('./meeting_records', 'vad_20250220.csv'), index=False)
