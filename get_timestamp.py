import matplotlib.pyplot as plt
import pandas as pd
import json 
import os 
import ast

def plot_timestamps(whisper_timestamps, vad_timestamps, diar_timestamps):
    plt.figure(figsize=(12, 4))

    for start, end in whisper_timestamps:
        plt.hlines(y=0, xmin=start, xmax=end, color="blue", linewidth=3, label="Whisper" if start == whisper_timestamps[0][0] else "")

    for start, end in vad_timestamps:
        plt.hlines(y=1, xmin=start, xmax=end, color="red", linewidth=3, label="VAD" if start == vad_timestamps[0][0] else "")

    for start, end in diar_timestamps:
        plt.hlines(y=2, xmin=start, xmax=end, color="green", linewidth=3, label="DIAR" if start == diar_timestamps[0][0] else "")

    plt.xlim(0, 7000)
    plt.xlabel("Time (seconds)")
    
    plt.yticks([0, 1, 2], ["Whisper", "VAD", "DIAR"])
    plt.legend()
    plt.title("Whisper vs VAD vs Diarization Timestamps")
    plt.show()


whisper_result = pd.read_csv(os.path.join('./meeting_records', 'local_20250220.csv'))
str_whisper_timestamps = whisper_result['timestamp'].values.tolist()
vad_result = pd.read_csv(os.path.join('./meeting_records', 'vad_20250220.csv'))
str_vad_timestamps = vad_result['timestamp'].values.tolist()
diar_result = pd.read_csv(os.path.join('./meeting_records', 'diar_20250220.csv'))
str_diar_timestamps = diar_result['timestamp'].values.tolist()

whisper_timestamps = [] 
vad_timestamps = []
diar_timestamps = [] 
for str_whisper_timestamp in str_whisper_timestamps: 
    whisper_tuple = ast.literal_eval(str_whisper_timestamp)  # (6498.23909375, 6499.065968750001)
    whisper_timestamp_set = list(whisper_tuple)
    whisper_timestamps.append(whisper_timestamp_set)
    
for str_vad_timestamp in str_vad_timestamps: 
    vad_tuple = ast.literal_eval(str_vad_timestamp)  # (6498.23909375, 6499.065968750001)
    vad_timestamp_set = list(vad_tuple)
    vad_timestamps.append(vad_timestamp_set)

for str_diar_timestamp in str_diar_timestamps: 
    diar_tuple = ast.literal_eval(str_diar_timestamp)  # (6498.23909375, 6499.065968750001)
    diar_timestamp_set = list(diar_tuple)
    diar_timestamps.append(diar_timestamp_set)
    
plot_timestamps(whisper_timestamps, vad_timestamps, diar_timestamps)
