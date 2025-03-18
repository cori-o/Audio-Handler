from pathlib import Path
from pyannote.audio import Pipeline
import torch
import os 

def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory

    # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)
    return pipeline

PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
pipeline.to(device)
print(pipeline)
diarization = pipeline("./dataset/20250220.wav")
diar_result = [] 
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start_time = segment.start 
    end_time = segment.end
    duration = end_time - start_time 
    if duration >= duration_thresh:
        diar_result.append([(start_time, end_time), speaker])

print(diar_result)