from pyannote.audio import Inference
from pyannote.audio import Pipeline
from pyannote.audio import Model 
from pyannote.core import Segment
from pathlib import Path
import numpy as np 
import torch
import random 
import os  

class Pyannot:
    def __init__(self): 
        self.set_seed()
        self.set_gpu()

    def set_seed(self, seed=42):
        """랜덤 시드 설정"""
        self.seed = seed
        random.seed(seed)  
        np.random.seed(seed)  
        torch.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)    # GPU 연산을 위한 시드 설정
        torch.backends.cudnn.deterministic = True   # 연산 재현성을 보장
        torch.backends.cudnn.benchmark = False    # 성능 최적화 옵션 비활성화

    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    
    def load_pipeline_from_pretrained(self, path_to_config: str) -> Pipeline:
        '''
        the paths in the config are relative to the current working directory
        so we need to change the working directory to the model path
        and then change it back
        * first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
        '''
        path_to_config = Path(path_to_config)
        print(f"Loading pyannote pipeline from {path_to_config}...")
        cwd = Path.cwd().resolve()    # store current working directory

        cd_to = path_to_config.parent.parent.resolve()
        os.chdir(cd_to)

        pipeline = Pipeline.from_pretrained(path_to_config)
        os.chdir(cwd)
        return pipeline.to(self.device)


class PyannotEMB(Pyannot):
    def __init__(self):
        super().__init__()

    def set_config(self):
        pass 

    def set_inference(self, config, window, duration=None, step=None, model_loc='local'):
        '''
        window = whole : 
        window = sliding : 
        '''
        self.config = config
        if window == 'whole':
            self.inference = Inference(os.path.join(self.config['local_model_path'], 'emb', 'config.yaml'), window=window, device=self.device)
        elif window == 'sliding':
            self.inference = Inference(os.path.join(self.config['local_model_path'], 'emb'), 
                                        window=window, duration=duration, step=step, device=self.device)

    def get_embedding(self, inference, audio_file, time_s=None, time_e=None):
        '''
        audio_file: test.wav
        '''
        if time_s == None: 
            embedding = inference(audio_file)
            print(np.shape(embedding.data))
            return embedding
        else:
            excerpt = Segment(time_s, time_e)
            embedding = inference.crop(audio_file, excerpt)
            print(np.shape(embedding.data))
            return embedding


class PyannotVAD(Pyannot): 
    ''' voice activity detection  - pytorch.bin 모델 없음 ''' 
    def __init__(self):
        super().__init__()

    def get_vad_timestamp(self, pipeline, audio_file):
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        vad_result = pipeline(audio_in_memory)

        vad_timestamp = [] 
        for speech in vad_result.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))
        return vad_timestamp


class PyannotDIAR(Pyannot):
    def __init__(self):
        super().__init__()
  
    def get_diar_result(self, pipeline, audio_file):
        diarization = pipeline(audio_file)
        diar_result = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start 
            end_time = segment.end
            duration = end_time - start_time 
            if duration >= 0.7:
                diar_result.append([(start_time, end_time), speaker])
        return diar_result