from pyannote.audio import Inference
from pyannote.audio import Model 
from pyannote.core import Segment
import numpy as np 
import torch
import os  

class Pyannot:
    def __init__(self, config):
        self.config = config 
        self.set_gpu()
    
    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    
    def set_model(self, model_name):
        self.model = Model.from_pretrained(model_name, use_auth_token=self.config['hf_key'], force_download=False)
    

class PyannotEMB(Pyannot):
    def __init__(self, config):
        super().__init__(config)
        # self.set_model(self.config['emb_model'])

    def set_config(self):
        pass 

    def set_inference(self, window, duration=None, step=None, model_loc='local'):
        '''
        window = whole : 
        window = sliding : 
        '''
        if model_loc == 'hub':
            if window == 'whole':
                self.inference = Inference(self.config['emb_model'], window=window, device=self.device)
            elif window == 'sliding':
                self.inference = Inference(self.config['emb_model'], window=window, duration=duration, step=step, device=self.device)
        elif model_loc == 'local':
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
    def __init__(self, config):
        super().__init__(config)

    def set_inference(self, model_loc='local'):
        if model_loc == 'hub':
            self.inference = Inference(self.config['voice_activity_detection'], device=self.device)
        elif model_loc == 'local':
            self.inference = Inference(os.path.join(self.config['local_model_path'], 'segment'), device=self.device)
    
    def set_config(self):
        pass 

    def get_vad_timestamp(self, inference, audio_file):
        output = inference(audio_file)
        vad_timestamp = [(speech.start, speech.end) for speech in output.get_timeline().support()]
        return vad_timestamp


class PyannotDIAR(Pyannot):
    def __init__(self, args):
        super().__init__(args)

    def set_inference(self, model_loc='local'):
        if model_loc == 'hub':
            self.inference = Inference(self.config['diarization_model'], device=self.device)
        elif model_loc == 'local':
            self.inference = Inference(os.path.join(self.config['local_model_path'], 'diar'), device=self.device)
    
    def set_config(self):
        pass 

    def get_diar_result(self, inference, audio_file):
        output = inference(audio_file)
        diar_result = [] 
        for segment, _, speaker in output.itertracks(yield_label=True):
            start_time = segment.start 
            end_time = segment.end
            duration = end_time - start_time 
            if duration >= duration_thresh:
                diar_result.append([(start_time, end_time), speaker])
        return diar_result
    