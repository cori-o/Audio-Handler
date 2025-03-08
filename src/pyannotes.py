from pyannote.audio import Inference
from pyannote.audio import Model 
from pyanntoe.core import Segment
import numpy as np 
import torch 

class Pyannot:
    def __init__(self, args):
        self.args = args 
        self.set_gpu()
    
    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    
    def set_model(self, model_name):
        self.model = Model.from_pretrained(model_name, use_auth_token=self.args['hf_key'])
    

class PyannotEMB(Pyannot):
    def __init__(self, args):
        super().__init__(args)

    def set_inference(self, model, window, duration=None, step=None):
        '''
        window = whole : 
        window = sliding : 
        '''
        if window == 'whole':
            self.inference = Inference(model, window=window)
            self.inference.to(self.device)
        elif window == 'sliding':
            self.inference = Inference(model, window=window, duration=duration, step=step)
            self.inference.to(self.device)
    
    def get_embedding(self, inference, audio_file, time_s=None, time_e=None):
        '''
        audio_file: test.wav
        '''
        if time_s != None: 
            embedding = inference(audio_file)
            print(np.shape(embedding.data))
            return embedding
        else:
            excerpt = Segment(time_s, time_e)
            embedding = inference.crop(audio_file, excerpt)
            print(np.shape(embedding.data))
            return embedding