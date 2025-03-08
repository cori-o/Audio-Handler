from pyannote.audio import Pipeline 
from abc import ABC, abstractmethod
import torch 

class PyannotePipe:
    def __init__(self, args):
        self.args = args 
        self.set_gpu()

    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

    def set_pipeline(self, model_name):
        self.pipeline = Pipeline(model_name, use_auth_token=self.args['hf_key'])
        self.pipeline.to(self.device)


class PyannoteVADP(PyannotePipe):
    def __init__(self, args):
        super().__init__(args)

    def get_vad(self, pipeline, audio_file, onset=0.5, offset=0.5, min_duration_on=0.5, min_duration_off=0.5):
        '''
        audio_file = "data_path/tets.wav"
        '''
        hyper_params = {
            "onset": onset, 
            "offset": offset, 
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        }
        pipeline.instantiate(hyper_params)
        output = pipeline(audio_file)

        vad_timestamp = [] 
        for speech in output.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))
        return vad_timestamp
    

class PyannoteDIAR(PyannotePipe):
    def __init__(self, args):
        super().__init__(args)
    
    def get_diar()