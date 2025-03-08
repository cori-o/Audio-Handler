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

    @abstractmethod 
    def set_pipe_config(self, **kwargs):
        pass 

class PyannoteVADP(PyannotePipe):
    def __init__(self, args):
        super().__init__(args)
    
    def set_pipe_config(self, pipeline, onset=None, offset=None):
        '''
        onset: 감지 민감도 증가 
        offset: 감지 지속시간 조정 
        '''
        pipeline.instantiate({
            "onset": onset,  
            "offset": offset, 
        })
        return pipeline
    
    def get_vad(self, pipeline, audio_file, onset=0.5, offset=0.5):
        '''
        audio_file = "data_path/tets.wav"
        '''
        vad_timestamp = [] 
        pipeline = self.set_pipe_config(pipeline, onset, offset)
        output = pipeline(audio_file)
        for speech in output.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))