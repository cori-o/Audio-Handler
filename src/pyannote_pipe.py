from pyannote.audio.pipelines import Resegmentation
from pyannote.audio import Pipeline 
from pyannote.audio import Model
import torch 


class PyannotePipe:
    def __init__(self, config):
        self.config = config 
        self.set_gpu()

    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

    def set_pipeline(self, model_name):
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=self.config['hf_key'])
        self.pipeline.to(self.device)
        print(f'pipeline gpu 설정 완료')


class PyannoteVADP(PyannotePipe):
    def __init__(self, config):
        super().__init__(config)
        self.set_pipeline(self.config['vad_model'])

    def get_vad(self, audio_file, onset=0.5, offset=0.5, min_duration_on=0.5, min_duration_off=0.5):
        '''
        audio_file = "data_path/tets.wav"
        '''
        hyper_params = {
            "onset": onset, 
            "offset": offset, 
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        }
        self.pipeline.instantiate(hyper_params)
        output = self.pipeline(audio_file)
        # print(f'output: {output}')
        vad_timestamp = [] 
        for speech in output.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))
        return vad_timestamp
    

class PyannoteDIARP(PyannotePipe):
    def __init__(self, config):
        super().__init__(config)
        self.set_pipeline(self.config['diarization_model'])
    
    def get_diar(self, audio_file, duration_thresh=0.7):
        output = self.pipeline(audio_file)
        diar_result = [] 
        for segment, _, speaker in output.itertracks(yield_label=True):
            start_time = segment.start 
            end_time = segment.end
            duration = end_time - start_time 
            if duration >= duration_thresh:
                diar_result.append([(start_time, end_time), speaker])
        return diar_result
    