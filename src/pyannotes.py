from pyannote.audio import Inference
from pyannote.audio import Pipeline
from pyannote.audio import Model 
from pyannote.audio import Audio
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
        random.seed(self.seed)  
        np.random.seed(self.seed)  
        torch.manual_seed(self.seed)  
        torch.cuda.manual_seed_all(self.seed)    # GPU 연산을 위한 시드 설정
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


class PyannotVAD(Pyannot): 
    ''' voice activity detection  - pytorch.bin 모델 없음 ''' 
    def __init__(self):
        super().__init__()

    def get_vad_timestamp(self, pipeline, audio_file):
        if isinstance(audio_file, AudioSegment):
            buffer = BytesIO()
            audio_file.export(buffer, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)
        elif isinstance(audio_file, (str, bytes, os.PathLike)):
            waveform, sample_rate = torchaudio.load(audio_file)
        else:
            raise TypeError("지원되지 않는 오디오 형식입니다.")
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        vad_result = pipeline(audio_in_memory)
        vad_timestamp = []
        for speech in vad_result.get_timeline().support():
            vad_timestamp.append((speech.start, speech.end))
        return vad_timestamp

class PyannotDIAR(Pyannot):
    def __init__(self):
        super().__init__()
  
    def get_diar_result(self, pipeline, audio_file, num_speakers=None, return_embeddings=False):
        diarization = pipeline(audio_file, num_speakers=num_speakers, return_embeddings=return_embeddings)
        diar_result = []
        embeddings = None 
        if return_embeddings == False:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                start_time = segment.start 
                end_time = segment.end
                duration = end_time - start_time 
                if duration >= 0.3:
                    diar_result.append([(start_time, end_time), speaker])
        else:
            embeddings = diarization[1]
            for segment, _, speaker in diarization[0].itertracks(yield_label=True):
                start_time, end_time = segment.start, segment.end 
                duration = end_time - start_time 
                if duration >= 0.3:
                    diar_result.append([(start_time, end_time), speaker])
        return diar_result, embeddings

    def concat_diar_result(self, diar_result, chunk_offset=None):
        total_diar_result = []
        for idx, diar in enumerate(diar_result):
            offset_result = [((start_time + chunk_offset * idx, end_time + chunk_offset * idx), speaker) for (start_time, end_time), speaker in diar]
            total_diar_result.extend(offset_result)
        return total_diar_result

    def resegment_result(self, vad_result, diar_result):
        '''
        resegment diar result using vad result
        diar_result = [[diar result of chunk 1], [diar result of chunk 2], ... ]    (time_s, time_e), 'SPEAKER_00' 
        1. delete speaker which not in vad_result 
        2. add speaker info which in vad_result and not in diar_result   - new speaker: unknown   - skip 
        '''
        non_overlapped_segments = []
        resegmented_diar = [] 
        vad_tree = IntervalTree(Interval(time_s, time_e) for time_s, time_e in vad_result)
        for (time_s, time_e), speaker in diar_result:
            intersections = vad_tree.overlap(time_s, time_e)
            if not intersections:
                non_overlapped_segments.append(((time_s, time_e), speaker))
            for interval in intersections:
                resegmented_diar.append(((time_s, time_e), speaker))
        print(f'non overlapped segment: {non_overlapped_segments}')
        return resegmented_diar 

    def save_as_rttm(self, diar_result, output_rttm_path=None, file_name=None):
        '''
        rttm: SPEAKER <file-id> <channel> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
        '''
        with open(output_rttm_path, "w") as f:
            for (start_time, end_time), speaker in diar_result:
                duration = end_time - start_time
                rttm_line = f"SPEAKER {file_name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
                f.write(rttm_line)

    def save_as_emb(self, embeddings, output_emb_path=None):
        import numpy as np 
        np.save(output_emb_path, embeddings)

    def load_emb_npy(self, npy_emb_path=None):
        if npy_emb_path is None:
            raise ValueError("npy_emb_path must be specified.")
        embeddings = np.load(npy_emb_path)
        return embeddings


class PyannotOSD(Pyannot):   # Overlap Speech Detection
    def __init__(self):
        super().__init__()

    def get_overlapped_result(self, pipeline, audio_file):
        overlap_result = pipeline(audio_file)
        return overlap_result
