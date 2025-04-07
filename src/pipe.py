from src import NoiseHandler, AudioFileProcessor
from src import PyannotDIAR
from abc import abstractmethod
from pydub import AudioSegment
from io import BytesIO


class BasePipeline:
    @abstractmethod
    def set_env(self):
        pass 


class FrontendProcessor(BasePipeline):
    def set_env(self):
        self.noise_handler = NoiseHandler()
        self.audio_file_processor = AudioFileProcessor()

    def process_audio(self, audio_file, fade_ms=200, chunk_length=600, deverve=False):
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunk_ms = chunk_length * 1000 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=chunk_length)
        print(f"[DEBUG] Chunk count: {len(chunks)}, chunk_length={chunk_length} sec")
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_io = BytesIO()
            chunk.export(chunk_io, format='wav')
            chunk_io.seek(0)
            denoised = self.noise_handler.denoise_audio(chunk_io)
            print(f"[DEBUG] Denoised chunk duration: {len(chunk) / 1000} sec")
            if deverve == True:             
                clean_chunk = self.noise_handler.deverve_audio(denoised)
                clean_chunk.seek(0)
                seg = self.audio_file_processor.audiofile_to_AudioSeg(clean_chunk)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            else:
                seg = self.audio_file_processor.audiofile_to_AudioSeg(denoised)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            processed_chunks.append(seg)
        clean_audio = self.audio_file_processor.concat_chunk(processed_chunks)
        return clean_audio

    def save_audio(self, audio_file, file_name=None):
        self.audio_file_processor.save_audio(audio_file, file_name=file_name)
             

class SpeechActivityDetector(BasePipeline):
    pass 

class SegmentManager(BasePipeline):
    pass
