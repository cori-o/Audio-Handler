from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from abc import ABC, abstractmethod
from pydub import AudioSegment
from openai import OpenAI
import tempfile
import logging
import torchaudio
import argparse
import torch
import time
import json
import io
import os

class STTModule:
    def __init__(self, openai_api_key=None, ms_api=None):
        self.openai_api_key = openai_api_key
        self.ms_api = ms_api 
    
    @abstractmethod
    def set_client(self):
        pass

    @abstractmethod
    def convert_text_to_speech(self, audio_path, save_path):
        pass 


class WhisperSTT(STTModule):
    def __init__(self, openai_api_key):
        super().__init__(openai_api_key=openai_api_key)
    
    def set_client(self):
        self.openai_client = OpenAI(api_key=self.openai_api_key)

    def set_generate_kwargs(self):
        self.generate_kwargs = {
            "num_beams": 7,
            "condition_on_prev_tokens": False,
            "temperature": 0.0,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.5,
            "language": "ko",
            "return_timestamps": "word",
            "return_dict": True
        }
    
    def set_gpu_device(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def load_word_dictionary(self, json_path):
        with open(json_path, mode='r', encoding='utf-8') as file:
            self.word_dict = json.load(file)    # JSON 데이터를 한번만 로드
            return self.word_dict
    
    def apply_word_dictionary(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = stt_text.replace(incorrect_word, correct_word)
        return stt_text
    
    def apply_word_dictionary_regex(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = re.sub(rf'\b{re.escape(incorrect_word)}\b', correct_word, stt_text)
        return stt_text
    
    def filter_stt_result(self, results):
        filtered_results = []
        prev_text = None
        for segment in results:
            text = segment['text'].strip()
            if text == prev_text:    # 이전 텍스트와 동일하면 제거
                continue
            prev_text = text
            filtered_results.append(segment)
        return filtered_results

    def transcribe_text_api(self, audio_p, audio_file, logger=None, meeting_id=None, table_editor=None, chunk_offset=None):
        if isinstance(audio_file, AudioSegment):
            whisper_audio = audio_file.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_buffer = io.BytesIO()
            whisper_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            audio_file = audio_buffer
        elif isinstance(audio_file, io.BytesIO):
            audio_file = audio_p.bytesio_to_tempfile(audio_file)
        
        audio = AudioSegment.from_file(audio_file)     # 컨텍스트 매니저 제거
        whisper_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        self.load_word_dictionary(os.path.join('./config', 'word_dict.json'))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            whisper_audio.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language='ko',
                    response_format="verbose_json",
                    # timestamp_granularities=["segment"],
                    prompt="The sentence may be cut off. do not make up words to fill in the rest of the sentence." 
                )
        segments = transcription.segments
        print(f'trans result: {transcription.segments}')   # id, avg_logprob, compression_ratio, end, no_speech_prob, seek, start, temperature (0.0), text, tokens
        previous_text = None 
        logs = [] 
        total_text = ""
        for segment in segments:
            segment.text = segment.text.strip()
            stt_log = {
                "start_time": segment.start,
                "end_time": segment.end, 
                "text": segment.text,
                "prob": segment.no_speech_prob,
                "seek": segment.seek, 
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob
            }
            if segment.temperature > 0.1:
                continue
            if segment.no_speech_prob < 0.75 and segment.avg_logprob > -2.0:
                if segment.text == previous_text:
                    continue
                previous_text = segment.text
                modified_text = self.apply_word_dictionary(segment.text, self.word_dict) + " "
                total_text += modified_text 
                segment.start += chunk_offset 
                segment.end += chunk_offset
                    
                stt_result = (segment.start, segment.end, modified_text, 'UNKNOWN')
                logger.info(stt_log)
                logs.append(stt_log)
                if table_editor != None:
                    table_editor.edit_poc_conf_log_tb(task='insert', table_name='ibk_poc_conf_log', data=meeting_id, val=stt_result)
        return logs

    def transcribe_text(self, model_id, audio_file_path):
        model_id = model_id
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            # chunk_length_s=30,
            # batch_size=8,  # batch size for inference - set based on your device
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        result = pipe(audio_file_path, generate_kwargs=self.generate_kwargs, return_timestamps=True)
        print(result.keys())
        print(f"chunk: {result['chunks']}")
        result['text'] = self.apply_word_dictionary(result['text'], self.word_dict)
        return result