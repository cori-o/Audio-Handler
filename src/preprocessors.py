
from scipy.spatial.distance import cdist
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import numpy as np 
import tempfile
import librosa
import wave
import re
import os
import io


class DataProcessor:    
    def cleanse_text(self, text):
        '''
        특수문자("'@) 및 문장 맨 끝 마침표 제거
        '''
        if not isinstance(text, str):
            return text 
        
        text = re.sub(r"[^a-zA-Z0-9가-힣\s.,!?]", "", text)
        if re.fullmatch(r"[\d.]+", text):
            text = text.rstrip(".")
        text = re.sub(r"\s+", " ", text).strip()
        return text 


class VectorProcessor:
    def calc_similarity(self, emb1, emb2, metric):
        distance = cdist(emb1, emb2, metric=metric)[0, 0]
        return distance


class AudioFileProcessor:
    def save_audio(self, audio_file, file_name):
        '''
        audio_file: AudioSegment or BytesIO
        '''
        if isinstance(audio_file, BytesIO):
            audio_file.seek(0)
            audio_file = AudioSegment.from_file(audio_file, format="wav")

        if isinstance(audio_file, AudioSegment):
            audio_file.export(file_name, format="wav")
            print(f"✔️ 오디오가 저장되었습니다: {file_name}")
        else:
            raise TypeError("지원되지 않는 형식입니다: AudioSegment 또는 BytesIO만 지원됩니다.")

    def chunk_audio(self, audio_file, chunk_length=600):
        if isinstance(audio_file, BytesIO):
            audio_file.seek(0)
            audio_file = AudioSegment.from_file(audio_file, format="wav")

        if isinstance(audio_file, AudioSegment):
            chunk_length_ms = chunk_length * 1000
            chunks = [audio_file[i:i + chunk_length_ms] for i in range(0, len(audio_file), chunk_length_ms)]
            return chunks
        else:
            raise TypeError("지원되지 않는 형식입니다: AudioSegment 또는 BytesIO만 지원됩니다.")

    def concat_chunk(self, chunk_list):
        return sum(chunk_list)

    def bytesio_to_tempfile(self, audio_bytesio):
        """
        BytesIO 객체를 임시 파일로 저장
        args:
            audio_bytesio (BytesIO): BytesIO 객체
        returns:
            str: 임시 파일 경로.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytesio.getvalue())
            temp_file.flush()
            return temp_file.name

    def audiofile_to_AudioSeg(self, audio_file):
        if isinstance(audio_file, str):
            audio = AudioSegment.from_wav(audio_file)
        elif isinstance(audio_file, BytesIO):
            audio_file.seek(0)
            audio = AudioSegment.from_file(audio_file, format="wav")
        elif isinstance(audio_file, AudioSegment):
            audio = audio_file
        else:
            raise TypeError("지원되지 않는 입력 타입입니다.")
        return audio
    
    def pcm_to_wav(self, pcm_file_path, wav_file_path, sample_rate=44100, channels=1, bit_depth=16):
        try:
            with open(pcm_file_path, 'rb') as pcm_file:   # PCM 파일 열기
                pcm_data = pcm_file.read()

            with wave.open(wav_file_path, 'wb') as wav_file:   # WAV 파일 생성
                wav_file.setnchannels(channels)   # 채널 수 (1: 모노, 2: 스테레오)
                wav_file.setsampwidth(bit_depth // 8)   # 샘플 당 바이트 수
                wav_file.setframerate(sample_rate)   # 샘플링 속도
                wav_file.writeframes(pcm_data)   # PCM 데이터 쓰기
            print(f"WAV 파일이 성공적으로 생성되었습니다: {wav_file_path}")
        except Exception as e:
            print(f"오류 발생: {e}")

    def m4a_to_wav(self, m4a_path):
        audio_file = AudioSegment.from_file(m4a_path, format='m4a')
        wav_path = m4a_path.replace('m4a', 'wav')
        audio_file.export(wav_path, format='wav')