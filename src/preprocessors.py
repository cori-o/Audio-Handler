
from scipy.spatial.distance import cdist
from pydub import AudioSegment
import soundfile as sf
import numpy as np 
import tempfile
import librosa
import wave
import re
import os

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
    def align_audio(self, reference_file, target_file, output_file):
        lag, sr = self.calculate_time_lag(reference_file, target_file)   # 시간차 계산
        target, _ = librosa.load(target_file, sr=sr)   # 오디오 로드
        aligned_target = np.pad(target, (lag, 0), mode='constant') if lag > 0 else target[-lag:]
        sf.write(output_file, aligned_target, sr)     # 정렬된 오디오 저장
        print(f"Aligned audio saved to {output_file}")\

    def chunk_audio(self, audio_file_path, chunk_length=60, chunk_file_path=None, chunk_file_name=None):
        audio = AudioSegment.from_file(audio_file_path)
        chunk_length_ms = chunk_length * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        if chunk_file_path:
            for idx, chunk in enumerate(chunks):
                temp_file_path = os.path.join(chunk_file_path, f"{chunk_file_name}_{idx}.wav")
                chunk.export(temp_file_path, format="wav")
        else:
            return chunks
    
    def concat_chunk(self, chunk_list, save_path=None):
        final_audio = sum(chunk_list)
        if save_path:    
            final_audio.export("processed_audio.wav", format="wav")
        else:
            return final_audio

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