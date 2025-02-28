from pydub.effects import high_pass_filter, low_pass_filter
from pydub.silence import detect_nonsilent
from pyannote.audio import Pipeline
from collections import Counter
from pydub import AudioSegment
import pyloudnorm as pyln
import noisereduce as nr
import soundfile as sf
import numpy as np 
import subprocess
import tempfile
import librosa
import torch
import wave
import io
import re
import os

class DataProcessor:
    def flatt_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):  # 리스트 내부의 리스트 처리
                flat_list.extend(self.flatt_list(item))
            else:
                flat_list.append(item)
        return flat_list
    
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

    def remove_repeated_patterns(self, text):
        """
        - 한 단어 내에서 반복되는 패턴 제거 (예: "네네네네네" → "네")
        - 연속된 단어도 하나만 남기기 (예: "네 네 네 네" → "네")
        """
        text = re.sub(r"(.)\1{2,}", r"\1", text)  # 같은 글자가 3번 이상 반복되면 1개로 축소
        words = text.split()
        cleaned_words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i - 1]]
        return " ".join(cleaned_words) if cleaned_words else None


class TimeProcessor:
    def is_similar(self, diar_seg, stt_result):
        '''
        두 세그먼트 간 겹치는 길이와 발화 시간이 유사한지 검사
        stt_result: (conv_id, content, start, end, cuser_id, conf_id)
        '''
        TIME_TOLERANCE = 1.5   # 허용 오차(초)
        diar_start, diar_end = diar_seg['start'], diar_seg['end']
        # stt_start, stt_end = stt_result[2], stt_result[3]
        stt_start, stt_end = stt_result['start_time'], stt_result['end_time']

        diar_duration = diar_end - diar_start
        stt_duration = stt_end - stt_start
    
        # 겹치는 구간 계산
        overlap_start = max(diar_start, stt_start)
        overlap_end = min(diar_end, stt_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        print(f'overlap duration: {overlap_duration}, abs: {abs(diar_duration - stt_duration)}')

        if overlap_duration > 1 and abs(diar_duration - stt_duration) < TIME_TOLERANCE:
            return True
        else:
            return False


class AudioFileProcessor:
    def align_audio(self, reference_file, target_file, output_file):
        lag, sr = self.calculate_time_lag(reference_file, target_file)   # 시간차 계산
        target, _ = librosa.load(target_file, sr=sr)   # 오디오 로드
        aligned_target = np.pad(target, (lag, 0), mode='constant') if lag > 0 else target[-lag:]
        sf.write(output_file, aligned_target, sr)     # 정렬된 오디오 저장
        print(f"Aligned audio saved to {output_file}")

    def calculate_time_lag(self, reference_file, target_file):
        ref, sr_ref = librosa.load(reference_file, sr=None)     # 샘플링 속도 유지, 기준 오디오 
        target, sr_target = librosa.load(target_file, sr=None)    # 타겟 오디오 로드

        if sr_ref != sr_target:   # 샘플링 속도 확인 및 일치
            raise ValueError("Sample rates of the audio files must be the same.")
        correlation = np.correlate(target, ref, mode='full')   # Cross-Correlation 계산
        lag = np.argmax(correlation) - len(ref)
        return lag, sr_ref

    def audio_chunk(self, audio_file_path, chunk_length=60, chunk_file_path=None, chunk_file_name=None):
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

    def calculate_nonsilent_start(self, audio_file_path, silence_thresh=-30, min_silence_len=500):
        """
        Calculate the start time of the first non-silent portion in an audio file.
        args:
            audio_file_path (str): Path to the audio file.
            silence_thresh (int): Silence threshold in dBFS.
            min_silence_len (int): Minimum silence length in ms.   
        returns:
            float: Start time of the first non-silent portion in seconds.
        """
        audio = AudioSegment.from_file(audio_file_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        start_time_ms = nonsilent_ranges[0][0] if nonsilent_ranges else 0
        return start_time_ms / 1000  # Convert ms to seconds
    

class NoiseHandler: 
    '''
    음성 파일에서 노이즈를 제거한다.
    '''
    def remove_background_noise(self, input_file, output_file=None, prop_decrease=None):
        """
        배경 잡음을 제거하여 가까운 목소리를 강조
        args:
            input_file (str): 입력 오디오 파일.
            output_file (str): 출력 오디오 파일.
        """
        try:
            y, sr = librosa.load(input_file, sr=None)   # 오디오 로드
        except:
            audio_buffer = io.BytesIO()
            input_file.export(audio_buffer, format="wav")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            y, sr = librosa.load(audio_buffer, sr=None)
        # noise_profile = y[:sr]   # 배경 잡음 프로파일 추출 (초기 1초)
        # y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, prop_decreaese=prop_decrease)   # 잡음 제거
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        if output_file: 
            sf.write(output_file, y_denoised, sr)
            print(f"Saved denoised audio to {output_file}")
        
        wav_buffer = io.BytesIO()   # 메모리 내 WAV 파일 생성
        sf.write(wav_buffer, y_denoised, sr, format='WAV')
        wav_buffer.seek(0)   # 파일 포인터를 처음으로 이동
        return wav_buffer

    def filter_audio_with_pydub(self, input_file, high_cutoff=200, low_cutoff=3000, output_file=None):
        '''
        pydub (python)을 이용한 오디오 필터링 (고역대, 저역대)
        '''
        try:
            audio = AudioSegment.from_file(input_file)
        except:
            audio = input_file
        filtered_audio = high_pass_filter(audio, cutoff=high_cutoff)    # 고역 필터 (200Hz 이상)
        filtered_audio = low_pass_filter(filtered_audio, cutoff=low_cutoff)    # 저역 필터 (3000Hz 이하)
        if output_file:
            filtered_audio.export(output_file, format="wav")
            print(f"Filtered audio saved to {output_file}")
        else:
            audio_buffer = io.BytesIO()
            filtered_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            return audio_buffer

    def filter_audio_with_ffmpeg(self, input_file, high_cutoff=100, low_cutoff=3500, output_file=None):
        """
        FFmpeg을 사용한 오디오 필터링 (고역대, 저역대).
        Args:
            input_file (str or BytesIO): 입력 오디오 파일 경로 또는 BytesIO 객체.
            high_cutoff (int): 고역 필터 컷오프 주파수 (Hz).
            low_cutoff (int): 저역 필터 컷오프 주파수 (Hz).
            output_file (str, optional): 필터링된 오디오 저장 경로. 지정되지 않으면 메모리로 반환.
        Returns:
            io.BytesIO: 필터링된 오디오 데이터 (output_file이 None인 경우).
        """
        input_source = None   # 변수 초기화
        temp_files = []   # 임시 파일을 저장할 리스트
        try:
            if isinstance(input_file, AudioSegment):   # AudioSegment 객체 처리
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    input_file.export(temp_input, format="wav")   # AudioSegment -> WAV 변환
                    temp_input.flush()
                    input_source = temp_input.name
                    temp_files.append(temp_input.name)   # 임시 파일 관리
            elif isinstance(input_file, io.BytesIO):   # BytesIO 객체 처리
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    temp_input.write(input_file.getvalue())
                    temp_input.flush()
                    input_source = temp_input.name
                    temp_files.append(temp_input.name)   # 임시 파일 관리
            elif isinstance(input_file, (str, os.PathLike)):   # 파일 경로 처리
                input_source = input_file
            else:
                raise ValueError("Invalid input_file type. Must be AudioSegment, file path, or BytesIO object.")

            if input_source is None:
                raise RuntimeError("Failed to determine input source.")

            command = [   # FFmpeg 명령 실행
                "ffmpeg",
                "-i", input_source,  # 입력 파일
                "-af", f"highpass=f={high_cutoff},lowpass=f={low_cutoff}",  # 필터 적용
                "-f", "wav",  # 출력 형식
                "pipe:1" if not output_file else output_file  # 메모리로 반환하거나 파일로 저장
            ]

            if output_file:
                subprocess.run(command, check=True)
                print(f"Filtered audio saved to {output_file}")
            else:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
                return io.BytesIO(stdout)  # BytesIO로 반환
        finally:
            for temp_file in temp_files:   # 임시 파일 삭제
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class VoiceEnhancer:
    '''
    음성 파일에서 음성을 강화한다. 
    '''
    def amplify_volume(self, audio, target_db=-20, output_file="amplified_output.wav"):
        """
        볼륨 증폭 및 파일 저장
        Args:
            audio (AudioSegment): Pydub AudioSegment 객체
            target_db (int): 목표 음압 레벨 (dBFS)
            output_file (str): 저장할 파일 경로
        Returns:
            AudioSegment: 증폭된 AudioSegment 객체
        """
        # 현재 음압 계산 및 증폭량 적용
        current_db = audio.dBFS
        difference = target_db - current_db
        amplified_audio = audio.apply_gain(difference)

        amplified_audio.export(output_file, format="wav")
        print(f"Amplified audio saved to {output_file}")
        return amplified_audio

    def emphasize_nearby_voice(self, input_file, threshold=0.05, output_file=None):
        """
        가까운 음성을 강조하고 먼 목소리를 줄임
        args:
            input_file (str): 입력 오디오 파일
            output_file (str): 출력 오디오 파일
            threshold (float): 에너지 기준값 (낮을수록 약한 신호 제거)
        """
        try:
            y, sr = librosa.load(input_file, sr=None)   # 오디오 로드
        except:
            audio_buffer = io.BytesIO()
            input_file.export(audio_buffer, format="wav")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            y, sr = librosa.load(audio_buffer, sr=None)           
        rms = librosa.feature.rms(y=y)[0]         # RMS 에너지 계산
        mask = rms > threshold                    # 에너지 기준으로 마스크 생성

        expanded_mask = np.repeat(mask, len(y) // len(mask) + 1)[:len(y)]   # RMS 값을 전체 신호 길이에 맞게 확장
        y_filtered = y * expanded_mask.astype(float)   # 입력 신호에 확장된 마스크 적용

        if output_file:
            sf.write(output_file, y_filtered, sr)   # 강조된 오디오 저장
            print(f"Saved emphasized audio to {output_file}")
        else:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, y_filtered, sr, format="WAV")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            return audio_buffer

    def normalize_audio_pydub(self, audio_file_path, audio_file_name, target_dbfs=-14):
        audio = AudioSegment.from_file(os.path.join(audio_file_path, audio_file_name))
        
        # RMS(root mean square) 기반 볼륨 정규화
        normalized_audio = audio.apply_gain(-audio.dBFS)

        # 정규화된 오디오 저장
        output_file = os.path.join(audio_file_path, "pydub_nr_" + audio_file_name)
        normalized_audio.export(output_file, format="wav")
        return output_file

    def normalize_audio_lufs(self, audio_input, target_lufs=-14.0, output_file=None):
        """
        LUFS 기반 오디오 정규화
        """
        #print(audio_input)
        if isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            data, rate = sf.read(audio_input)
        else:
            data, rate = sf.read(audio_input)

        # 현재 LUFS 계산 및 정규화
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)

        if output_file:
            sf.write(output_file, loudness_normalized_audio, rate)
            print(f"Saved normalized audio to {output_file}")
            return output_file
        else:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, loudness_normalized_audio, rate, format='WAV')
            wav_buffer.seek(0)
            return wav_buffer


class VoiceSeperator:
    '''
    음성 파일에서 vocal, drum, base 등의 소리 분리
    '''
    def separate_vocals_with_demucs(self, input_file, output_dir):
        subprocess.run([
            "demucs",
            "--two-stems", "vocals",  # 보컬만 분리
            "--out", output_dir,     # 출력 경로
            input_file       # 입력 파일
        ], check=True)
        print(f"Separated vocals saved in {output_dir}")

    def seperate_vocals_with_umix(self, input_file, output_file):
        subprocess.run([
            "umx",
            "--outdir", output_file,
            input_file
        ], check=True)
        print(f"Seperated vocals save in {output_file}")


class SpeakerDiarizer:
    def set_pyannotate(self, hf_api):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_api = hf_api 
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=self.hf_api)
        self.pipeline.to(self.device)

    def rename_speaker(self, result, num_speakers):
        """
        화자 재정렬 및 번호 재매핑:
        1. 발화량 기준으로 화자 번호 부여.
        2. 평균 발화 시간이 짧은 화자는 뒤로 배치.
        """
        speaker_counts = Counter(entry['speaker'] for entry in result)
        speaker_durations = {}
        for speaker in speaker_counts:
            total_time, avg_time = self.calc_speak_duration(result, speaker)
            speaker_durations[speaker] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "count": speaker_counts[speaker]
            }
        print(f"Speaker Durations: {speaker_durations}")
        sorted_speakers = sorted(
        speaker_durations.keys(),
            key=lambda spk: (
                -speaker_durations[spk]['total_time'],  # 총 발화 시간 내림차순
                -speaker_durations[spk]['avg_time'],    # 평균 발화 시간 내림차순
                -speaker_durations[spk]['count']       # 발화 수 내림차순
            )
        )
        long_duration_speakers = [
            spk for spk in sorted_speakers if speaker_durations[spk]['avg_time'] >= 3
        ]
        short_duration_speakers = [
            spk for spk in sorted_speakers if speaker_durations[spk]['avg_time'] < 3
        ]   
        final_speakers = long_duration_speakers + short_duration_speakers   # 평균 발화 시간이 짧은 화자를 뒤로 배치 
        print(f"Final Speakers Order: {final_speakers}")

        final_speaker_mapping = {
            old_speaker: f"SPEAKER_{i:02d}" for i, old_speaker in enumerate(final_speakers)
        }
        filtered_result = []
        for entry in result:
            if entry['speaker'] not in final_speaker_mapping:
                continue
            mapped_speaker = final_speaker_mapping[entry['speaker']]
            if int(mapped_speaker.split('_')[-1]) >= num_speakers:
                continue
            entry['speaker'] = mapped_speaker
            filtered_result.append(entry)
        return filtered_result
    
    def filter_speaker_segments(self, segments):
        '''
        겹치는 발화 제거 (0~10, 3~5 -> 3~5 제거)
        '''
        filtered_segments = []
        for i, seg in enumerate(segments):
            if i > 0 and seg["speaker"] != segments[i - 1]["speaker"]:   # 이전 발화와 확인
                prev_seg = segments[i - 1]
                if prev_seg["start"] <= seg["start"] and seg["end"] <= prev_seg["end"]:
                    continue
            filtered_segments.append(seg)
        return filtered_segments
    
    def calc_speak_duration(self, segments, speaker):
        speak_time = 0; cnt = 0
        for seg in segments:
            if seg['speaker'] == speaker:
                speak_duration = seg['end'] - seg['start']
                speak_time += speak_duration 
                cnt += 1
        return speak_time, round((speak_time / cnt), 2)

    def convert_segments(self, result):
        """
        화자 분리 결과를 적절한 형식으로 변환.
        args:
            result (tuple): Pyannote diarization 결과.
        returns:
            dict: 변환된 결과.
        """
        segment, _, speaker = result
        return {
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        }

    def merge_diarization_segments_with_priority(self, diar_segments):
        """Diarization 구간 병합 시 추임새와 긴 발화 분리"""
        merged_segments = []
        FILLER_THRESHOLD = 2.0   # 추임새로 간주할 최대 길이 (초)
        for seg in diar_segments:
            if not merged_segments:
                merged_segments.append(seg)
                continue
            last_seg = merged_segments[-1]

            # 화자가 다르고, 추임새로 간주되는 경우 (추임새는 긴 발화 구간에 병합하지 않음)
            if last_seg['speaker'] != seg['speaker'] and (seg['end'] - seg['start']) <= FILLER_THRESHOLD:
                merged_segments.append(seg)
            elif last_seg['speaker'] == seg['speaker'] and last_seg['end'] >= seg['start']:
                last_seg['end'] = max(last_seg['end'], seg['end'])   # 같은 화자의 연속된 발화 병합
            else:
                merged_segments.append(seg)   # 새로운 화자 구간 추가
        return merged_segments
            
    def seperate_speakers(self, audio_p, audio_file, local=True, num_speakers=None, save_path=None, file_name=None):
        """
        화자 분리 실행 및 결과 저장.
        args:
            audio_file (str or BytesIO): 입력 오디오 파일 경로 또는 BytesIO 객체.
            save_path (str): 결과 저장 경로.
            file_name (str): 저장할 파일 이름.
            num_speakers (int, optional): 예상 화자 수. 기본값은 Pyannote에서 자동 감지.
        """
        if isinstance(audio_file, io.BytesIO):   # 입력 데이터 형식 확인 및 변환
            audio_file = audio_p.bytesio_to_tempfile(audio_file)

        results = []
        if local:
            print(f'start diarization')
            diarization = self.pipeline(audio_file, num_speakers=None)
            for result in diarization.itertracks(yield_label=True):   # result: (<Segment>, _, speaker)
                converted_info = self.convert_segments(result)
                results.append(converted_info)
        else:
            pass
        # filtered_result = self.filter_speaker_segments(results)
        # merged_result = self.merge_diarization_segments_with_priority(filtered_result)
        diar_result = self.rename_speaker(results, num_speakers)
        return diar_result


class ResultMapper:
    def map_speaker_with_nested_check(self, time_p, stt_result, diar_segments, meeting_id=None, table_editor=None):
        """ stt_result: conv_id, start_time, end_time, text, conf_id """
        if not meeting_id: 
            pass
        #    conv_id = stt_result[0]
        print(f'stt_segment: {stt_result}')
        stt_start, stt_end = stt_result['start_time'], stt_result['end_time']
        candidates = []
        for diar_seg in diar_segments:   # STT 결과값과 겹치는 Diar 구간 탐색 
            diar_start, diar_end = diar_seg['start'], diar_seg['end']
            if stt_start <= diar_end and stt_end >= diar_start:  # 겹침 조건
                candidates.append(diar_seg)

        if not candidates:
            print("No overlapping segments found.")
            return "Unknown"   # DB 업데이트 x 

        max_overlap = 0
        best_speaker = None 
        for candidate in candidates:
            diar_start, diar_end = candidate['start'], candidate['end']
            nested_segments = [     #  Diar 구간 내에 또 다른 구간 탐색 (0~19 -> 13~14)
                seg for seg in diar_segments
                if seg['start'] >= diar_start and seg['end'] <= diar_end and seg != candidate
            ]
            if nested_segments:   # 또 다른 발화가 있을 때 
                for nested in nested_segments:
                    if time_p.is_similar(nested, stt_result):
                        if meeting_id == None: 
                            stt_result['speaker'] = nested['speaker']
                            return nested['speaker']
                        else:
                            updated_result = (conv_id, nested['speaker'])
                            table_editor.edit_poc_conf_log_tb('update', 'ibk_poc_conf_log', data=meeting_id, val=updated_result)
                            return
            # 또 다른 발화가 없을 때 (겹치는 후보군들만 탐색)
            overlap_start = max(stt_start, diar_start)
            overlap_end = min(stt_end, diar_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration 
                best_speaker = candidate['speaker']
        if best_speaker:
            '''if not meeting_id:
                updated_result = (conv_id, best_speaker) 
                table_editor.edit_poc_conf_log_tb('update', 'ibk_poc_conf_log', data=meeting_id, val=updated_result)'''
            return best_speaker