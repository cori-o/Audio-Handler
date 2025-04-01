from pydub.effects import high_pass_filter, low_pass_filter
from nsnet2_denoiser import NSnet2Enhancer
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import noisereduce as nr
import soundfile as sf
import numpy as np
import subprocess
import tempfile
import torchaudio
import librosa
import torch
import io
import os 


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

    def separate_vocals_with_demucs(self, audio_file, output_dir='dataset/demucs'):
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        try:
            subprocess.run([
                "demucs",
                "--two-stems", "vocals",
                "--out", output_dir,
                audio_file
            ], check=True, env=env)
            print(f"Separated vocals saved in {output_dir}")
        except subprocess.CalledProcessError as e:
            print("🚨 Demucs 실행 중 오류 발생:", e)

    def denoise_audio(self, audio_file, model_type='nsnet', data_path=None):
        if model_type == 'nsnet':
            enhancer = NSnet2Enhancer(fs=48000)
            sigIn, fs = sf.read(audio_file)
            outSig = enhancer(sigIn, fs)

            audioIn = AudioSegment.from_wav(audio_file)
            audioOut = enhancer.pcm_16le(audioIn.raw_data)
            audio_clean = AudioSegment(
                data=audioOut,
                sample_width=2,         # 16-bit PCM = 2 bytes
                frame_rate=audioIn.frame_rate,
                channels=audioIn.channels
            )
            if data_path: 
                file_name = 'denoised_' + audio_file.split('/')[-1].split('.')[0] + '.wav'
                save_path = os.path.join(data_path, file_name)
                audio_clean.export(save_path, format="wav")
                print(f"✔️ 잡음 제거된 오디오가 저장되었습니다: {save_path}")
        elif model_type == '':
            pass



class VoiceEnhancer:
    '''
    음성 파일에서 음성을 강화한다. 
    '''
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


class AudioVisualizer:
    def __init__(self, n_fft=1024, hop_length=256, n_mels=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def get_waveform(self, y, sr, title_prefix="", file_name=None):
        plt.figure(figsize=(14, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"{title_prefix} - Waveform")
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.close()

    def get_spectrogram(self, y, sr, title_prefix="", file_name=None):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)), ref=np.max)
        plt.figure(figsize=(14, 3))
        librosa.display.specshow(D, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title_prefix} - STFT Spectrogram")
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.close()

    def get_melspectrogram(self, y, sr, title_prefix="", file_name=None):
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(14, 3))
        librosa.display.specshow(S_dB, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title_prefix} - Mel Spectrogram")
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.close()

    def visualize_all(self, y, sr, title_prefix="", file_name=None):
        # 전체를 한 번에 시각화 (waveform + STFT + Mel)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)), ref=np.max)

        plt.figure(figsize=(14, 10))

        # 1. Waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"{title_prefix} - Waveform")

        # 2. STFT Spectrogram
        plt.subplot(3, 1, 2)
        librosa.display.specshow(D, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title_prefix} - STFT Spectrogram")

        # 3. Mel Spectrogram
        plt.subplot(3, 1, 3)
        librosa.display.specshow(S_dB, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title_prefix} - Mel Spectrogram")

        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.close()

    def visualize_before_after_all(self, y_before, sr_before, y_after, sr_after, file_name=None):
        n_fft = 1024
        hop_length = 256
        n_mels = 128

        S_before = librosa.feature.melspectrogram(y=y_before, sr=sr_before, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        S_dB_before = librosa.power_to_db(S_before, ref=np.max)
        D_before = librosa.amplitude_to_db(np.abs(librosa.stft(y_before, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

        S_after = librosa.feature.melspectrogram(y=y_after, sr=sr_after, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        S_dB_after = librosa.power_to_db(S_after, ref=np.max)
        D_after = librosa.amplitude_to_db(np.abs(librosa.stft(y_after, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

        # --- Plot 순서 ---
        plt.figure(figsize=(18, 28))
        
        # 1. Waveform
        plt.subplot(6, 1, 1)
        librosa.display.waveshow(y_before, sr=sr_before)
        plt.title("Before - Waveform", fontsize=14)

        plt.subplot(6, 1, 2)
        librosa.display.waveshow(y_after, sr=sr_after)
        plt.title("After - Waveform", fontsize=14)

        # 2. STFT
        plt.subplot(6, 1, 3)
        librosa.display.specshow(D_before, sr=sr_before, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Before - STFT Spectrogram", fontsize=14)

        plt.subplot(6, 1, 4)
        librosa.display.specshow(D_after, sr=sr_after, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("After - STFT Spectrogram", fontsize=14)

        # 3. Mel
        plt.subplot(6, 1, 5)
        librosa.display.specshow(S_dB_before, sr=sr_before, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Before - Mel Spectrogram", fontsize=14)

        plt.subplot(6, 1, 6)
        librosa.display.specshow(S_dB_after, sr=sr_after, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("After - Mel Spectrogram", fontsize=14)
        plt.subplots_adjust(hspace=0.8)
        plt.tight_layout(pad=3)
        if file_name:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()