import torch
import torchaudio
import matplotlib.pyplot as plt

audio_file = "./dataset/chunk/chunk_20250220_0.wav"
signal, fs = torchaudio.load(audio_file)
signal = signal.squeeze()
time = torch.linspace(0, signal.shape[0]/fs, steps=signal.shape[0])

# plt.plot(time, signal)
# plt.show()

from speechbrain.inference.VAD import VAD

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
boundaries = VAD.get_speech_segments(audio_file)
VAD.save_boundaries(boundaries)
prob_chunks = VAD.get_speech_prob_file(audio_file)
plt.plot(prob_chunks.squeeze())
plt.show()