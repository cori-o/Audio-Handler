import matplotlib.pyplot as plt

def plot_timestamps(whisper_chunks, vad_segments):
    whisper_starts = [chunk["timestamp"][0] for chunk in whisper_chunks]
    vad_starts = [seg[0] for seg in vad_segments]

    plt.figure(figsize=(10, 4))
    plt.scatter(whisper_starts, [1] * len(whisper_starts), label="Whisper", color="red")
    plt.scatter(vad_starts, [0] * len(vad_starts), label="VAD", color="blue")
    plt.xlabel("Time (seconds)")
    plt.yticks([0, 1], ["VAD", "Whisper"])
    plt.legend()
    plt.title("Whisper vs VAD Timestamps")
    plt.show()

# ✅ Whisper vs VAD 비교 시각화
plot_timestamps(shifted_results, vad_segments)