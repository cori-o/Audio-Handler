from scipy.io import wavfile
import noisereduce as nr

# load data
rate, data = wavfile.read("./dataset/chunk/chunk_20250211_0.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("noisereduce_chunk_20250211_0.wav", rate, reduced_noise)