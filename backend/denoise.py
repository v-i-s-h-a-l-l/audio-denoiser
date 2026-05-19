import scipy.signal
import soundfile as sf
import numpy as np
from pyrnnoise import RNNoise

INPUT_FILE  = "noise_test4.wav"
RESAMPLED   = "resampled_48k.wav"
OUTPUT_FILE = "clean04.wav"

# Step 1: resample to 48kHz if needed
audio, sr = sf.read(INPUT_FILE, always_2d=False)
print(f"Loaded: sr={sr}, shape={audio.shape}")

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if sr != 48000:
    audio = scipy.signal.resample_poly(audio, 48000, sr)
    sr = 48000
    print("Resampled to 48000 Hz")

sf.write(RESAMPLED, audio, sr, subtype='PCM_16')

# Step 2: denoise using built-in wav method
denoiser = RNNoise(sample_rate=48000)
for speech_prob in denoiser.denoise_wav(RESAMPLED, OUTPUT_FILE):
    pass  # speech_prob per frame available here if you want it

print(f"Done! Saved to {OUTPUT_FILE}")
