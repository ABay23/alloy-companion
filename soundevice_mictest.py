import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import whisper

duration = 5  # seconds
sample_rate = 16000  # Whisper prefers 16kHz

print("🎤 Recording from default mic...")
audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print("✅ Recording complete.")

# Save as .wav
scipy.io.wavfile.write("test_audio.wav", sample_rate, audio)

# Load Whisper model and transcribe
model = whisper.load_model("base")
result = model.transcribe("test_audio.wav")
print("🧠 Whisper recognized:", result["text"])

