import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=2)  # MacBook Pro Mic

with mic as source:
    print("🎛️ Calibrating for ambient noise...")
    try:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Speak now...")
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        print("📡 Got audio! Transcribing...")
        text = recognizer.recognize_whisper(audio, model="base")
        print("🧠 Whisper recognized:", text)
    except Exception as e:
        print("❌ Error during mic capture:", e)

