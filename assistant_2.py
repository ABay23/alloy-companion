import base64
import sounddevice as sd
import scipy.io.wavfile
import whisper
import cv2
from threading import Lock, Thread
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
from pyaudio import PyAudio, paInt16

load_dotenv()

# ----------------- Webcam Stream -----------------
class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            with self.lock:
                self.frame = frame

    # def read(self, encode=False):
    #     with self.lock:
    #         frame = self.frame.copy()
    #     if encode:
    #         _, buffer = cv2.imencode(".jpeg", frame)
    #         return base64.b64encode(buffer)
    #     return frame
    
    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        if frame is None:
            return None
        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame


    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()


# ----------------- Assistant Logic -----------------
class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    # def answer(self, prompt, image):
    #     if not prompt:
    #         print("🛑 No prompt provided.")
    #         return
    #     print(f"🧠 Prompt: {prompt}")
    #     response = self.chain.invoke(
    #         {"prompt": prompt, "image_base64": image.decode()},
    #         config={"configurable": {"session_id": "unused"}}
    #     ).strip()
    #     print("🤖 Assistant:", response)
    #     return response
    
    def answer(self, prompt, image):
        if not prompt:
            print("🛑 No prompt provided.")
            return
        print(f"🧠 Prompt: {prompt}")
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}}
        ).strip()
        print("🤖 Assistant:", response)
        self._tts(response)

    
    # Inside Assistant class
    def _tts(self, response_text):
        print("🗣️ Speaking...")

        player = PyAudio().open(
            format=paInt16,
            channels=1,
            rate=24000,
            output=True
        )

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response_text
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

        player.stop_stream()
        player.close()

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Be short and helpful, show personality.
        """

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                [
                    {"type": "text", "text": "{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{image_base64}",
                    },
                ],
            ),
        ])

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()

        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history"
        )

def show_webcam_feed(stream: WebcamStream):
    while stream.running:
        frame = stream.read()

        if frame is None:
            print("⚠️ No frame received from webcam.")
            continue

        cv2.imshow("📷 Webcam View", frame)

        if cv2.waitKey(1) in [27, ord("q")]:
            stream.stop()
            break

    cv2.destroyAllWindows()


# ----------------- Voice + Vision Flow -----------------
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    scipy.io.wavfile.write(filename, samplerate, audio)
    print("✅ Audio recorded.")

def transcribe_audio(filename="input.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

# ----------------- Main -----------------
if __name__ == "__main__":
    webcam_stream = WebcamStream().start()
    model = ChatOpenAI(model="gpt-4o")
    assistant = Assistant(model)
    
    # 👇 Launch webcam window in its own thread
    from threading import Thread
    Thread(target=show_webcam_feed, args=(webcam_stream,), daemon=True).start()

    try:
        print("\n👂 Press [Enter] to speak, or type 'q' + [Enter] to quit.")
        # while True:
        #     user_input = input("🎤 Ready to listen? ")
        #     if user_input.lower() == "q":
        #         break

        #     record_audio()
        #     prompt = transcribe_audio()
        #     img = webcam_stream.read(encode=True)
        #     assistant.answer(prompt, img)
        
        while True:
            user_input = input("🎤 Press [Enter] to speak or type 'q' to quit: ")
            if user_input.lower() == "q":
                break

            record_audio()
            prompt = transcribe_audio()
            img = webcam_stream.read(encode=True)
            assistant.answer(prompt, img)

            # 👇 Show the webcam frame
            frame = webcam_stream.read()
            cv2.imshow("📷 Webcam View", frame)

            # ⌨️ Close with 'q' or Esc key
            if cv2.waitKey(1) in [27, ord("q")]:
                break


    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()
        print("🛑 Session ended.")
