import asyncio
import warnings
import os
import pyaudio
import wave
import subprocess
from dotenv import load_dotenv, find_dotenv
import requests
from qdrant_client import QdrantClient
from llama_index.llms.cohere import Cohere
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from faster_whisper import WhisperModel  

# Load environment variables from .env file
load_dotenv(find_dotenv())

api_key = os.getenv("COHERE_API_KEY")
warnings.filterwarnings("ignore")

class LanguageModelProcessor:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        self._llm = Cohere(model="command-r-plus", api_key=api_key)
        self._service_context = ServiceContext.from_defaults(llm=self._llm)
        self._index = None
        if self._create_kb():
            self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=["/Users/hassn-/Desktop/مستمع-النسخة النهائية/mustma-code/menu.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="restaurant_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("تم إنشاء قاعدة المعرفة بنجاح!")
            return True
        except FileNotFoundError:
            print("حدث خطأ: الملف المحدد غير موجود.")
            return False
        except Exception as e:
            print(f"حدث خطأ أثناء إنشاء قاعدة المعرفة: {e}")
            return False

    def process(self, text):
        if not self._chat_engine:
            return "عذرًا، النظام غير جاهز في الوقت الحالي."
        AgentChatResponse = self._chat_engine.chat(text)
        answer = AgentChatResponse.response
        print(f"LLM: {answer}")
        return answer

    @property
    def _prompt(self):
        return """أنت مساعد AI محترف تعمل كموظف استقبال في أحد أفضل المطاعم في الرياض. مهمتك هي مساعدة العملاء في تقديم طلباتهم والإجابة على استفساراتهم باللغة العربية الفصحى.

### التعليمات:
1. اسأل عن الاسم ورقم الاتصال.
2. اسأل عن تفاصيل الطلب.
3. أكد الطلب وقدم التكلفة النهائية.
4. أنهِ المحادثة بالتحيات.

### نصائح:
- اسأل سؤالاً واحداً في كل مرة.
- إذا لم تعرف الإجابة، قل أنك لا تعرف.
- قدم إجابات قصيرة (لا تزيد عن 10 كلمات).
- لا تتحدث مع نفسك.

### خطوات المحادثة:
1. "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
2. "ما هو اسمك ورقم الاتصال؟"
3. "ما هي طلباتك بالتفصيل؟"
4. "شكراً. التكلفة الإجمالية هي ___ ريال."
5. "شكرًا لتواصلك معنا. نتمنى لك يوماً سعيداً!"

[ابدأ بالسؤال عن الاسم ورقم الاتصال، ثم السؤال عن الطلبات، أكد الطلب وأنهِ المحادثة بالتحيات!]
"""
class TextToSpeech:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    def speak(self, text):
        if self.api_key is None or self.voice_id is None:
            print("Error: Missing API key or voice ID. Cannot proceed with TTS request.")
            return

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.1,
                "similarity_boost": 0.1,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200 and response.content:
            with open("output.wav", "wb") as out:
                out.write(response.content)

            if os.path.getsize("output.wav") > 0:
                # Play the audio file
                player_command = ["ffplay", "-autoexit", "output.wav", "-nodisp"]
                player_process = subprocess.Popen(
                    player_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                player_process.wait()
            else:
                print("Audio file is empty. There might be an issue with the audio generation.")
        else:
            print(f"Failed to synthesize speech: {response.status_code} - {response.text}")

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

def record_audio(filename, duration, sample_rate=16000, chunk_size=1024):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        filename = "input.wav"
        record_audio(filename, 8)  # Record a 5-second audio file

        # Check if the file exists
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return

        model = WhisperModel("base", device="cpu", compute_type="int8" , cpu_threads=8 , num_workers  = 4 )

        segments, info = model.transcribe(filename, beam_size=5)
        sentence = " ".join([segment.text for segment in segments])
        print(f"Human: {sentence}")

        transcript_collector.add_part(sentence)
        full_sentence = transcript_collector.get_full_transcript()
        callback(full_sentence)
        transcript_collector.reset()
        transcription_complete.set()

    except Exception as e:
        print(f"Could not process transcription: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        #Greeting message
        greeting_message  = "مرحبا انا مستمع مساعد الطلب الذكي"
        print(f"AI: {greeting_message}")
        tts = TextToSpeech()
        tts.speak(greeting_message)

        # Loop indefinitely until "مع السلامة" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "مع السلامة" to exit the loop
            if "مع السلامة" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)
            print(f"LLM: {llm_response}")

            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())