# import requests
# import dotenv
# import os
# import json
# import audioop
# dotenv.load_dotenv()
# import base64

# class DeepgramTTS():
#     def __init__(self, audio_handler, model="aura-zeus-en"):
#         self.model = model
#         self.isReady = False
#         self.audio_handler = audio_handler
#         self.api_key = os.getenv("SARVAM_API_KEY")
#         self.endpoint = "https://api.sarvam.ai/text-to-speech"

#     def send_text(self, text):
#         # Prepare the payload for the TTS API
#         data = {
#             "inputs": [text],
#             "target_language_code": "en-IN",
#             "speaker": "arvind",
#             "pitch": 0,
#             "pace": 1,
#             "loudness": 1.2,
#             "speech_sample_rate": 16000,
#             "enable_preprocessing": True,
#             "model": "bulbul:v1",
#         }

#         # Send a POST request to the Sarvam TTS API
#         response = requests.post(
#             self.endpoint,
#             headers={'api-subscription-key': f'{self.api_key}', 'Content-Type': 'application/json'},
#             data=json.dumps(data)  # Send data as JSON
#         )

#         if response.status_code == 200:
#             response_json = response.json()
#             audio_data = response_json.get('audios', [None])[0]
#             if audio_data:
                
#                 if isinstance(audio_data, str):
#                     # audio_data = bytes(audio_data, 'utf-8')
#                     audio_data = base64.b64decode(audio_data)
                

#                 # pcm_data = audioop.ulaw2lin(audio_data, 2)
#                 # self.audio_handler(pcm_data)
#                 self.audio_handler(audio_data)
#             else:
#                 print("Error: No audio data found in response.")
#         else:
#             print("Error:", response.status_code, response.text)
    
    

#     def disconnect(self):
#         print("Disconnected from Sarvam TTS (REST mode).")




import requests
import dotenv
import os
import json
import audioop
dotenv.load_dotenv()
import asyncio
from smallestai.waves import AsyncWavesClient
from time import time

class DeepgramTTS():
    def __init__(self, model="aura-zeus-en"):
        self.model = model
        self.isReady = False
        self.api_key = os.getenv("SMALLEST_AI_API_KEY")
        self.tts = None
    
    async def load(self):
        try:
            client = AsyncWavesClient(api_key=self.api_key)
            async with client as tts:
                self.tts = tts
        except Exception as e:
            print("Error",e)

    async def send_text(self, text):
        try:
            s = time()
            print("audio generating...")
            audio_bytes = await self.tts.synthesize(text,voice_id="arman",sample_rate=16000)
            print("audio generated sucessfully",time()-s)
            return audio_bytes
        except Exception as e:
            print("failed to generate audio",e)
            return None

    
    

    def disconnect(self):
        print("Disconnected from Sarvam TTS (REST mode).")
