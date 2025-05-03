from dotenv import load_dotenv
import logging
from deepgram.utils import verboselogs
import dotenv
dotenv.load_dotenv()
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import os
import base64
import asyncio

load_dotenv()

class TranscriptionService:
    def __init__(self, handle_transcript,event_loop=None,model="nova-3", language="en-US", sample_rate=16000):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.is_finals = []
        self.connected = False
        self.handle_transcript = handle_transcript
        self.loop = event_loop or asyncio.get_event_loop()

        self.deepgram = DeepgramClient(self.api_key)
        self.dg_connection = None

    def connect(self):
        """Method to establish the WebSocket connection."""
        try:
            self.dg_connection = self.deepgram.listen.websocket.v("1")
            
            def on_open(selfd, open, **kwargs):
                print("Connection Open")
                self.connected = True

            def on_message(selfd,result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return
                if result.is_final:
                    self.is_finals.append(sentence)
                    asyncio.run_coroutine_threadsafe(
                        self.handle_transcript(sentence), self.loop
                    )
                    
                    if result.speech_final:
                        utterance = " ".join(self.is_finals)
                        print(f"Speech Final: {utterance}")
                        self.is_finals = []
                    else:
                        
                        print(f"Is Final: {sentence}")
                else:
                    print(f"Interim Results: {sentence}")

            def on_metadata(selfd, metadata, **kwargs):
                print(f"Metadata: {metadata}")

            def on_speech_started(selfd, speech_started, **kwargs):
                print("Speech Started")

            def on_utterance_end(selfd, utterance_end, **kwargs):
                print("Utterance End")
                if len(self.is_finals) > 0:
                    utterance = " ".join(self.is_finals)
                    print(f"Utterance End: {utterance}")
                    self.is_finals = []

            def on_close(selfd, close, **kwargs):
                self.connected = False
                print("Connection Closed")

            def on_error(selfd, error, **kwargs):
                print(f"Handled Error: {error}")

            def on_unhandled(selfd, unhandled, **kwargs):
                print(f"Unhandled Websocket Message: {unhandled}")


            # Set up event listeners
            self.dg_connection.on(LiveTranscriptionEvents.Open, on_open)
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
            self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

            options = LiveOptions(
                model=self.model,
                language=self.language,
                smart_format=True,
                channels=1,
                sample_rate=self.sample_rate,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                endpointing=300,
            )

            addons = {
                "no_delay": "true"
            }

            # Start the connection
            if self.dg_connection.start(options, addons=addons) is False:
                print("Failed to connect to Deepgram")
                return False

            print("Connection Established")
            return True

        except Exception as e:
            print(f"Error connecting: {e}")
            return False

    def send_audio_chunk(self, audio_chunk):
        """Method to send custom audio chunks to Deepgram."""
        if not self.dg_connection or self.connected == False:
            print("WebSocket is not open. Cannot send audio.")
            return
        
        try:
            binary_data = base64.b64decode(audio_chunk)
            self.dg_connection.send(binary_data)
        except Exception as e:
            print(f"Error sending audio chunk: {e}")

   
    def finish(self):
        """Finish the connection and cleanup."""
        if self.dg_connection:
            self.dg_connection.finish()
            print("Finished connection")

