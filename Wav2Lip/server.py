from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import dotenv
from wav2lip_service import Wav2lipService
import os
import json
import traceback
from transcription_service import TranscriptionService
dotenv.load_dotenv()
PORT = int(os.getenv("PORT"))
from llm import LLM
from constant import SYSTEM_PROMPT

app = FastAPI()
wav2lipService = None

async def load_model():
    global wav2lipService
    wav2lipService = Wav2lipService()
    await wav2lipService.async_setup()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




#root route
@app.get("/")
async def root():
    return {"message": "Stream Server is running!"}


#handle web call
@app.websocket("/transcribtion")
async def media_stream_web(websocket: WebSocket):
    """
       Handle Connections
    """
    await websocket.accept()
    print("Connection accepted")

    llm = LLM(instruction=SYSTEM_PROMPT)
    async def handle_transcript(text):
        response = llm.generate(text=text)
        print(f"BOT: {response}")
        base64_video = await wav2lipService.send(response)
        await websocket.send_text(json.dumps({
            "event": "media",
            "media": {
                "payload": base64_video,
                "transcription": response
                }
            }))
        

    transcription_service = TranscriptionService(handle_transcript=handle_transcript)
    transcription_service.connect()

    print("here")
    async def handle_connection():
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                if data.get('event') == 'media':
                    payload_base64 = data['media']['payload']
                    transcription_service.send_audio_chunk(payload_base64)

        except WebSocketDisconnect:
            print(f"WebSocket disconnected")
            await transcription_service.finish()
        except Exception as e:
            print(f"Error in handle_connection: {e}")
            await transcription_service.finish()
            traceback.print_exc()


    task = asyncio.create_task(handle_connection())
    await task



if __name__ == "__main__":
    asyncio.run(load_model())
    uvicorn.run(app, host="0.0.0.0", port=PORT)