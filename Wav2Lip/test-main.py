import logging
import os
import asyncio
import numpy as np
from dotenv import load_dotenv
import json
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.plugins import silero
import cv2
import soundfile as sf
# from wav2lip_service import Wav2lipService



# Load environment variables
load_dotenv()

logger = logging.getLogger("voice-assistant")

# Constants
WIDTH = 640
HEIGHT = 480

SAMPLE_RATE = 48000
NUM_CHANNELS = 1  # mono audio
AMPLITUDE = 2 ** 8 - 1
SAMPLES_PER_CHANNEL = 480  # 10ms at 48kHz


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
 
    # ---------------- VIDEO TRACK SETUP ----------------
    video_source = rtc.VideoSource(WIDTH, HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("video-track", video_source)
    
    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=True,
        video_encoding=rtc.VideoEncoding(
            max_framerate=30,
            max_bitrate=3_000_000,
        ),
        video_codec=rtc.VideoCodec.H264,
    )

    await ctx.agent.publish_track(video_track, video_options)

    # COLOR = [255, 255, 255, 0]  # White RGBA

    # async def _draw_color():
    #     argb_frame = bytearray(WIDTH * HEIGHT * 4)
    #     while True:
    #         await asyncio.sleep(0.1)  # 10 fps
    #         argb_frame[:] = COLOR * WIDTH * HEIGHT
            
    #         frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
           
    #         video_source.capture_frame(frame)

    # asyncio.create_task(_draw_color())


  

    async def _stream_video():
        cap = cv2.VideoCapture("input/video.mp4")

        if not cap.isOpened():
            logger.error("Failed to open video file")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                continue

            # Resize to (WIDTH, HEIGHT) if needed
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # Convert from BGR (OpenCV default) to RGBA
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # Flatten frame data to bytearray
            argb_frame = bytearray(frame_rgba.tobytes())


            # Create VideoFrame and capture
            video_frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
            video_source.capture_frame(video_frame)

            await asyncio.sleep(1 / 30)  # Assume 30 FPS

    asyncio.create_task(_stream_video())
    


    # ---------------- AUDIO TRACK SETUP ----------------
    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio-track", audio_source)

    audio_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )

    await ctx.agent.publish_track(audio_track, audio_options)

    frequency = 440  # Hz

    # async def _sinewave():
    #     total_samples = 0
    #     while True:
    #         audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL)
    #         audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

    #         time = (total_samples + np.arange(SAMPLES_PER_CHANNEL)) / SAMPLE_RATE
    #         sinewave = (AMPLITUDE * np.sin(2 * np.pi * frequency * time)).astype(np.int16)
    #         np.copyto(audio_data, sinewave)

    #         await audio_source.capture_frame(audio_frame)
    #         total_samples += SAMPLES_PER_CHANNEL

    #         await asyncio.sleep(SAMPLES_PER_CHANNEL / SAMPLE_RATE)  # maintain real-time pacing

    # asyncio.create_task(_sinewave())


    async def _stream_audio():
        audio_file = "input/audio.wav"

        # Read the entire audio file
        data, samplerate = sf.read(audio_file, dtype='int16')

        if samplerate != SAMPLE_RATE:
            logger.error(f"Audio sample rate {samplerate} doesn't match required {SAMPLE_RATE}")
            return

        # If stereo (2 channels), convert to mono (1 channel) by averaging
        if data.ndim == 2:  # Stereo audio
            data = np.mean(data, axis=1).astype(np.int16)

        num_samples = len(data)
        ptr = 0

        while True:
            if ptr + SAMPLES_PER_CHANNEL > num_samples:
                ptr = 0  # Loop audio

            chunk = data[ptr:ptr + SAMPLES_PER_CHANNEL]
            ptr += SAMPLES_PER_CHANNEL

            audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL)
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

            np.copyto(audio_data, chunk)



            await audio_source.capture_frame(audio_frame)

            await asyncio.sleep(SAMPLES_PER_CHANNEL / SAMPLE_RATE)

    asyncio.create_task(_stream_audio())

    # ---------------- Wav2lip setup ----------------
    # wav2lipService = Wav2lipService(audio_source=audio_source,video_source=video_source)


        # Define the sync handler with correct parameters
    def data_received(event):
        data = event.data.decode("utf-8")
        data = json.loads(data)
        message = data.get('message')
        # wav2lipService.send(message)
       
    
    ctx.room.on("data_received", data_received)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            shutdown_process_timeout=2
        )
    )
