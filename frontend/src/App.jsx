import React, { useEffect, useRef, useState } from 'react'
import { TranscriptionService } from './services/transcription.service';

const App = () => {
  const [state, setState] = useState("connecting");
  const transcribtionRef = useRef(null);
  const videoRef = useRef(null);
  const [started, setStarted] = useState(false);


  const handlePlay = (src) => {
    console.log('Play button clicked');
    videoRef.current.src = src;
    videoRef.current.loop = false;
    videoRef.current.play().then(() => {
      console.log('Video is playing');
    }).catch((error) => {
      console.error('Error playing video:', error);
    });


    setState("Speaking...");
    videoRef.current.addEventListener('ended', () => {
      setState("listening");
      videoRef.current.src = '/idle.mp4';
      videoRef.current.loop = true;
      videoRef.current.play().then(() => {
        console.log('Video is playing');
      }).catch((error) => {
        console.error('Error playing video:', error);
      });
    });
  }
  const handleIntrupt = () => {
    console.log('Interrupt button clicked')
  }

  const handleStart = () => {
    videoRef.current.src = '/introducation.mp4';
    videoRef.current.play().then(() => {
      setState("Speaking...");
    }).catch((error) => {
      console.error('Error playing video:', error);
    });

    videoRef.current.addEventListener('ended', () => {
      setState("listening");
      videoRef.current.src = '/idle.mp4';
      videoRef.current.loop = true;
      videoRef.current.play().then(() => {
        console.log('Video is playing');
      }).catch((error) => {
        console.error('Error playing video:', error);
      });
    });

    setStarted(true);
    transcribtionRef.current = new TranscriptionService(handlePlay, handleIntrupt, setState);
  }
  return (
    <div className='container'>
      {
        !started && <button className='button' onClick={handleStart}>Start</button>
      }
      {
        started && <div className='status-box'>{state}</div>
      }
      <video className='video-container' ref={videoRef} poster='/poster.jpeg'></video>
    </div>
  )
}

export default App