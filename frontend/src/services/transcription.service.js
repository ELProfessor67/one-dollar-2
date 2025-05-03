import { BACKEND_URL } from "../constants";
import { VideoSDKNoiseSuppressor } from "@videosdk.live/videosdk-media-processor-web";

export class TranscriptionService {
    audioStream = null;
    socket = null;
    play = null
    stop = null
    setStatus = null;
    noiseProcessor = new VideoSDKNoiseSuppressor();

    constructor(handlePlayAudio, handleIntrupt, setStatus) {
        this.play = handlePlayAudio;
        this.stop = handleIntrupt;
        this.setStatus = setStatus;
        this.socket = new WebSocket(`${BACKEND_URL}/transcribtion`);
        this.sendAudioStream = this.sendAudioStream.bind(this); 
        this.socket.onopen = () => {
            console.log("WebSocket connection opened");
            this.setStatus("listening");
            setTimeout(() => this.sendAudioStream(), 1000);
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.event) {
                case 'clear':
                    handleIntrupt();
                    break;
                case 'transcript':
                    const transcript = data.transcript.value;
                    console.log("User:", transcript);
                    break;
           
                case 'media':
                    const response = data.media;
                    console.log("George Washington:", response.transcription);
                    this.play(`data:video/mp4;base64,${response.payload}`);
                    break;
         
                case 'state':
                    const value = data.state.value;
                    setStatus(value);
                    break;
            }
        };
        
        this.socket.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
        
        this.socket.onclose = () => {
            console.log("WebSocket connection closed");
        };
    }




    async getAudioStream() {
        const stream = navigator.mediaDevices.getUserMedia({ audio: true })
        return stream;
    }

    sendAudioStream() {
        this.getAudioStream().then(async (stream) => {
            this.audioStream = stream;

            let processedStream = stream;

            if ('MediaStreamTrackGenerator' in window) {
                try {
                    processedStream = await this.noiseProcessor.getNoiseSuppressedAudioStream(stream);
                } catch (err) {
                    console.warn('Noise suppression failed:', err);
                    processedStream = stream; // fallback to raw stream
                }
            } else {
                console.warn('Noise suppression not supported on this browser (e.g. iOS Safari)');
                processedStream = stream; // fallback
            }

            const mediaRecorder = new MediaRecorder(processedStream);
            mediaRecorder.addEventListener('dataavailable', async (event) => {
                if (event.data.size > 0 && this.socket.readyState == 1) {
                    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                        const blob = event.data;
                        const reader = new FileReader();
                        reader.onload = () => {
                          if (reader.readyState == 2) {
                            const data = {
                              event: 'media',
                              media: {
                                payload: reader?.result?.split('base64,')[1]
                              }
                            }
                            console.log("sending...")
                            this.socket.send(JSON.stringify(data));
                          }
                        }
                        reader.readAsDataURL(blob);
                      }
                }
            });
            mediaRecorder.start(250);

           
        }).catch(err => console.log(err.message));
    }



    handlemute(value) {
        this.audioStream.getAudioTracks().forEach(track => {
            track.enabled = !value;
        });
    }


    disconnect() {
        this.socket.close();
    }
}