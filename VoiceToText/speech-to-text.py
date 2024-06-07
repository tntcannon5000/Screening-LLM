import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import queue
import threading

# --- Settings ---
fs = 44100  # Sampling frequency
threshold = 30  # Volume threshold for silence (adjust this)
silence_duration = 2  # Seconds of silence before stopping (adjust this)
chunk_size = 1024  # Process audio in chunks for efficiency

# --- Global Variables ---
audio_queue = queue.Queue()
recording_stopped = False  # Flag to signal recording stop

# --- Functions ---
def is_silent(data):
    rms = np.sqrt(np.mean(data**2))
    return rms < threshold

def record_audio():
    global recording_stopped

    print("Recording... Speak now!")
    audio_data = np.array([], dtype=np.int16)  # Initialize empty array
    silent_chunks = 0

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while not recording_stopped:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("Warning: Input overflowed!")

            audio_data = np.append(audio_data, chunk)
            audio_queue.put(chunk)  # Add chunk to the queue

            if is_silent(chunk):
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > int(silence_duration * fs / chunk_size):
                print("Silence detected, stopping recording.")
                recording_stopped = True
                break

    wav.write("recording.wav", fs, audio_data)
    print("Recording saved to 'recording.wav'")

def transcribe_audio():
    global recording_stopped
    model = whisper.load_model("base", device="cpu")  # Use CUDA if available

    temp_audio = np.array([], dtype=np.int16)
    while not recording_stopped or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=1)  # Get chunk from queue
            temp_audio = np.append(temp_audio, chunk)

            # Transcribe when enough data is accumulated (experiment with chunk size)
            if len(temp_audio) >= 4096: 
                wav.write("temp_audio.wav", fs, temp_audio)
                result = model.transcribe("temp_audio.wav")
                print(result["text"], end=" ", flush=True)
                temp_audio = np.array([], dtype=np.int16)  # Reset temp_audio

        except queue.Empty:
            pass  # Continue waiting for more audio

if __name__ == "__main__":
    # Create and start threads
    recording_thread = threading.Thread(target=record_audio)
    transcription_thread = threading.Thread(target=transcribe_audio)

    recording_thread.start()
    transcription_thread.start()

    # Wait for threads to finish
    recording_thread.join()
    transcription_thread.join()