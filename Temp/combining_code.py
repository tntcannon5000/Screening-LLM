# Imports
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from joblib import load, dump
from AnthropicWrapper import ClaudeChatCV
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
import pyaudio
import threading
import time
import keyboard

# Get the path to the immediate parent folder of the current working directory
parent_folder_path = os.path.dirname(os.getcwd())

# Construct the path to the .env file in the parent folder
dotenv_path = os.path.join(parent_folder_path, ".env")

# Load the .env file
load_dotenv(dotenv_path)

# --- Settings ---
FS = 44100               # Sampling frequency
THRESHOLD = 50          # Volume threshold for silence (adjust this)
SILENCE_DURATION = 1.5   # Seconds of silence before stopping (adjust this)
CHUNK_SIZE = 1024        # Process audio in chunks for efficiency
SPEAKING_SPEED = 1.1     # Speed of speaking

# --- Globals ---
pause_loop = False

job_role = "Machine Learning Engineer"
candidate_skill = "Entry-Level"

role_description = """
Do you want to tackle the biggest questions in finance with near infinite compute power at your fingertips?

G-Research is a leading quantitative research and technology firm, with offices in London and Dallas. We are proud to employ some of the best people in their field and to nurture their talent in a dynamic, flexible and highly stimulating culture where world-beating ideas are cultivated and rewarded.

This is a role based in our new Soho Place office - opened in 2023 - in the heart of Central London and home to our Research Lab.

The role

We are looking for exceptional machine learning engineers to work alongside our quantitative researchers on cutting-edge machine learning problems.

As a member of the Core Technical Machine Learning team, you will be engaged in a mixture of individual and collaborative work to tackle some of the toughest research questions.

In this role, you will use a combination of off-the-shelf tools and custom solutions written from scratch to drive the latest advances in quantitative research.

Past projects have included:

Implementing ideas from a recently published research paper
Writing custom libraries for efficiently training on petabytes of data
Reducing model training times by hand optimising machine learning operations
Profiling custom ML architectures to identify performance bottlenecks
Evaluating the latest hardware and software in the machine learning ecosystem
Who are we looking for?

Candidates will be comfortable working both independently and in small teams on a variety of engineering challenges, with a particular focus on machine learning and scientific computing.

The ideal candidate will have the following skills and experience:

Either a post-graduate degree in machine learning or a related discipline, or commercial experience working on machine learning models at scale. We will also consider exceptional candidates with a proven record of success in online data science competitions, such as Kaggle
Strong object-oriented programming skills and experience working with Python, PyTorch and NumPy are desirable
Experience in one or more advanced optimisation methods, modern ML techniques, HPC, profiling, model inference; you dont need to have all of the above
Excellent ML reasoning and communication skills are crucial: off-the-shelf methods dont always work on our data so you will need to understand how to develop your own models in a collaborative environment working in a team with complementary skills
Finance experience is not necessary for this role and candidates from non-financial backgrounds are encouraged to apply.

Why should you apply?

Highly competitive compensation plus annual discretionary bonus
Lunch provided (via Just Eat for Business) and dedicated barista bar
35 days annual leave
9 percent company pension contributions
Informal dress code and excellent work/life balance
Comprehensive healthcare and life assurance
Cycle-to-work scheme
Monthly company events
"""

system_prompt = f"""You are a skilled interviewer who is conducting an initial phone screening interview for a candidate for a {candidate_skill} {job_role} role to see if the candidate is at minimum somewhat qualified for the role and worth the time to be fully interviewed. The role and company description is copypasted from the job posting as follows: {role_description}. Parse through it to extract any information you feel is relevant.
Your job is to begin a friendly discussion with the candidate, and ask questions relevant to the {job_role} role, which may or may not be based on the interviewee's CV, which you have access to. Be sure to stick to this topic even if the candidate tries to steer the conversation elsewhere. If the candidate has other experience on his CV, you can ask about it, but keep it within the context of the {job_role} role.
After the candidate responds to each of your questions, you should not summarise or provide feedback on their responses. THIS POINT IS KEY! You should not summarise or provide feedback on their responses. You must keep your responses short and concise without reiterating what is good about the candidate's response or experience when they reply.
You can ask follow-up questions if you wish.
Once you have asked sufficient questions such that you deem the candidate is or isn't fitting for the role, end the interview by thanking the candidate for their time and informing them that they will receive word soon on the outcome of the screening interview. If the candidate does not seem fititng for the role, or if something feels off such as the candidate being unconfident or very very vague feel free to end the interview early. There is no need to inform them of your opinion of their performance, as this will be evaluated later.
The candidate will begin the interview by greeting you. You are to greet them back, and begin the interview.
For this specific run, keep the interview to a maximum of 3 questions."""

# Initialising Claude and ConversationChain
chat_model_name = "claude-3-5-sonnet-20240620"

# Give the path to a CV in your disk
pdf_path = r"D:\Kent\University Of Kent UK\Jobs\CV-DebapratimKundu.pdf"
chat_model = ClaudeChatCV(chat_model_name, system_prompt, pdf_path)

# Initialising whisper
stt_model = whisper.load_model("medium", device="cuda")

# Initialising text2speech
tts_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def stream_tts(input_string):
    def _stream_tts():
        p = pyaudio.PyAudio()
        stream = p.open(format=8,
                        channels=1,
                        rate=round(24_005 * SPEAKING_SPEED),
                        output=True)
        with tts_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=input_string,
            response_format="pcm"
        ) as response:
            for chunk in response.iter_bytes(1024):
                stream.write(chunk)
                
        #print("FINISHED!!!!!!!!!!!!!!!!!!!!")
        thread_done.set()

    thread_done = threading.Event()

    thread = threading.Thread(target=_stream_tts)
    thread.start()
    thread_done.wait()
    
    # --- Functions ---
def is_silent(data):
    rms = np.sqrt(np.mean(data**2))
    print("RMS: ", rms)
    return rms < THRESHOLD
  
  # Initialising speech2text without silence detection
def record_speech():
    print("Recording... Speak now!")
    audio_data = np.array([], dtype=np.int16)  # Initialize empty array

    with sd.InputStream(samplerate=FS, channels=1, dtype='int16') as stream:
        while True:
            chunk, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("Warning: Input overflowed!")
            audio_data = np.append(audio_data, chunk)

            if pause_loop:
                break
    
    wav.write("g97613g9f0g8.wav", FS, audio_data)

    return "g97613g9f0g8.wav"
  
  def keyboard_listener():
    global pause_loop
    
    def on_key_press(event):
        global pause_loop
        #print("Key pressed: {}" .format(event.name))
        if event.name == "+":
            pause_loop = False
            print("Set to continue on next loop")
        elif event.name == "-":
            pause_loop = True
            print("Set to pause on next loop")
    
    keyboard.on_press(on_key_press)
    keyboard.wait('esc')

listener_thread = threading.Thread(target=keyboard_listener)
listener_thread.start()

while True:
    if not pause_loop:
        # --- Record Speech ---
        time.sleep(0.1)
        print("Recording speech...")
        wav_file = record_speech()

        # --- Speech to Text ---
        print("Converting speech to text...")
        text = stt_model.transcribe(wav_file, language="en")
        print("You said: ", text.get("text"))

        # --- Chatbot ---
        print("Chatting...")
        response = chat_model.chat_with_history_doc(text.get("text"))

        print("Chatbot: ", response)

        # --- Text to Speech ---
        print("Converting text to speech...")
        stream_tts(response)

    else:
        time.sleep(0.1)