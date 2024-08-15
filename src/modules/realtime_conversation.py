import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from joblib import load, dump
from ..utils.anthropicwrapper import ClaudeChatCV
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
import pyaudio
import threading
import tkinter as tk
from threading import Thread
import time
import re
from fpdf import FPDF

class InterviewBot:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(os.getcwd()), ".env"))

        # Settings
        self.FS = 44100
        self.THRESHOLD = 50
        self.SILENCE_DURATION = 1.5
        self.CHUNK_SIZE = 1024
        self.SPEAKING_SPEED = 1.1

        # Globals
        self.pause_loop = True
        self.end_loop = False
        self.unixtime = str(time.time())[:10]
        self.human_audio_n = 0

        self.setup_directories()
        self.setup_models()

    def setup_directories(self):
        if not os.path.exists(f"data/interviews/{self.unixtime}"):
            os.makedirs(f"data/interviews/{self.unixtime}")
            os.makedirs(f"data/interviews/{self.unixtime}/audio")
            os.makedirs(f"data/interviews/{self.unixtime}/pdfs")
            os.makedirs(f"data/interviews/{self.unixtime}/joblib")
        self.audio_directory = f"data/interviews/{self.unixtime}/audio/"
        self.pdf_directory = f"data/interviews/{self.unixtime}/pdfs/"
        self.joblib_directory = f"data/interviews/{self.unixtime}/joblib/"

    def setup_models(self):
        self.job_role = "RAG AI Engineer"
        self.candidate_skill = "Entry-Level"
        self.role_description = """
        Permanent

        London (Hybrid)

        Salary - £50,000 - £75,000 p/a + benefits

        My client are on the cutting edge of digital reinvention, helping clients reimagine how they serve their connected customers and operate enterprises. As an experienced AI Engineer, you'll play a pivotal role in their revolution. You'll leverage deep learning, neuro-linguistic programming (NLP), computer vision, chatbots, and robotics to enhance business outcomes and drive innovation. Join their multidisciplinary team to shape their AI strategy and showcase the potential of AI through early-stage solutions.

        Tasks

        1. Enhance Retrieval and Generation:
        Create and manage RAG pipelines to improve information retrieval and content generation tasks.
        1. LLMs Optimization:
        Understand the nuances between prompting and training large language models (LLMs) to enhance model performance.
        1. LLM Evaluation:
        Evaluate different LLMs to find the best fit for specific use cases.
        1. Model Efficiency:
        Address speed, performance, and cost-related issues in model implementation.
        1. Collaboration and Innovation:
        Work closely with cross-functional teams to integrate AI solutions into production environments.
        Stay informed about the latest advancements in AI and machine learning to continuously enhance our solutions.
        Requirements

        4+ years of hands-on Python development experience, especially with machine learning frameworks (e.g., TensorFlow, PyTorch).
        Proven experience setting up and optimizing retrieval-augmented generation (RAG) pipelines.
        Strong understanding of large language models (LLMs) and the differences between prompting and training.
        Production-level experience with AWS services.
        Hands-on experience testing and comparing different LLMs (OpenAI, Llama, Claude, etc.).
        Familiarity with model speed and cost optimization challenges.
        Excellent problem-solving skills and attention to detail.
        Strong communication and teamwork abilities.
        Benefits

        Endless Learning and Growth: Explore boundless opportunities for personal and professional development in our dynamic, AI-driven startup.
        Inclusive and Supportive Environment: Join a collaborative culture that prioritizes transparency, trust, and open dialogue among team members.
        Generous Benefits: Enjoy comprehensive perks, including unlimited annual leave, birthday leave, and exciting team trips.
        Impactful Work: Contribute to the financial industry by working with cutting-edge AI technologies that make a difference.
        Please apply for this exciting role ASAP!!
        """

        self.system_prompt = f"""
        You are a skilled interviewer who is conducting an initial phone screening interview for a candidate for a {self.candidate_skill} {self.job_role} role to see if the candidate is at minimum somewhat qualified for the role and worth the time to be fully interviewed. The role and company description is copypasted from the job posting as follows: {self.role_description}. Parse through it to extract any information you feel is relevant.
        Your job is to begin a friendly discussion with the candidate, and ask questions relevant to the {self.job_role} role, which may or may not be based on the interviewee's CV, which you have access to. Be sure to stick to this topic even if the candidate tries to steer the conversation elsewhere. If the candidate has other experience on his CV, you can ask about it, but keep it within the context of the {self.job_role} role.
        After the candidate responds to each of your questions, you should not summarise or provide feedback on their responses. THIS POINT IS KEY! You should not summarise or provide feedback on their responses. You must keep your responses short and concise without reiterating what is good about the candidate's response or experience when they reply.
        You can ask follow-up questions if you wish.
        Once you have asked sufficient questions such that you deem the candidate is or isn't fitting for the role, end the interview by thanking the candidate for their time and informing them that they will receive word soon on the outcome of the screening interview. If the candidate does not seem fititng for the role, or if something feels off such as the candidate being unconfident or very very vague feel free to end the interview early. There is no need to inform them of your opinion of their performance, as this will be evaluated later.
        The candidate will begin the interview by greeting you. You are to greet them back, and begin the interview.
        For this specific run, keep the interview to a maximum of 4 questions.
        """

        chat_model_name = "claude-3-5-sonnet-20240620"
        cv_path = "data/cvs/cv-deb.pdf"
        self.chat_model = ClaudeChatCV(chat_model_name, self.system_prompt, cv_path)
        self.stt_model = whisper.load_model("medium", device="cuda")
        self.tts_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def stream_tts(self, input_string):
        def _stream_tts():
            p = pyaudio.PyAudio()
            stream = p.open(format=8,
                            channels=1,
                            rate=round(24_005 * self.SPEAKING_SPEED),
                            output=True)
            with self.tts_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=input_string,
                response_format="pcm"
            ) as response:
                for chunk in response.iter_bytes(1024):
                    stream.write(chunk)
                    
            thread_done.set()

        thread_done = threading.Event()
        thread = threading.Thread(target=_stream_tts)
        thread.start()
        thread_done.wait()

    def record_speech(self):
        print("Recording... Speak now!")
        audio_data = np.array([], dtype=np.int16)

        with sd.InputStream(samplerate=self.FS, channels=1, dtype='int16') as stream:
            while True:
                chunk, overflowed = stream.read(self.CHUNK_SIZE)
                if overflowed:
                    print("Warning: Input overflowed!")
                audio_data = np.append(audio_data, chunk)

                if self.pause_loop:
                    break
        
        self.human_audio_n += 1
        wavstring = f"/audio_{self.human_audio_n}_{self.unixtime}.wav"
        wav.write(self.audio_directory + wavstring, self.FS, audio_data)

        return wavstring

    def create_ui(self):
        def toggle_pause():
            self.pause_loop = not self.pause_loop
            pause_button.config(text="Stop" if not self.pause_loop else "Speak now",
                                bg="red" if not self.pause_loop else "green")

        def end_program():
            self.end_loop = True
            root.quit()
            root.destroy()

        root = tk.Tk()
        root.title("Control Panel")
        root.geometry("300x200")
        root.configure(bg='#f0f0f0')

        frame = tk.Frame(root, bg='#f0f0f0')
        frame.pack(expand=True, fill='both', padx=20, pady=20)

        pause_button = tk.Button(frame, text="Stop", command=toggle_pause, 
                                bg="red", fg="white", font=("Arial", 12), 
                                width=10, height=2)
        pause_button.pack(pady=10)

        end_button = tk.Button(frame, text="End Program", command=end_program, 
                            bg="gray", fg="white", font=("Arial", 12), 
                            width=10, height=2)
        end_button.pack(pady=10)

        root.protocol("WM_DELETE_WINDOW", end_program)
        root.mainloop()

    def run_interview(self):
        ui_thread = Thread(target=self.create_ui)
        ui_thread.start()

        while True:
            if not self.pause_loop:
                time.sleep(0.1)
                print("Recording speech...")
                wav_file = self.record_speech()

                if self.end_loop:
                    break

                print("Converting speech to text...")
                text = self.stt_model.transcribe(self.audio_directory + wav_file, language="en")
                print("You said: ", text.get("text"))

                if self.end_loop:
                    break

                print("Chatting...")
                response = self.chat_model.chat_with_history_doc(text.get("text"))

                print("Chatbot: ", response)

                print("Converting text to speech...")
                self.stream_tts(response)

            else:
                time.sleep(0.1)
                if self.end_loop:
                    break

    def save_conversation(self):
        conversation = self.chat_model.get_message_history()
        dump(conversation, self.joblib_directory + "conversation.joblib")
        print(conversation)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(left=10, top=20, right=10)
        pdf.set_font("Helvetica", size=12)
        usable_width = pdf.w - pdf.l_margin - pdf.r_margin - 0

        for turn in conversation[2:]:
            if turn['role'] and turn['content']:
                pdf.set_font("Helvetica", style="B", size=14)
                pdf.cell(0, 10, txt=f"{turn['role'].capitalize()}:", ln=True)
                pdf.set_font("Helvetica", size=12)
                pdf.x = pdf.l_margin
                pdf.multi_cell(usable_width, 6, txt=turn['content'])
                pdf.ln(3)

        pdf.output(self.pdf_directory + "conversation.pdf")

def main():
    bot = InterviewBot()
    bot.run_interview()
    bot.save_conversation()
    return bot.unixtime

if __name__ == "__main__":
    main()