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
from threading import Thread,Event
import time
import re
from fpdf import FPDF
from tkinter import messagebox

class InterviewBot:
    """
    Manages an AI-driven interview process, handling speech recognition,
    text-to-speech, and conversation flow.
    """
    def __init__(self):
        """Initialize the InterviewBot with default settings and models."""
        load_dotenv(os.path.join(os.path.dirname(os.getcwd()), ".env"))

        # Settings
        self.FS = 44100
        self.THRESHOLD = 50
        self.SILENCE_DURATION = 1.5
        self.CHUNK_SIZE = 1024
        self.SPEAKING_SPEED = 1.1
        self.PASS_PERCENTAGE = 50

        # Globals
        self.pause_loop = True
        self.end_loop = False
        self.unixtime = str(time.time())[:10]
        self.human_audio_n = 0

        self.setup_directories()
        self.settings_updated = Event()
        self.setup_models()

    def setup_directories(self):
        """Create necessary directories for storing interview data."""
        if not os.path.exists(f"data/interviews/{self.unixtime}"):
            os.makedirs(f"data/interviews/{self.unixtime}")
            os.makedirs(f"data/interviews/{self.unixtime}/audio")
            os.makedirs(f"data/interviews/{self.unixtime}/pdfs")
            os.makedirs(f"data/interviews/{self.unixtime}/joblib")
        self.audio_directory = f"data/interviews/{self.unixtime}/audio/"
        self.pdf_directory = f"data/interviews/{self.unixtime}/pdfs/"
        self.joblib_directory = f"data/interviews/{self.unixtime}/joblib/"

    def setup_models(self):
        """Initialize AI models and prompts for the interview process."""
        #Run UI thread to update settings
        ui_thread = Thread(target=self.create_ui_1)
        ui_thread.start()
        self.settings_updated.wait()
        #Uncomment these lines if you want the to override the UI settings.
        self.job_role = "Morning Receptionist (Part Time)"
        self.candidate_skill = "Entry-Level"
        self.role_description = """
        A rare and exciting opportunity has arisen to join a busy and friendly research institute as a morning receptionist with some administrative duties.

        We are seeking a flexible and enthusiastic team player, who is willing to learn and develop as part of a wider group of friendly and highly skilled professional services staff. If you think that describes you then this is an amazing opportunity to gain experience in a dynamic and skilled environment, so apply today or contact hr@gurdon.cam.ac.uk for an informal discussion about the role.

        As the first point of contact for the Institute, you will need excellent communication and interpersonal skills and be professional and friendly.

        Essential Skills For This Post Are

        Excellent customer service skills
        Willingness to be flexible and versatile
        Highly organised
        Capacity to remain patient and tactful at all times and maintain confidentiality
        Ability to multi-task, work under pressure and learn new skills quickly
        Excellent communication skills (both written and verbal)
        Ability to communicate in a friendly and professional manner with everyone
        Computing skills

        The successful applicant will open reception at 8:30. Reception closes at 17:00, Monday to Friday. The post will cover the morning hours from 8:30 to 12:45 (21.25 hours per week). Some flexibility in covering some of the hours of the afternoon post (12:45 to 17:00) during holiday periods would be desirable.

        The position is permanent and available immediately.

        If you have any questions about this vacancy or the application process, please contact the HR team (hr@gurdon.cam.ac.uk).

        Click the 'Apply' button below to register an account with our recruitment system (if you have not already) and apply online.

        Please quote reference PR42849 on your application and in any correspondence about this vacancy.

        The University actively supports equality, diversity and inclusion and encourages applications from all sections of society.

        The University has a responsibility to ensure that all employees are eligible to live and work in the UK.

        Further information

        Further Particulars
        """

        self.system_prompt = f"""
        You are a skilled interviewer who is conducting an initial phone screening interview for a candidate for a {self.candidate_skill} {self.job_role} role to see if the candidate is at minimum somewhat qualified for the role and worth the time to be fully interviewed. The role and company description is copypasted from the job posting as follows: {self.role_description}. Parse through it to extract any information you feel is relevant.
        Your job is to begin a friendly discussion with the candidate, and ask questions relevant to the {self.job_role} role, which may or may not be based on the interviewee's CV, which you have access to. Be sure to stick to this topic even if the candidate tries to steer the conversation elsewhere. If the candidate has other experience on his CV, you can ask about it, but keep it within the context of the {self.job_role} role.
        After the candidate responds to each of your questions, you should not summarise or provide feedback on their responses. THIS POINT IS KEY! You should not summarise or provide feedback on their responses. You must keep your responses short and concise without reiterating what is good about the candidate's response or experience when they reply.
        You can ask follow-up questions if you wish.
        Once you have asked sufficient questions such that you deem the candidate is or isn't fitting for the role, end the interview by thanking the candidate for their time and informing them that they will receive word soon on the outcome of the screening interview. If the candidate does not seem fititng for the role, or if something feels off such as the candidate being unconfident or very very vague feel free to end the interview early. There is no need to inform them of your opinion of their performance, as this will be evaluated later.
        The candidate will begin the interview by greeting you. You are to greet them back, and begin the interview.
        For this specific run, keep the interview to a maximum of 4 questions. Please end the interview with the phrase 'Thank you for your time'.
        """
        try:
            chat_model_name = "claude-3-5-sonnet-20240620"
            cv_path = "data/cvs/cv-deb.pdf"
            self.chat_model = ClaudeChatCV(chat_model_name, self.system_prompt, cv_path)
            self.stt_model = whisper.load_model("medium", device="cuda")
            self.tts_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"Error setting up models: {str(e)}")
            raise

    def stream_tts(self, input_string):
        """Stream text-to-speech audio output.
        
        Args:
            input_string (str): The text to be converted to speech."""
        def _stream_tts():
            try:
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
            except Exception as e:
                print(f"Error in TTS streaming: {str(e)}")
                raise
            finally:
                if 'stream' in locals():
                    stream.stop_stream()
                    stream.close()
                if 'p' in locals():
                    p.terminate()
                thread_done.set()

        thread_done = threading.Event()
        thread = threading.Thread(target=_stream_tts)
        thread.start()
        thread_done.wait()

    def record_speech(self):
        """Record audio input from the user.
        
        Returns:
            str: The filename of the recorded audio."""
        print("Recording... Speak now!")
        audio_data = np.array([], dtype=np.int16)
        try:
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
        except Exception as e:
            print(f"Error recording speec: {str(e)}")
            return None
    
    def create_ui_1(self):
        """Create the first UI for job details and interview settings."""
        def update_settings():
            # Get values from input fields
            self.FS = int(fs_entry.get())
            self.CHUNK_SIZE = int(chunk_size_entry.get())
            self.SPEAKING_SPEED = float(speaking_speed_entry.get())
            self.PASS_PERCENTAGE = float(pass_percentage_entry.get())
            self.job_role = job_role_entry.get()
            self.candidate_skill = candidate_skill_entry.get()
            self.role_description = role_description_text.get("1.0", tk.END).strip()

            # Basic input validation
            if not all([self.job_role, self.candidate_skill, self.role_description]):
                messagebox.showerror("Error", "Please fill in all job details.")
                return

            root.destroy()  # Close the first UI window
            self.settings_updated.set()  # Signal that settings have been updated

        root = tk.Tk()
        root.title("Job Details and Settings")
        root.geometry("900x400") 
        root.configure(bg='#f0f0f0')

        # Job Details Tile
        job_frame = tk.LabelFrame(root, text="Job Details", bg='#e0e0e0', padx=10, pady=10)
        job_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        tk.Label(job_frame, text="Job Role:", bg='#e0e0e0').grid(row=0, column=0, sticky='e', pady=5)
        job_role_entry = tk.Entry(job_frame, width=30)
        job_role_entry.grid(row=0, column=1, sticky='w', pady=5)

        tk.Label(job_frame, text="Candidate Skill:", bg='#e0e0e0').grid(row=1, column=0, sticky='e', pady=5)
        candidate_skill_entry = tk.Entry(job_frame, width=30)
        candidate_skill_entry.grid(row=1, column=1, sticky='w', pady=5)

        tk.Label(job_frame, text="Role Description:", bg='#e0e0e0').grid(row=2, column=0, sticky='ne', pady=5)
        role_description_text = tk.Text(job_frame, width=60, height=10)
        role_description_text.grid(row=2, column=1, sticky='w', pady=5)

        # Settings Tile
        settings_frame = tk.LabelFrame(root, text="Interview Settings", bg='#e0e0e0', padx=10, pady=10)
        settings_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        settings = [
            ("Pass %", self.PASS_PERCENTAGE),
            ("Sampling Frequency", self.FS),
            ("Chunk Size", self.CHUNK_SIZE),
            ("Speaking Speed", self.SPEAKING_SPEED)
        ]

        entries = []
        for i, (text, value) in enumerate(settings):
            tk.Label(settings_frame, text=text, bg='#e0e0e0').grid(row=i, column=0, sticky='e', pady=5)
            entry = tk.Entry(settings_frame, width=10)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, sticky='w', pady=5)
            entries.append(entry)

        pass_percentage_entry, fs_entry, chunk_size_entry, speaking_speed_entry = entries

        # Configure grid weights
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)  # Add this line for the new row

        # Create a new frame for the Save button
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Create the Save button in the new frame
        save_button = tk.Button(button_frame, text="Save", command=update_settings, 
                                bg="blue", fg="white", font=("Arial", 12), width=10)
        save_button.pack()

        root.protocol("WM_DELETE_WINDOW", lambda: self.settings_updated.set())  # Handle window close
        root.mainloop()

        # Continue with the rest of your code here
        print("Settings updated, continuing with execution...")
        print(f"Job Role: {self.job_role}")
        print(f"Candidate Skill: {self.candidate_skill}")
        print(f"Role Description: {self.role_description}")
        print(f"FS: {self.FS}")
        print(f"CHUNK_SIZE: {self.CHUNK_SIZE}")
        print(f"SPEAKING_SPEED: {self.SPEAKING_SPEED}")
        print(f"PASS_PERCENTAGE: {self.PASS_PERCENTAGE}")

    # Add the rest of your program logic here

    def create_ui_2(self):
        """Create the second UI for controlling the interview process."""
        def toggle_pause():
            self.pause_button.config(text="Stop" if self.pause_loop else "Speak",
                                bg="red" if self.pause_loop else "green")
            self.pause_loop = not self.pause_loop

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

        self.pause_button = tk.Button(frame, text="Speak", command=toggle_pause, 
                                bg="green", fg="white", font=("Arial", 12), 
                                width=10, height=2)
        self.pause_button.pack(pady=10)

        end_button = tk.Button(frame, text="End Program", command=end_program, 
                            bg="gray", fg="white", font=("Arial", 12), 
                            width=10, height=2)
        end_button.pack(pady=10)

        root.protocol("WM_DELETE_WINDOW", end_program)
        
        self.root = root
        
        self.root.mainloop()

    
    def run_interview(self):
        """Execute the main interview loop."""
        
        ui_thread = Thread(target=self.create_ui_2)
        ui_thread.start()
        self.pause_loop = True
        while True:
            if not self.pause_loop:
                try:
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
                    self.root.after(0, lambda: self.pause_button.config(state="disabled"))
                    print("Chatting...")
                    response = self.chat_model.chat_with_history_doc(text.get("text"))

                    print("Chatbot: ", response)

                    print("Converting text to speech...")
                    
                    self.stream_tts(response)
                    self.root.after(0, lambda: self.pause_button.config(state="normal"))
                    # Check if the response contains the exit phrase
                    if "Thank you for your time" in response:
                        print("Interview ending...")
                        time.sleep(5)
                        self.end_loop = True
                        self.root.after(0, self.root.quit)
                        break  # Exit the interview loop
                except Exception as e:
                    print(f"Error in interview loop : {str(e)}")
                    raise

            else:
                time.sleep(0.1)
                if self.end_loop:
                    break
        
        ui_thread.join()

    def save_conversation(self):
        """Save the interview conversation to a file and generate a PDF report."""
        try:
            conversation = self.chat_model.get_message_history()
            dump(conversation, self.joblib_directory + "conversation.joblib")
            print(conversation)
        except Exception as e:
            print(f"Error saving joblib file: {str(e)}")
            raise
        try:    
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
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
            

    def main(self):
        """Main entry point for running the interview process.
        
        Returns:
            str: The Unix timestamp of the interview session."""
        self.run_interview()
        self.save_conversation()
        return self.unixtime