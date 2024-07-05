# Please Import Whisper before running this by running the following command in the terminal:
# 1. pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git (update to the latest version of whisper).ValueError
# 2. sudo apt update && sudo apt install ffmpeg (load ffmpeg tool to parse audio files)
# 3. Install pytorch and torchaudio.


import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("Recording.m4a")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)