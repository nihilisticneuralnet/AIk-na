import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcriber = pipeline(
    "automatic-speech-recognition", model="mnkbcs22021/whisper-small-mar", device=device
)

def transcribe(audio_data):
    return transcriber(audio_data)["text"]
