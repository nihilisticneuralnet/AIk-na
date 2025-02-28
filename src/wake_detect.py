import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

def detect_wake_word(audio_data, wake_word="on", prob_threshold=0.5):
    predictions = classifier(audio_data)
    for pred in predictions:
        if pred["label"] == wake_word and pred["score"] > prob_threshold:
            return True
    return False
