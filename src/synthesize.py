import torch
from transformers import AutoProcessor, VitsModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("facebook/mms-tts-mar")
model = VitsModel.from_pretrained("facebook/mms-tts-mar").to(device)

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = model(input_ids=inputs["input_ids"].to(device)).waveform
    return speech.cpu().numpy()[0]
