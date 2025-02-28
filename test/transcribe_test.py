!pip install transformers torch torchaudio gradio soundfile ffmpeg-python pydub ipywebrtc

from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display
import torchaudio
import tempfile

pipe = pipeline(model="mnkbcs22021/whisper-small-mar")

uploader = widgets.FileUpload(accept=".wav, .mp3, .ogg", multiple=False)

button = widgets.Button(description="Transcribe")

output = widgets.Output()

def transcribe(audio_path):
    text = pipe(audio_path)["text"]
    return text

def on_button_click(b):
    with output:
        output.clear_output()
        if uploader.value:
            file_info = next(iter(uploader.value.values()))  
            audio_bytes = file_info["content"]  # Get audio content as bytes

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name 

            text = transcribe(temp_audio_path)
            print("Transcription:", text)
        else:
            print("Please upload an audio file.")

button.on_click(on_button_click)

display(uploader, button, output)


# from huggingface_hub import notebook_login
# notebook_login()
# from transformers import pipeline
# import gradio as gr

# pipe = pipeline(model="sanchit-gandhi/whisper-small-hi")  # change to "your-username/the-name-you-picked"

# def transcribe(audio):
#     text = pipe(audio)["text"]
#     return text

# iface = gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(sources="microphone", type="filepath"),
#     outputs="text",
#     title="Whisper Small Hindi",
#     description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
# )

# iface.launch()
