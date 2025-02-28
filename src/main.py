import gradio as gr
import numpy as np
import pyaudio
from deep_translator import GoogleTranslator
from wake_detect import detect_wake_word
from query import query
from synthesize import synthesise
from transcribe import transcribe
import requests
import wave
import threading
import time

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(np.frombuffer(in_data, dtype=np.float32))
            return (in_data, pyaudio.paContinue)
        return (in_data, pyaudio.paComplete)

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.callback
        )
        self.stream.start_stream()
        return "Recording started..."

    def stop_recording(self):
        if self.stream is not None:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()

            if len(self.frames) > 0:
                audio_data = np.concatenate(self.frames)
                return (audio_data, self.RATE), "Recording stopped. Click 'Process Audio' to analyze."
        return None, "No audio recorded."

    def __del__(self):
        self.audio.terminate()

recorder = AudioRecorder()

def translate(text, src_lang, dest_lang):
    return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)

def process_audio(audio, wake_word="on", prob_threshold=0.2):
    if audio is None:
        return None, "No audio input received."

    audio_data, sample_rate = audio

    # Detect wake word
    if not detect_wake_word(audio_data, wake_word, prob_threshold):
        return None, "Wake word not detected. Please try again."

    # Transcribe audio
    transcription = transcribe(audio_data)

    # Translate to English
    english_input = translate(transcription, src_lang="mr", dest_lang="en")

    # Get AI response
    english_output = query(english_input)

    # Translate back to Marathi
    marathi_output = translate(english_output, src_lang="en", dest_lang="mr")

    # Synthesize speech
    audio_output = synthesise(marathi_output)

    return (audio_output, 16000), f"""
    Transcription: {transcription}
    Response (Marathi): {marathi_output}
    """
# English Translation: {english_input}
#  Response (English): {english_output}

with gr.Blocks('Zarkel/IBM_Carbon_Theme') as interface:
    gr.Markdown("# AIk-na (aka ऐक ना): Marathi AI Voice Assistant")

    with gr.Row():
        audio_display = gr.Audio(label="Recorded Audio")
        process_btn = gr.Button("Process Audio")
        threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Wake Word Threshold")

    with gr.Row():
        audio_output = gr.Audio(label="Assistant Response")
        process_details = gr.Textbox(label="Process Details", lines=5)

    process_btn.click(
        fn=process_audio,
        inputs=[audio_display, threshold],
        outputs=[audio_output, process_details]
    )

if __name__ == "__main__":
    interface.launch(debug=True)
