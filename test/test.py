import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoProcessor, VitsModel
import time
import gc
import os

class MarathiWavProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        gc.collect()

        try:
            import psutil
            print(f"Available system memory: {psutil.virtual_memory().available/1e9:.2f} GB")
        except ImportError:
            print("psutil not installed, skipping memory check")

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self.device = "cuda" if total_gpu_memory > 4e9 else "cpu"  
        print(f"Using device: {self.device}")

        print("Loading models...")
        try:
            self.load_whisper()

            self.load_tts()

            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error during model loading: {str(e)}")
            raise

    def load_whisper(self):
        try:
            self.transcriber = WhisperModel(
                "tiny",
                device=self.device,
                compute_type="float32", 
                download_root="./models",
                num_workers=1  
            )
            print("Whisper model loaded")
        except Exception as e:
            print(f"Error loading Whisper model: {str(e)}")
            raise

    def load_tts(self):
        try:
            self.processor = AutoProcessor.from_pretrained(
                "facebook/mms-tts-mar",
                local_files_only=False
            )
            self.tts_model = VitsModel.from_pretrained(
                "facebook/mms-tts-mar"
            ).to(self.device)
            self.tts_model.eval()
            print("TTS model loaded")
        except Exception as e:
            print(f"Error loading TTS model: {str(e)}")
            raise

    def process_audio(self, audio_data, sample_rate):
        """Process audio data with error handling and memory management"""
        try:
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_data = resampler(audio_data)
                sample_rate = 16000

            # Convert to mono if stereo
            if audio_data.shape[0] > 1:
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)

            # Convert to numpy and normalize
            audio_np = audio_data.numpy().flatten()
            audio_np = audio_np / np.max(np.abs(audio_np))

            return audio_np, sample_rate
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise

    def process_wav(self, wav_path):
        print(f"Processing {wav_path}")
        start_time = time.time()

        try:
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Audio file not found: {wav_path}")

            audio_data, sample_rate = torchaudio.load(wav_path)
            print(f"Loaded audio: {audio_data.shape}, {sample_rate}Hz")

            audio_np, sample_rate = self.process_audio(audio_data, sample_rate)

            del audio_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print("Transcribing...")
            try:
                segments, _ = self.transcriber.transcribe(
                    audio_np,
                    language="mr",
                    beam_size=1,  # Reduce beam size
                    vad_filter=True,  # Filter out non-speech
                    initial_prompt="मराठी"  # Help with Marathi recognition
                )
                transcription = " ".join([segment.text for segment in segments])
                print(f"Transcription: {transcription}")
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                raise

            response = "तुम्ही म्हणालात: " + transcription
            print(f"Response: {response}")

            print("Synthesizing speech...")
            try:
                with torch.no_grad():
                    inputs = self.processor(text=response, return_tensors="pt")
                    speech = self.tts_model(
                        input_ids=inputs["input_ids"].to(self.device)
                    ).waveform
            except Exception as e:
                print(f"Speech synthesis error: {str(e)}")
                raise

            output_path = "response.wav"
            torchaudio.save(output_path, speech.cpu(), 16000)

            del speech
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            return output_path

        except Exception as e:
            print(f"Error in process_wav: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        processor = MarathiWavProcessor()
        output_file = processor.process_wav("/content/sample.wav")
        print(f"Response saved to: {output_file}")
    except Exception as e:
        print(f"Main execution error: {str(e)}")
