# AIk na (aka ऐक ना): Marathi AI Voice Assistant


**AIk na** (aka **ऐक ना**) is a Marathi AI Voice Assistant that processes voice inputs to provide intelligent responses. It performs the following tasks:

1. Wake Word Detection: Detects the wake word "on" to activate the assistant.
2. Speech Transcription: Converts Marathi speech into text using an Automatic Speech Recognition (ASR) model.
3. Translation: Translates the transcribed Marathi text into English.
4. AI Response Generation: Sends the English text to an AI model to generate a response.
5. Back Translation: Converts the AI-generated English response back to Marathi.
6. Speech Synthesis: Synthesizes the Marathi response as an audio output.
   
## Project Structure

   ```plaintext
AIk-na/src
│── wake_detect.py       # Wake word detection module
│── transcribe.py        # Marathi speech-to-text transcription
│── query.py             # AI model response generation
│── synthesize.py        # Marathi text-to-speech synthesis
│── main.py              # Main application using Gradio
│── .env                 # Store token

```

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**: Run `git clone https://github.com/nihilisticneuralnet/AIk-na.git` to clone the repository to your local machine.

2. **Install Dependencies**: Navigate to the project directory and install the required packages by running `cd <repository-directory>` followed by `pip install -r requirements.txt`. 

3. **Set Up Environment Variables**: In the `.env` file in the project root directory and insert your Gemini and Sarvam API keys as follows:
   ```plaintext
   HF_TOKEN= "<huggingface_token>"
   ```
   Replace `<huggingface_token>` with your actual API key.

4. **Run the Application**: Finally, run the application using `python main.py`.

Ensure you have all the necessary libraries installed before running these commands.

*See [examples](https://github.com/nihilisticneuralnet/Manya/tree/main/examples) for working in a Python notebook.*



## Finetune

#### Dataset used: [OpenSLR](https://openslr.org/64/)


## References

- https://huggingface.co/blog/fine-tune-whisper
- https://huggingface.co/facebook/xglm-1.7B
- https://huggingface.co/google/mt5-small
- https://huggingface.co/coqui/XTTS-v2
- https://huggingface.co/spaces/ai4bharat/indic-parler-tts
- https://huggingface.co/learn/audio-course/en/chapter7/voice-assistant
- https://openslr.org/64/
