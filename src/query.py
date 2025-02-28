import requests
import os
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

def query(text, model_id="tiiuae/falcon-7b-instruct"):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    payload = {"inputs": text}

    print(f"Querying...: {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"][len(text) + 1 :]
    else:
        return "Error in AI response."
