import os 
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization":f"Bearer {api_key}"
    }
)


response.raise_for_status()

models = response.json()["data"]

for model in models:
    print(model["id"])