from huggingface_hub import InferenceClient
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="groq",
    api_key=HF_TOKEN
)



def run_llm(prompt: str, max_tokens: int = 3000):
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    raw = response.choices[0].message.content
    # attempt to extract JSON block from the text
    parsed = raw
    return parsed if parsed is not None else raw


