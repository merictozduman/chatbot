# test_ollama_chat.py
# Amaç: Ollama local LLM çalışıyor mu test etmek

import ollama

MODEL_NAME = "llama3.1:8b"

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Kısa ve net cevap ver."},
        {"role": "user", "content": "Merhaba, sen kimsin?"}
    ]
)

print("Model cevabı:")
print(response["message"]["content"])
