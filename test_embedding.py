import ollama

resp = ollama.embeddings(
    model="nomic-embed-text",
    prompt="Bu bir embedding testidir"
)

print(type(resp["embedding"]), len(resp["embedding"]))
