from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg
import time
import os
from openai import OpenAI

# ---------------- CONFIG ----------------
DB_CONN = {
    "dbname": "vektordb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

RETRIEVE_K = 15
USE_TOP_N = 10
MAX_TOKENS = 600

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- APP ----------------
app = FastAPI(title="Fez-GPT")

# UI root
@app.get("/")
def root():
    return FileResponse("static/index.html")

# Static files (css/js if later added)
app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str

# ---------------- EMBEDDING ----------------
def embed(text: str):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding

def to_pgvector(v):
    return "[" + ",".join(str(x) for x in v) + "]"

# ---------------- DB SEARCH ----------------
def search_db(question):
    q_emb = embed(question)
    q_vec = to_pgvector(q_emb)

    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT content
        FROM rag_documents
        ORDER BY embedding <=> '{q_vec}'::vector
        LIMIT {RETRIEVE_K}
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [r[0] for r in rows]

# ---------------- API ----------------
@app.post("/ask")
def ask(q: Question):
    start = time.time()

    chunks = search_db(q.question)
    context = "\n\n".join(chunks[:USE_TOP_N])

    prompt = f"""
You are Fez-GPT, a travel assistant for FezTravel.

Use the information below if relevant.
If incomplete, infer meaning.
Do not say "not found" unless nothing is related.

Context:
{context}

Question:
{q.question}
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=MAX_TOKENS
    )

    return {
        "question": q.question,
        "answer": resp.choices[0].message.content,
        "latency_sec": round(time.time() - start, 2)
    }
