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

MAX_TOKENS = 600
MAX_TOURS = 50
CHUNKS_PER_TOUR = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- KEYWORDS ----------------
KEYWORDS = {
    "locations": ["gobeklitepe", "cappadocia", "pamukkale"],
    "transport": ["flight", "transfer", "bus"],
    "countries": ["turkey", "greece", "egypt"],
    "pricing": ["price", "usd", "eur", "$"]
}

# ---------------- APP ----------------
app = FastAPI(title="Fez-GPT")

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str

# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(txt: str) -> str:
    return (
        txt.lower()
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ç", "c")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ı", "i")
    )

# ---------------- EMBEDDING ----------------
def embed(text: str):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding

def to_pgvector(v):
    return "[" + ",".join(str(x) for x in v) + "]"

# ---------------- KEYWORD PARSER ----------------
def parse_keywords(question: str):
    q = normalize_text(question)
    found = []

    for kws in KEYWORDS.values():
        for kw in kws:
            if kw in q:
                found.append(kw)

    return list(set(found))

# ---------------- HYBRID DB SEARCH ----------------
def search_db_hybrid(question: str):
    keywords = parse_keywords(question)
    q_vec = to_pgvector(embed(question))

    print("\n====== HYBRID SEARCH ======")
    print("Question:", question)
    print("Keywords:", keywords)

    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    where_clauses = ["source_type = 'pdf'"]
    params = []

    for kw in keywords:
        where_clauses.append(
            "translate(lower(content), 'öüçşğı', 'oucs gi') LIKE %s"
        )
        params.append(f"%{kw}%")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT DISTINCT source_name
        FROM rag_documents
        WHERE {where_sql}
        LIMIT %s
    """

    print("[SQL]", sql)
    print("[PARAMS]", params + [MAX_TOURS])

    cur.execute(sql, params + [MAX_TOURS])
    tours = [r[0] for r in cur.fetchall()]

    print("[DB] Tours found:", len(tours))
    for t in tours:
        print(" -", t)

    results = []

    for tour in tours:
        cur.execute(
            """
            SELECT content
            FROM rag_documents
            WHERE source_name = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (tour, q_vec, CHUNKS_PER_TOUR)
        )

        chunks = cur.fetchall()
        print(f"[DB] {tour} -> chunks:", len(chunks))

        combined = "\n".join(c[0] for c in chunks)
        results.append((tour, combined))

    cur.close()
    conn.close()

    print("====== END SEARCH ======\n")

    return results

# ---------------- API ----------------
@app.post("/ask")
def ask(q: Question):
    start = time.time()

    rows = search_db_hybrid(q.question)

    context = "\n\n".join(
        f"Tour: {r[0]}\n{r[1]}"
        for r in rows
    )

    prompt = f"""
You are Fez-GPT, a travel assistant for FezTravel.

List ALL tours provided in the context.
Do NOT merge similar tours.
Do NOT omit any tour.

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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_TOKENS
    )

    return {
        "question": q.question,
        "answer": resp.choices[0].message.content,
        "matched_tours": len(rows),
        "latency_sec": round(time.time() - start, 2)
    }
