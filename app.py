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

# ---------------- CANONICAL KEYWORD STEMS ----------------
KEYWORD_STEMS = {
    "gobek": ["gobek", "göbek", "gobekli", "gobekli tepe", "gobeklitep"],
    "cappadocia": ["kapadok", "kapadokya", "cappadoc"],
    "pamukkale": ["pamukkal"],
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

# ---------------- CANONICAL PARSER ----------------
def parse_canonical_keywords(question: str):
    q = normalize_text(question)
    found = set()

    for canonical, stems in KEYWORD_STEMS.items():
        for stem in stems:
            if normalize_text(stem) in q:
                found.add(canonical)

    return list(found)

# ---------------- HYBRID DB SEARCH (ONLY WHEN KEYWORD EXISTS) ----------------
def search_db_hybrid(question: str, canonical_keywords: list):
    q_vec = to_pgvector(embed(question))

    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    where_clauses = ["source_type = 'pdf'"]
    params = []
    or_blocks = []

    for canonical in canonical_keywords:
        stems = KEYWORD_STEMS.get(canonical, [])
        stem_conditions = []

        for stem in stems:
            stem_conditions.append(
                "translate(lower(content),'öüçşğı','oucsg') LIKE %s"
            )
            params.append(f"%{normalize_text(stem)}%")

        if stem_conditions:
            or_blocks.append("(" + " OR ".join(stem_conditions) + ")")

    if or_blocks:
        where_clauses.append("(" + " OR ".join(or_blocks) + ")")

    where_sql = " AND ".join(where_clauses)

    cur.execute(
        f"""
        SELECT DISTINCT source_name
        FROM rag_documents
        WHERE {where_sql}
        ORDER BY source_name
        LIMIT %s
        """,
        params + [MAX_TOURS]
    )

    tours = [r[0] for r in cur.fetchall()]

    print("\n🔎 MATCHED TOURS FROM DB:")
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
        combined = "\n".join(c[0] for c in chunks)
        results.append((tour, combined))

    cur.close()
    conn.close()
    return results

# ---------------- API ----------------
@app.post("/ask")
def ask(q: Question):
    start = time.time()

    canonical_keywords = parse_canonical_keywords(q.question)

    # ✅ SADECE keyword varsa DB tur listesi
    if canonical_keywords:
        rows = search_db_hybrid(q.question, canonical_keywords)
        tour_names = [tour for tour, _ in rows]

        return {
            "question": q.question,
            "tours": tour_names,
            "answer": None,
            "matched_tours": len(tour_names),
            "latency_sec": round(time.time() - start, 2)
        }

    # 🔁 AKSİ HALDE → VECTOR + LLM
    prompt = f"""
You are Fez-GPT, a travel assistant for FezTravel.

Answer the question naturally and informatively.

Question:
{q.question}
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=MAX_TOKENS
    )

    return {
        "question": q.question,
        "tours": [],
        "answer": resp.choices[0].message.content,
        "matched_tours": 0,
        "latency_sec": round(time.time() - start, 2)
    }
