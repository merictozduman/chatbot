import requests
from bs4 import BeautifulSoup
import psycopg
import os
from openai import OpenAI

URL = "https://www.feztravel.com/fez/FAQ"

DB_CONN = {
    "dbname": "vektordb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def to_pgvector(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"

print("Downloading HTML...")
html = requests.get(URL).text
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(separator=" ")

chunks = [text[i:i+800] for i in range(0, len(text), 800)]

with psycopg.connect(**DB_CONN) as conn:
    with conn.cursor() as cur:
        for i, chunk in enumerate(chunks):
            vec = to_pgvector(embed(chunk))
            cur.execute(
                """
                INSERT INTO rag_documents (source_name, source_type, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                ("fez_faq.html", "html", chunk, vec)
            )
    conn.commit()

print("Embedding completed:", len(chunks), "chunks")
