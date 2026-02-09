import os
import psycopg
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Eğer VS Code PATH görmüyorsa (güvenli olsun diye)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- CONFIG ----------------
PDF_ROOT = r"C:\Users\damndamn\Desktop\tours\fez"

DB_CONN = {
    "dbname": "vektordb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------- HELPERS ----------------
def embed(text):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def to_pgvector(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i + size] for i in range(0, len(text), size)]

def extract_text_normal(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)

def extract_text_ocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text_pages = []
    for img in images:
        text = pytesseract.image_to_string(img, lang="eng")
        if text.strip():
            text_pages.append(text)
    return "\n".join(text_pages)

# ---------------- MAIN ----------------
inserted = 0

with psycopg.connect(**DB_CONN) as conn:
    with conn.cursor() as cur:

        for root, _, files in os.walk(PDF_ROOT):
            for file in files:
                if not file.lower().endswith(".pdf"):
                    continue

                pdf_path = os.path.join(root, file)
                tour_name = os.path.splitext(file)[0]
                category = os.path.basename(root)

                print(f"Processing: {category} / {file}")

                # 1️⃣ Try normal text
                text = extract_text_normal(pdf_path)

                # 2️⃣ Fallback to OCR
                if not text.strip():
                    print("  🔄 OCR fallback")
                    text = extract_text_ocr(pdf_path)

                if not text.strip():
                    print("  ❌ Still empty, skipped")
                    continue

                chunks = chunk_text(text)

                # 🔥 BURASI KRİTİK DÜZELTME
                for idx, chunk in enumerate(chunks, 1):
                    print(f"    ➜ Embedding chunk {idx}/{len(chunks)}")

                    vec = to_pgvector(embed(chunk))

                    cur.execute(
                        """
                        INSERT INTO rag_documents
                        (source_name, source_type, content, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        (
                            f"{tour_name} - {category}",
                            "pdf",
                            chunk,
                            vec
                        )
                    )

                    conn.commit()
                    inserted += 1

print("✅ OCR PDF embedding completed. Total chunks:", inserted)
