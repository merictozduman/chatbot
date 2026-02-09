from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg
import time
import os
from difflib import SequenceMatcher
from openai import OpenAI

# ---------------- CONFIG ----------------
DB_CONN = {
    "dbname": "vektordb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

GPT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 900

EXISTING_TOUR_MAX_ROWS = 60
EXISTING_TOUR_MAX_CHARS = 16000

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SESSION STATE ----------------
SESSIONS = {}

# ---------------- APP ----------------
app = FastAPI(title="Fez-GPT")

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- GENERAL INFO PROMPT ----------------
GENERAL_INFO_PROMPT = """
You are a professional and friendly sales assistant for Fez Travel.

You MUST answer questions ONLY using the official Fez Travel information below.
Do NOT invent information.
If the answer is not covered below, politely say that the user should contact Fez Travel.

=====================
OFFICIAL FAQ CONTENT
=====================

Q: What is the best time of year to take a tour?
A:
Fez Travel tours run year-round. Spring (April–June) and autumn (September–November) are ideal.
November–March offers mild climate and discounted prices.
Summer (July–August) is best for coastal tours like Gulet cruises but can be hot.

Q: Are the tours suitable for solo travellers?
A:
Yes. Solo travellers are welcome.
Single travellers must book a single room. Shared rooms are not available.

Q: What type of transportation is used?
A:
Modern air-conditioned minibuses or coaches.
Some tours include domestic flights.
Gulet cruises use traditional wooden boats.

Q: How much free time is available?
A:
Tours are guided and structured.
Free time may be available:
- Arrival & departure days
- Evenings
- Pre/Post tour accommodation
- Private tailor-made tours offer full flexibility

Q: What currency should I bring?
A:
Turkish Lira (TRY).
Credit cards widely accepted.
Carry some cash.
Euro and USD are commonly accepted.

Q: Will I need a visa?
A:
Depends on nationality.
Check https://www.evisa.gov.tr

Q: Can I request vegetarian or gluten-free meals?
A:
Yes, inform Fez Travel in advance.

Q: Are hotels centrally located?
A:
Yes, boutique hotels with easy access to key sites.

Q: Will hotels have Wi-Fi?
A:
Most hotels provide free Wi-Fi (speed may vary).

Q: Can I upgrade hotels?
A:
Group tours: hotel change not possible.
Room upgrades may be available.
Private tailor-made tours allow full customization.

Q: What clothing should I pack?
A:
Spring/Autumn: layers
Summer: light clothing
Winter: warm clothing (especially Eastern Turkey)

Q: How much walking is involved?
A:
Moderate walking. Comfortable shoes recommended.

Q: Is there a luggage limit?
A:
One checked bag + one carry-on.
Domestic flights usually allow 20–25 kg.

Q: Are airport transfers available?
A:
Yes, usually included on first and last day.
Pre/Post accommodation can adjust transfers.
If hotel is not booked via Fez Travel, transfers are optional.

Q: What optional activities are available?
A:
Listed in Trip Notes.
Sent 3–4 weeks before departure.
Booked with guide or in advance for Istanbul.

Q: Can I skip activities?
A:
Yes. Inform your guide one day before.

Q: Do I need to book optional activities in advance?
A:
Most can be booked during the tour.
Istanbul activities should be booked in advance.

Q: Are there dress codes for religious sites?
A:
Yes. Modest clothing required.

Q: Is tipping expected?
A:
Customary but not mandatory.
Details in Trip Notes.

Q: Will we try local food?
A:
Yes. Authentic Turkish cuisine included.

Q: Shopping opportunities?
A:
Yes. Grand Bazaar, pottery workshops, local markets.

Q: Is tap water safe?
A:
Bottled water recommended.

Q: Do I need travel insurance?
A:
Yes, strongly recommended.

Q: Vaccinations required?
A:
No mandatory vaccines.

Q: Medical emergencies?
A:
Guide will assist.
Modern hospitals available.

Q: SIM cards & mobile data?
A:
Available at airports and cities.
eSIM supported.

Q: Charging outlets on buses?
A:
Some buses have outlets.
Power bank recommended.

=====================
END OF OFFICIAL INFO
=====================

Answer the user clearly and politely.

"""

# ---------------- MODELS ----------------
class Question(BaseModel):
    question: str
    session_id: str | None = None

# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(txt: str) -> str:
    return (
        (txt or "")
        .lower()
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ç", "c")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ı", "i")
        .strip()
    )

# ---------------- LLM HELPER ----------------
def llm_answer(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )
    return resp.choices[0].message.content.strip()

# ---------------- EXISTING TOUR HELPERS ----------------
def find_tour_candidates_by_name(tour_name: str):
    norm = normalize_text(tour_name)
    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT source_name
        FROM rag_documents
        WHERE source_type = 'pdf'
          AND translate(lower(source_name),'öüçşğı','oucsgi') LIKE %s
    """, (f"%{norm}%",))

    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

def pick_best_candidate(user_input: str, candidates: list[str]):
    u = normalize_text(user_input)
    best, score = None, -1
    for c in candidates:
        s = SequenceMatcher(None, u, normalize_text(c)).ratio()
        if s > score:
            best, score = c, s
    return best

def get_tour_content(source_name: str):
    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    cur.execute("""
        SELECT content
        FROM rag_documents
        WHERE source_type='pdf'
          AND source_name=%s
        ORDER BY id
        LIMIT %s
    """, (source_name, EXISTING_TOUR_MAX_ROWS))

    text = "\n".join(r[0] for r in cur.fetchall())
    cur.close()
    conn.close()

    return text[:EXISTING_TOUR_MAX_CHARS]

# ---------------- SUITABLE TOURS INPUT PARSER ----------------
def parse_period_location(user_text: str):
    """
    Accepts:
      - 202611-gallipoli
      - 2026-11-gallipoli
      - 2026-11-Gallipoli  (note: two hyphens before location)
    Returns:
      (period_yyyymm='YYYY-MM', location='...')
    """
    raw = (user_text or "").strip()

    # Case A: starts with YYYY-MM-...
    # Example: 2026-11-Gallipoli
    if len(raw) >= 8 and raw[4] == "-" and raw[7] == "-":
        yyyy = raw[0:4]
        mm = raw[5:7]
        loc = raw[8:].strip()
        if yyyy.isdigit() and mm.isdigit() and len(mm) == 2:
            return f"{yyyy}-{mm}", loc

    # Case B: YYYYMM-location (single hyphen after YYYYMM)
    # Example: 202611-gallipoli
    if "-" in raw:
        left, right = raw.split("-", 1)
        left = left.strip()
        right = right.strip()
        if left.isdigit() and len(left) == 6:
            yyyy = left[0:4]
            mm = left[4:6]
            return f"{yyyy}-{mm}", right

    raise ValueError("Invalid format")

# ---------------- SUITABLE TOURS DB SEARCH (rag_detail_search) ----------------
def search_suitable_tours(period_yyyymm: str, location: str):
    """
    DÜZELTİLMİŞ VERSİYON:
    - period_yyyymm -> LIKE '%YYYYMM%'
    - location -> lower + translate + LIKE
    """

    conn = psycopg.connect(**DB_CONN)
    cur = conn.cursor()

    # Normalize kullanıcı girdisi
    loc_norm = normalize_text(location)

    print("\n[SUITABLE] INPUT -> period_yyyymm:", period_yyyymm, " | location:", loc_norm)

    cur.execute("""
        SELECT DISTINCT source_name
        FROM rag_detail_search
        WHERE 
            translate(lower(period_yyyymm),'öüçşğı','oucsgi') LIKE %s
        AND 
            translate(lower(btrim(location)),'öüçşğı','oucsgi') LIKE %s
        ORDER BY source_name
    """, (f"%{period_yyyymm.replace('-', '')}%", f"%{loc_norm}%"))

    tours = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    print("[SUITABLE] FOUND TOURS:", len(tours))
    for t in tours[:50]:
        print(" -", t)

    return tours


# ---------------- API ----------------
@app.post("/ask")
def ask(q: Question):
    session_id = q.session_id or "default"
    user_text = (q.question or "").strip()

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"step": "INIT"}

    step = SESSIONS[session_id]["step"]

    # 1️⃣ INIT
    if step == "INIT":
        SESSIONS[session_id]["step"] = "MODE"
        return {
            "session_id": session_id,
            "answer": (
                "Would you like information about an existing tour, suitable tours, "
                "or general information?\n\n"
                "- existing tour\n"
                "- suitable tours\n"
                "- general information\n\n"
                "Note: Please refresh the page when switching between modes to ensure a smooth experience.                              "
                "   "
            )
        }

    # 2️⃣ MODE SELECTION
    if step == "MODE":
        t = normalize_text(user_text)

        if t == "existing tour":
            SESSIONS[session_id]["step"] = "EXISTING_TOUR"
            return {"session_id": session_id, "answer": "Please type the tour name."}

        if t == "suitable tours":
            SESSIONS[session_id]["step"] = "SUITABLE_INPUT"
            return {
                "session_id": session_id,
                "answer": (
                    "Please tell me:\n"
                    "- When you plan to travel\n"
                    "- Where you want to go\n\n"
                    "Format: YYYYMM-location (e.g. 202606-istanbul)\n"
                    "You can also use the search feature on https://www.feztravel.com/ from the main menu."
                )
            }

        if t == "general information":
            SESSIONS[session_id]["step"] = "GENERAL_INFO"
            return {"session_id": session_id, "answer": "Please ask your general question about Fez Travel."}

        return {
            "session_id": session_id,
            "answer": "Please choose: existing tour / suitable tours / general information"
        }

    # 3️⃣ GENERAL INFO
    if step == "GENERAL_INFO":
        return {
            "session_id": session_id,
            "answer": llm_answer(f"{GENERAL_INFO_PROMPT}\n\nUser question:\n{user_text}")
        }

    # 4️⃣ EXISTING TOUR
    if step == "EXISTING_TOUR":
        candidates = find_tour_candidates_by_name(user_text)
        chosen = pick_best_candidate(user_text, candidates)

        if not chosen:
            return {
                "session_id": session_id,
                "answer": "Tour not found. Please check the name and try again."
            }

        content = get_tour_content(chosen)

        prompt = f"""
Provide a detailed explanation of the tour below using ONLY the provided content.
Do NOT tell the user to contact Fez Travel unless the content is genuinely insufficient.

Tour name:
{chosen}

Content:
{content}
"""
        SESSIONS[session_id]["step"] = "INIT"
        return {
            "session_id": session_id,
            "tour": chosen,
            "answer": llm_answer(prompt)
        }

    # 5️⃣ SUITABLE TOURS (rag_detail_search)
    if step == "SUITABLE_INPUT":
        try:
            period_yyyymm, loc = parse_period_location(user_text)
        except:
            return {
                "session_id": session_id,
                "answer": "Invalid format. Please use YYYYMM-location (e.g. 202606-istanbul)"
            }

        # Location match should be user-friendly (case-insensitive)
        # We pass the raw loc (trimmed) to ILIKE
        loc = loc.strip()
        tours = search_suitable_tours(period_yyyymm, loc)

        SESSIONS[session_id]["step"] = "INIT"

        if not tours:
            return {
                "session_id": session_id,
                "answer": "No tours found for the given date and location."
            }

        return {
            "session_id": session_id,
            "answer": "Available tours:\n" + "\n".join(f"- {t}" for t in tours),
            "tours": tours
        }

    return {
        "session_id": session_id,
        "answer": "Something went wrong. Please refresh and try again."
    }
 