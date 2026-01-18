import re
import psycopg
from collections import Counter

# ---------------- DB CONFIG ----------------
DB_CONN = {
    "dbname": "vektordb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

# ---------------- SEED LISTS ----------------

TRANSPORT_WORDS = [
    "flight", "flights", "transfer", "transfers",
    "bus", "coach", "ferry", "train"
]

COUNTRY_WORDS = [
    "turkey", "greece", "egypt",
    "italy", "france", "spain",
    "portugal", "jordan"
]

# Cümle başı / genel kelimeleri elemek için
STOP_WORDS = {
    "Accommodation", "After", "Arrival", "Airport", "Adventure",
    "About", "Before", "During", "Includes", "Included",
    "Day", "Days", "Tour", "Tours", "Hotel", "Hotels",
    "Transfer", "Transfers", "Flight", "Flights",
    "Breakfast", "Dinner", "Lunch",
    "Morning", "Evening", "Afternoon",
    "Price", "Prices"
}

# Location bağlamı
COUNTRY_CONTEXT = [
    "in turkey", "in greece", "in egypt",
    "through turkey", "across turkey",
    "visit", "visiting", "visited",
    "located in"
]

# ---------------- COLLECTIONS ----------------
location_counter = Counter()
transport_found = set()
countries_found = set()
price_patterns_found = set()

# ---------------- DB READ ----------------
conn = psycopg.connect(**DB_CONN)
cur = conn.cursor()

cur.execute("""
    SELECT content
    FROM rag_documents
    WHERE source_type = 'pdf'
""")

rows = cur.fetchall()

# ---------------- PROCESS ----------------
for (content,) in rows:
    content_lower = content.lower()

    # ---- TRANSPORT DISCOVERY ----
    for t in TRANSPORT_WORDS:
        if re.search(r"\b" + re.escape(t) + r"\b", content_lower):
            transport_found.add(t)

    # ---- COUNTRY DISCOVERY ----
    for c in COUNTRY_WORDS:
        if re.search(r"\b" + re.escape(c) + r"\b", content_lower):
            countries_found.add(c)

    # ---- PRICE DISCOVERY ----
    # Örnekler: 1999 USD, $2490, EUR 3200
    for m in re.findall(
        r"(\$?\b\d{3,5}\b\s?(usd|eur|\$)?)",
        content_lower
    ):
        price_patterns_found.add(m[0].strip())

    # ---- LOCATION DISCOVERY (CONTEXT BASED) ----
    lines = content.splitlines()

    for line in lines:
        line_lower = line.lower()

        if any(ctx in line_lower for ctx in COUNTRY_CONTEXT):
            for m in re.findall(r"\b[A-Z][a-z]{3,}\b", line):
                if m not in STOP_WORDS:
                    location_counter[m] += 1

cur.close()
conn.close()

# ---------------- POST FILTERING ----------------

# En az 3 kez geçenleri al (gürültüyü azaltır)
locations_clean = [
    loc for loc, cnt in location_counter.items()
    if cnt >= 3
]

# ---------------- OUTPUT ----------------

print("\n📍 DISCOVERED LOCATIONS (frequency >= 3):")
print(sorted(locations_clean))

print("\n✈️ DISCOVERED TRANSPORT TYPES:")
print(sorted(transport_found))

print("\n🌍 DISCOVERED COUNTRIES:")
print(sorted(countries_found))

print("\n💰 DISCOVERED PRICE PATTERNS (sample):")
print(sorted(list(price_patterns_found))[:20])
