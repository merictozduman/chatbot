# imports_test.py
# Amaç: sadece kütüphaneler doğru env'de mi kontrol etmek

import sys

print("Python executable:")
print(sys.executable)
print("-" * 40)

try:
    import ollama
    print("✅ ollama import OK")
except Exception as e:
    print("❌ ollama import FAIL:", e)

try:
    import pydantic
    print("✅ pydantic import OK")
except Exception as e:
    print("❌ pydantic import FAIL:", e)

try:
    import psycopg
    print("✅ psycopg import OK")
except Exception as e:
    print("❌ psycopg import FAIL:", e)

try:
    import fastapi
    print("✅ fastapi import OK")
except Exception as e:
    print("❌ fastapi import FAIL:", e)

print("-" * 40)
print("Import testi tamamlandı.")
