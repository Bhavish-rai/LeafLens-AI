
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Project root: D:\leaflens_ai
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found in .env")
    sys.exit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)

MODEL = "gemini-3.8-flash"

prompt = """
You are LeafLens AI, a helpful plant disease assistant.
Briefly introduce yourself and explain how you help users
understand plant diseases. Keep your response under 100 words.
"""

for attempt in range(3):
    try:
        print(f"Connecting to Gemini... (attempt {attempt + 1}/3)")

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        result = response.text

        if result:
            print("\nGemini response:")
            print(result)
        else:
            print("Gemini returned an empty response.")

        break

    except errors.APIError as exc:
        print(f"Gemini API error: {exc}")

        code = getattr(exc, "code", None)

        # Retry temporary server errors and rate limits.
        if code not in (429, 500, 502, 503, 504):
            sys.exit(1)

        if attempt < 2:
            delay = 2 ** (attempt + 1)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Gemini is still unavailable. Try again later.")
            sys.exit(1)

    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(1)
