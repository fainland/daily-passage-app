
import os
from openai import OpenAI
import numpy as np
import random
from supabase import create_client
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from flask import Flask, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # 👈 This enables CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load other keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-4"

def get_today_passage():
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    result = supabase.table("passages").select("passage_text").gte("created_at", today_start).execute()

    if result.data:
        return result.data[0]["passage_text"]
    return None

def get_prompt_variation():
    variations = [
        (
            "You are a spiritually attuned creative, channeling one luminous offering each day. "
            "What you write may be a poem, a parable, an image, a confession, or a short prayer. "
            "The only rule is: let it shimmer with originality, truth, and hope. Surprise the reader. Do not repeat yourself."
        ),
        (
            "You are an oracle of subtle truths, whispering one daily insight into the world. "
            "Today’s message should be unlike yesterday’s. Be sparse or lush, strange or simple. "
            "It may be playful, haunting, profound — but it must not be predictable."
        ),
        (
            "You are a weaver of timeless language — a mystic poet whose daily task is to offer a brief yet evocative spark. "
            "Let this be surprising, soulful, experimental in tone. Short and unforgettable."
        ),
        (
            "You write one distilled fragment of beauty, clarity, or hope — no two alike. "
            "Avoid patterns. Avoid sentimentality. Let this feel alive, sacred, and new."
        ),
        (
            "You are to write a very short piece — no more than 120 words — that offers the reader something they’ve never quite heard before. "
            "It may be cryptic or clear, sacred or subversive. But it must not feel familiar. Let it linger in the soul like music."
        )
    ]
    return random.choice(variations)


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def is_similar(new_emb, existing_embs, threshold=0.90):
    new_vec = np.array(new_emb, dtype=np.float32).flatten()

    for vec in existing_embs:
        sim = 1 - cosine(new_vec, vec)
        if sim >= threshold:
            return True
    return False



def fetch_existing_embeddings():
        result = supabase.table("passages").select("embedding").execute()
        cleaned = []

        for row in result.data:
            emb = row.get('embedding')

            # Skip if embedding is missing or malformed
            if not emb or not isinstance(emb, list):
                continue

            # If embedding is nested (e.g., [[...]]), extract the inner list
            if isinstance(emb[0], list):
                emb = emb[0]

            vec = np.array(emb, dtype=np.float32).flatten()

            if vec.shape == (1536,):  # sanity check
                cleaned.append(vec)
            else:
                print("⚠️ Skipping malformed embedding with shape:", vec.shape)

        return cleaned




def store_passage(text, embedding):
    # Flatten just in case it's nested
    if isinstance(embedding[0], list):
        embedding = embedding[0]
    supabase.table("passages").insert({
        "passage_text": text,
        "embedding": embedding
    }).execute()


def generate_passage():
    system_prompt = get_prompt_variation()
    
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Please generate today’s brief passage of luminous insight. "
                    "Keep it under 120 words. Let it surprise and nourish. "
                    "Avoid starting with 'In the...', and avoid repetition of structure or phrase from previous entries."
                )
            }
        ],
        temperature=0.98,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


app = Flask(__name__)

@app.route("/generate-passage", methods=["GET"])
def generate():
    # Check if there's already a passage stored today
    existing_passage = get_today_passage()
    if existing_passage:
        return jsonify({"passage": existing_passage})

    # Try to generate a unique new passage
    MAX_TRIES = 5
    for attempt in range(MAX_TRIES):
        new_passage = generate_passage()
        new_emb = np.array(get_embedding(new_passage))
        existing_embs = fetch_existing_embeddings()

        if is_similar(new_emb, existing_embs):
            print(f"Attempt {attempt + 1}: Too similar, regenerating...")
        else:
            store_passage(new_passage, new_emb.tolist())
            return jsonify({"passage": new_passage})

    return jsonify({"error": "No unique passage found"}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "OK", "message": "Flask is running"})


#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=8080)

# --- MAIN LOGIC ---
#MAX_TRIES = 5
#for attempt in range(MAX_TRIES):
#    new_passage = generate_passage()
#    new_emb = np.array(get_embedding(new_passage))
#    existing_embs = fetch_existing_embeddings()
#
#    if is_similar(new_emb, existing_embs):
#        print(f"Attempt {attempt + 1}: Too similar, regenerating...")
#    else:
#        store_passage(new_passage, new_emb.tolist())
#        print("✅ New unique passage stored:")
#        print(new_passage)
#        break
#else:
#    print("❌ Could not generate a sufficiently unique passage.")
