
import os
from openai import OpenAI
import numpy as np
from supabase import create_client
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from flask import Flask, jsonify

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load other keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-4"

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
        response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a spiritually attuned creative, channeling fresh, luminous insight each day. "
                    "Your task is to write one short passage per day — but the form it takes is entirely up to you: "
                    "a poem, a blessing, a riddle, a single sentence, a sacred fragment, or a surreal image with emotional resonance. "
                    "What matters is that it evokes wonder, depth, and hope — without repeating past phrases or falling into cliché. "
                    "You are not writing a motivational poster. You are whispering something timeless, alive, and unexpected into the heart of the reader. "
                    "Each day's passage should surprise and nourish. Let your voice be experimental, humble, and bold. Keep it brief. Let silence speak as much as words."
                )
            },
            {
                "role": "user",
                "content": (
                    "Please generate today's passage of hope and reflection. Limit it to 120 words or fewer. "
                    "Avoid conventional phrases, avoid repetition of earlier content, and feel free to use unusual structures or metaphor. "
                    "Let this one be unlike the last."
                )
            }
        ]
,
        temperature=0.95,
        max_tokens=300
    )
        return response.choices[0].message.content.strip()

app = Flask(__name__)

@app.route("/generate-passage", methods=["GET"])
def generate():
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
