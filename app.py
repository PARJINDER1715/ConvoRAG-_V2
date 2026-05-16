import os
import json
import re
import random
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

from core.persona_engine import PersonaDriftDetector
from core.intent_classifier import (
    IntentClassifier,
    TRAINING_DATA,
    LABELS,
    MODEL_PATH,
    _preprocess
)
from core.conflict_resolver import ConflictResolver
from core.db import (
    init_db,
    get_or_create_session,
    save_message,
    get_messages,
    save_persona,
    get_persona,
    list_sessions,
    delete_session,
    groq_cache_get,
    groq_cache_set,
    groq_cache_stats,
    add_contradiction_pair,
    list_contradiction_pairs,
    toggle_contradiction_pair,
)

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(BASE_DIR, "conversations.csv")
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# =========================================================
# Flask App
# =========================================================

app = Flask(__name__, static_folder="static")
CORS(app)

# =========================================================
# Initialize Database
# =========================================================

init_db()

print("=" * 60)
print("ConvoRAG v3 Starting...")
print("=" * 60)

# =========================================================
# Load CSV Safely
# =========================================================

ALL_MESSAGES = []
DAYS_DATA = []

if not os.path.exists(CSV_PATH):
    print(f"WARNING: CSV file not found -> {CSV_PATH}")

else:
    try:
        df = pd.read_csv(
            CSV_PATH,
            header=None,
            names=["conversation"]
        )

        for day_idx, row in df.iterrows():

            conv = str(row["conversation"])
            day_msgs = []

            for line in conv.strip().split("\n"):

                line = line.strip()

                if line.startswith("User 1:"):

                    msg = {
                        "idx": len(ALL_MESSAGES),
                        "day": int(day_idx),
                        "spk": "User 1",
                        "text": line[8:].strip()
                    }

                    ALL_MESSAGES.append(msg)

                    day_msgs.append({
                        "spk": "U1",
                        "text": line[8:].strip()
                    })

                elif line.startswith("User 2:"):

                    msg = {
                        "idx": len(ALL_MESSAGES),
                        "day": int(day_idx),
                        "spk": "User 2",
                        "text": line[8:].strip()
                    }

                    ALL_MESSAGES.append(msg)

                    day_msgs.append({
                        "spk": "U2",
                        "text": line[8:].strip()
                    })

            if day_msgs:
                DAYS_DATA.append({
                    "day": int(day_idx),
                    "messages": day_msgs
                })

        print(f"Loaded {len(ALL_MESSAGES)} messages")

    except Exception as e:
        print(f"CSV Loading Error: {e}")

# =========================================================
# Build RAG Segments
# =========================================================

TOPIC_SEGMENTS = []

for i in range(0, min(len(ALL_MESSAGES), 2000), 25):

    chunk = ALL_MESSAGES[i:i + 25]

    if not chunk:
        continue

    text = "\n".join(
        f"{m['spk']}: {m['text']}"
        for m in chunk
    )

    TOPIC_SEGMENTS.append({
        "id": i // 25,
        "start": chunk[0]["idx"],
        "end": chunk[-1]["idx"],
        "day": chunk[0]["day"],
        "text": text[:1500],
        "topic": "General"
    })

CHUNKS_100 = []

for i in range(0, min(len(ALL_MESSAGES), 2000), 100):

    chunk = ALL_MESSAGES[i:i + 100]

    if not chunk:
        continue

    text = "\n".join(
        f"{m['spk']}: {m['text']}"
        for m in chunk
    )

    CHUNKS_100.append({
        "id": i // 100,
        "start": chunk[0]["idx"],
        "end": chunk[-1]["idx"],
        "day": chunk[0]["day"],
        "text": text[:3000]
    })

# =========================================================
# Initialize AI Modules
# =========================================================

drift_detector = PersonaDriftDetector()

conflict_resolver = ConflictResolver(
    total_messages=len(ALL_MESSAGES)
)

intent_clf = IntentClassifier()

try:

    if not os.path.exists(MODEL_PATH):

        print("Training Intent Model...")
        intent_clf.train_and_save()

    else:

        intent_clf.load()
        print("Intent Model Loaded")

except Exception as e:
    print(f"Intent Model Error: {e}")

# =========================================================
# Groq Helper
# =========================================================

def groq_call(
    prompt,
    system="",
    model="llama-3.3-70b-versatile",
    max_tokens=500,
    temperature=0.3,
    use_cache=True
):

    cache_prompt = f"{system}\n\n{prompt}".strip()

    if use_cache:

        cached = groq_cache_get(
            model,
            cache_prompt
        )

        if cached:
            return cached

    if not GROQ_API_KEY:
        return "[No GROQ_API_KEY Configured]"

    try:

        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        msgs = []

        if system:
            msgs.append({
                "role": "system",
                "content": system
            })

        msgs.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature
        )

        result = response.choices[0].message.content.strip()

        if use_cache:
            groq_cache_set(
                model,
                cache_prompt,
                result
            )

        return result

    except Exception as e:
        return f"Groq API Error: {e}"

# =========================================================
# Routes
# =========================================================

@app.route("/")
def index():

    try:
        return send_from_directory(
            "static",
            "index.html"
        )

    except:
        return "ConvoRAG v3 Running Successfully!"

@app.route("/api/info")
def info():

    return jsonify({
        "version": "3.0",
        "messages": len(ALL_MESSAGES),
        "days": len(DAYS_DATA),
        "segments": len(TOPIC_SEGMENTS),
        "chunks": len(CHUNKS_100),
        "groq_cache": groq_cache_stats()
    })

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.json or {}

        messages = data.get("messages", [])

        session_id = data.get(
            "session_id",
            str(uuid.uuid4())
        )

        if not messages:
            return jsonify({
                "error": "No messages provided"
            }), 400

        last = messages[-1]

        user_msg = last.get("content", "")

        save_message(
            session_id,
            "user",
            user_msg
        )

        history = get_messages(
            session_id,
            limit=40
        )

        response = groq_call(
            prompt=user_msg,
            system="You are a helpful AI assistant."
        )

        save_message(
            session_id,
            "assistant",
            response
        )

        return jsonify({
            "reply": response,
            "session_id": session_id
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/sessions", methods=["GET"])
def sessions():

    return jsonify(
        list_sessions()
    )

@app.route("/api/sessions/<session_id>", methods=["GET"])
def session_details(session_id):

    return jsonify({
        "session_id": session_id,
        "messages": get_messages(session_id),
        "persona": get_persona(session_id)
    })

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session_route(session_id):

    delete_session(session_id)

    return jsonify({
        "deleted": session_id
    })

@app.route("/api/groq/cache/stats")
def cache_stats():

    return jsonify(
        groq_cache_stats()
    )

# =========================================================
# Run App
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
