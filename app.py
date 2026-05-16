import os, json, re, random, time, uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

from core.persona_engine    import PersonaDriftDetector
from core.intent_classifier import IntentClassifier, TRAINING_DATA, LABELS, MODEL_PATH, _preprocess
from core.conflict_resolver import ConflictResolver
from core.db import (
    init_db,
    get_or_create_session, save_message, get_messages, save_persona, get_persona,
    list_sessions, delete_session,
    groq_cache_get, groq_cache_set, groq_cache_stats,
    add_contradiction_pair, list_contradiction_pairs, toggle_contradiction_pair,
)

load_dotenv()

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.environ.get("CSV_PATH", os.path.join(BASE_DIR, "conversations.csv"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Bootstrap DB ──────────────────────────────────────────────────────────────
init_db()
print("="*60)
print("  ConvoRAG v3 — Loading...")
print("="*60)

# ── Load CSV ──────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, header=None, names=["conversation"])
ALL_MESSAGES = []
DAYS_DATA    = []

for day_idx, row in df.iterrows():
    conv     = str(row["conversation"])
    day_msgs = []
    for line in conv.strip().split("\n"):
        line = line.strip()
        if line.startswith("User 1:"):
            m = {"idx": len(ALL_MESSAGES), "day": int(day_idx), "spk": "User 1", "text": line[8:].strip()}
            ALL_MESSAGES.append(m); day_msgs.append({"spk": "U1", "text": line[8:].strip()})
        elif line.startswith("User 2:"):
            m = {"idx": len(ALL_MESSAGES), "day": int(day_idx), "spk": "User 2", "text": line[8:].strip()}
            ALL_MESSAGES.append(m); day_msgs.append({"spk": "U2", "text": line[8:].strip()})
    if day_msgs:
        DAYS_DATA.append({"day": int(day_idx), "messages": day_msgs})

print(f"  {len(ALL_MESSAGES):,} messages | {len(DAYS_DATA)} days")

# ── Build in-memory RAG index ─────────────────────────────────────────────────
TOPIC_SEGMENTS = []
for i in range(0, min(len(ALL_MESSAGES), 2000), 25):
    chunk = ALL_MESSAGES[i:i+25]
    if not chunk: continue
    text = "\n".join(f"{m['spk']}: {m['text']}" for m in chunk)
    TOPIC_SEGMENTS.append({
        "id": i//25, "start": chunk[0]["idx"], "end": chunk[-1]["idx"],
        "day": chunk[0]["day"], "day_start": chunk[0]["day"],
        "text": text[:1500], "topic": "General"
    })

CHUNKS_100 = []
for i in range(0, min(len(ALL_MESSAGES), 2000), 100):
    chunk = ALL_MESSAGES[i:i+100]
    if not chunk: continue
    text = "\n".join(f"{m['spk']}: {m['text']}" for m in chunk)
    CHUNKS_100.append({
        "id": i//100, "start": chunk[0]["idx"], "end": chunk[-1]["idx"],
        "day": chunk[0]["day"], "text": text[:3000]
    })

# ── Init modules ──────────────────────────────────────────────────────────────
drift_detector    = PersonaDriftDetector()
conflict_resolver = ConflictResolver(total_messages=len(ALL_MESSAGES))
intent_clf        = IntentClassifier()

if not os.path.exists(MODEL_PATH):
    print("  Training intent classifier...")
    intent_clf.train_and_save()
else:
    intent_clf.load()
    print(f"  Intent model loaded ({os.path.getsize(MODEL_PATH)//1024}KB)")

print("="*60 + "\n")


# ── Groq helper with idempotency cache ───────────────────────────────────────

def groq_call(prompt: str, system: str = "", model: str = "llama-3.3-70b-versatile",
              max_tokens: int = 500, temperature: float = 0.3,
              use_cache: bool = True) -> str:
    """
    Call Groq with transparent caching.
    Same (model, system+prompt) never hits the API twice.
    """
    cache_prompt = f"{system}\n\n{prompt}".strip()

    if use_cache:
        cached = groq_cache_get(model, cache_prompt)
        if cached is not None:
            return cached

    if not GROQ_API_KEY:
        return "[No GROQ_API_KEY configured]"

    from groq import Groq
    client  = Groq(api_key=GROQ_API_KEY)
    msgs    = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    resp    = client.chat.completions.create(
        model=model, messages=msgs,
        max_tokens=max_tokens, temperature=temperature
    )
    result  = resp.choices[0].message.content.strip()

    if use_cache:
        groq_cache_set(model, cache_prompt, result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Static
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/info")
def info():
    return jsonify({
        "version": "3.0",
        "total_messages": len(ALL_MESSAGES),
        "total_days": len(DAYS_DATA),
        "topic_segments": len(TOPIC_SEGMENTS),
        "chunks_100": len(CHUNKS_100),
        "intent_model_kb": os.path.getsize(MODEL_PATH)//1024 if os.path.exists(MODEL_PATH) else 0,
        "groq_cache": groq_cache_stats(),
        "parts": ["persona-drift", "intent-classifier", "conflict-resolver", "rag-query"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Persona Drift
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/persona/drift", methods=["POST"])
def persona_drift():
    data   = request.json or {}
    n_days = min(int(data.get("days", 15)), len(DAYS_DATA), 30)
    result = drift_detector.analyze(DAYS_DATA[:n_days])
    return jsonify(result)


@app.route("/api/persona/timeline", methods=["GET"])
def persona_timeline():
    n      = min(15, len(DAYS_DATA))
    result = drift_detector.analyze(DAYS_DATA[:n])
    return jsonify({"timeline": result["timeline"], "drifts": result["drift_events"], "summary": result["summary"]})


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Intent Classifier
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/intent", methods=["POST"])
def classify_intent():
    data    = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    return jsonify(intent_clf.predict(message))


@app.route("/api/intent/batch", methods=["POST"])
def classify_batch():
    data     = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    results = intent_clf.predict_batch(messages[:50])
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/intent/demo", methods=["GET"])
def intent_demo():
    random.seed(42)
    u1_msgs = [m for m in ALL_MESSAGES if m["spk"] == "User 1" and len(m["text"]) > 10]
    samples = random.sample(u1_msgs, min(20, len(u1_msgs)))
    results = []
    for m in samples:
        pred = intent_clf.predict(m["text"])
        results.append({
            "day": m["day"], "message": m["text"][:100],
            "intent": pred["intent"], "confidence": pred["confidence"],
            "latency_ms": pred["latency_ms"],
        })
    return jsonify({"demo_results": results, "model_size_kb": os.path.getsize(MODEL_PATH)//1024})


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Conflict Resolver
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/rag/resolve", methods=["POST"])
def resolve_conflict():
    data  = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    stopwords = {"did","i","mention","about","the","a","an","my","me","you",
                 "was","were","is","are","have","has","do","does","can","could",
                 "would","should","what","when","where","how","any","some"}
    query_words = set(re.findall(r'\w+', query.lower())) - stopwords

    matching_chunks = []
    for seg in TOPIC_SEGMENTS:
        text_words = set(re.findall(r'\w+', seg["text"].lower()))
        overlap    = len(query_words & text_words)
        if overlap > 0:
            sc = dict(seg); sc["_overlap"] = overlap
            sc["start_msg"] = seg["start"]; sc["end_msg"] = seg["end"]
            matching_chunks.append(sc)

    matching_chunks.sort(key=lambda x: x["_overlap"], reverse=True)
    top_chunks = matching_chunks[:6]
    result     = conflict_resolver.resolve(query, top_chunks)

    return jsonify({
        "query": query,
        "answer": result.answer,
        "contradictions_found": result.contradictions_found,
        "contradiction_details": result.contradiction_details,
        "merge_strategy": result.merge_strategy,
        "confidence": result.confidence,
        "chunks_found": len(top_chunks),
        "ranked_chunks": [
            {
                "chunk_id": c.chunk_id, "topic": c.topic, "day": c.day,
                "start_msg": c.start_msg, "end_msg": c.end_msg,
                "relevance_score": c.relevance_score, "recency_score": c.recency_score,
                "emotional_weight": c.emotional_weight, "emotion_valence": c.emotion_valence,
                "final_score": c.final_score, "snippet": c.text[:200],
            }
            for c in result.ranked_chunks[:5]
        ],
    })


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    data       = request.json or {}
    query      = data.get("query", "").strip()
    topic_cps  = data.get("topic_checkpoints", [])
    chunk_sums = data.get("chunk_summaries", [])
    if not query:
        return jsonify({"error": "No query"}), 400

    def score(text):
        qw = set(query.lower().split()); tw = set(text.lower().split())
        return len(qw & tw) / (len(qw | tw) + 1e-9)

    seg_hits = sorted(TOPIC_SEGMENTS, key=lambda x: score(x["text"]), reverse=True)[:3]
    t_hits   = sorted(topic_cps,  key=lambda x: score(x.get("summary","")), reverse=True)[:3] if topic_cps  else []
    c_hits   = sorted(chunk_sums, key=lambda x: score(x.get("summary","")), reverse=True)[:3] if chunk_sums else []

    context = "\n\n".join([
        "TOPIC SUMMARIES:\n"  + "\n".join(f"• {t.get('topic','')}: {t.get('summary','')[:200]}"              for t in t_hits),
        "CHUNK SUMMARIES:\n" + "\n".join(f"• Msgs {c.get('start')}-{c.get('end')}: {c.get('summary','')[:200]}" for c in c_hits),
        "RAW EXCERPTS:\n"    + "\n---\n".join(s["text"][:400] for s in seg_hits),
    ])

    try:
        answer = groq_call(
            prompt=f"Question: {query}\n\nContext:\n{context}",
            system="Answer questions about conversation history using only the provided context. Be specific and concise.",
            max_tokens=500,
        )
    except Exception as e:
        answer = f"[API error: {e}]\n\nContext retrieved:\n{context[:500]}"

    return jsonify({"answer": answer, "sources": {"segments": len(seg_hits), "topics": len(t_hits), "chunks": len(c_hits)}})


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Chat  (now with persistent sessions)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.json or {}
    messages   = data.get("messages", [])
    persona    = data.get("persona", {})
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not messages:
        return jsonify({"error": "No messages"}), 400

    # Persist incoming user message
    last = messages[-1]
    if last.get("role") == "user":
        save_message(session_id, "user", last["content"])

    if not GROQ_API_KEY:
        return jsonify({"reply": "No GROQ_API_KEY configured.", "session_id": session_id})

    user_msg = last.get("content", "")

    def score(text):
        qw = set(user_msg.lower().split()); tw = set(text.lower().split())
        return len(qw & tw) / (len(qw | tw) + 1e-9)

    seg_hits = sorted(TOPIC_SEGMENTS, key=lambda x: score(x["text"]), reverse=True)[:2]
    raw_ctx  = "\n---\n".join(s["text"][:300] for s in seg_hits)
    p_ctx    = f"PERSONA:\n{json.dumps(persona, indent=2)}" if persona else ""

    # Load full history from DB (up to 40 messages for context window)
    history = get_messages(session_id, limit=40)
    system  = (
        f"You are an assistant who knows this user from their conversations.\n"
        f"{p_ctx}\nRELEVANT EXCERPTS:\n{raw_ctx}\nAnswer warmly and specifically."
    )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + history,
            max_tokens=600
        )
        reply = resp.choices[0].message.content
        save_message(session_id, "assistant", reply)
        return jsonify({"reply": reply, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Sessions API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def sessions_list():
    return jsonify(list_sessions())


@app.route("/api/sessions/<session_id>", methods=["GET"])
def session_get(session_id):
    return jsonify({
        "session_id": session_id,
        "messages": get_messages(session_id, limit=200),
        "persona": get_persona(session_id),
    })


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def session_delete(session_id):
    delete_session(session_id)
    return jsonify({"deleted": session_id})


# ─────────────────────────────────────────────────────────────────────────────
# Groq cache stats
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/groq/cache/stats", methods=["GET"])
def cache_stats():
    return jsonify(groq_cache_stats())


# ─────────────────────────────────────────────────────────────────────────────
# Contradiction pairs API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/contradiction_pairs", methods=["GET"])
def contradiction_pairs_list():
    return jsonify(list_contradiction_pairs())


@app.route("/api/contradiction_pairs", methods=["POST"])
def contradiction_pairs_add():
    data = request.json or {}
    pos  = data.get("pos_words", [])
    neg  = data.get("neg_words", [])
    if not pos or not neg:
        return jsonify({"error": "pos_words and neg_words required"}), 400
    pair_id = add_contradiction_pair(pos, neg, data.get("label", ""))
    return jsonify({"id": pair_id, "message": "Pair added — active immediately"})


@app.route("/api/contradiction_pairs/<int:pair_id>", methods=["PATCH"])
def contradiction_pairs_toggle(pair_id):
    data   = request.json or {}
    active = bool(data.get("active", True))
    toggle_contradiction_pair(pair_id, active)
    return jsonify({"id": pair_id, "active": active})


# ─────────────────────────────────────────────────────────────────────────────
# Remaining original endpoints (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/segments")
def segments():
    return jsonify(TOPIC_SEGMENTS)


@app.route("/api/chunks100")
def chunks100():
    return jsonify(CHUNKS_100)


@app.route("/api/process_segment", methods=["POST"])
def process_segment():
    data       = request.json
    seg        = data["segment"]
    prev_topic = data.get("prev_topic", "none")

    prompt = (
        f'Previous topic: "{prev_topic}"\n\n'
        f'Conversation segment:\n{seg["text"][:800]}\n\n'
        f'Return ONLY valid JSON with no extra text, no markdown fences:\n'
        f'{{"topic":"short phrase","changed":true,"summary":"2-3 sentence summary"}}'
    )
    last_error = "Unknown"
    for attempt in range(3):
        try:
            raw   = groq_call(prompt, system="You are a JSON-only responder. Output raw JSON.", max_tokens=300, temperature=0.2, use_cache=True)
            clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
            clean = re.sub(r'\s*```$', '', clean).strip()
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if not match:
                last_error = f"No JSON in: {raw[:200]}"; continue
            parsed = json.loads(match.group())
            return jsonify({"topic": str(parsed.get("topic","General")), "changed": bool(parsed.get("changed",False)), "summary": str(parsed.get("summary",""))})
        except Exception as e:
            last_error = str(e); time.sleep(2 * (attempt+1))
    return jsonify({"topic": "General", "changed": False, "summary": f"Failed: {last_error}"})


@app.route("/api/summarize_chunk", methods=["POST"])
def summarize_chunk():
    chunk  = (request.json or {}).get("chunk", {})
    prompt = (
        f'Summarize the following conversation (messages {chunk.get("start")}–{chunk.get("end")}) '
        f'in 3-4 concise sentences:\n\n{chunk.get("text","")[:2000]}'
    )
    try:
        summary = groq_call(prompt, max_tokens=300, temperature=0.3, use_cache=True)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e), "summary": f"Failed: {e}"}), 500


@app.route("/api/extract_persona", methods=["POST"])
def extract_persona_route():
    session_id = (request.json or {}).get("session_id")
    random.seed(42)
    u1    = [m for m in ALL_MESSAGES if m["spk"] == "User 1" and len(m["text"]) > 15]
    texts = "\n".join(m["text"] for m in random.sample(u1, min(150, len(u1))))
    try:
        raw   = groq_call(
            prompt=f'Messages:\n{texts}\n\nReturn ONLY JSON:\n{{"habits":[],"personal_facts":[],"personality_traits":[],"interests":[],"communication_style":{{"tone":"","message_length":"","emoji_usage":"","patterns":[]}},"life_situation":"","summary":""}}',
            system="Extract user persona. Return strict JSON only.",
            max_tokens=900, use_cache=True
        )
        clean  = re.sub(r'^```(?:json)?', '', raw.strip()).rstrip('`').strip()
        persona = json.loads(clean)
        if session_id:
            save_persona(session_id, persona)
        return jsonify(persona)
    except Exception as e:
        return jsonify({"_error": str(e)}), 500


@app.route("/api/index_checkpoints", methods=["POST"])
def index_checkpoints():
    return jsonify({"topic_checkpoints_indexed": 0, "chunk_summaries_indexed": 0, "note": "Using keyword search in v3"})


if __name__ == "__main__":
    print("🚀 ConvoRAG v3 at http://localhost:5000\n")
    app.run(debug=True, port=5000)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
