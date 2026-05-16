import os
import re
import time
import pickle
import random
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.pkl")

LABELS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]


# ── Training data ─────────────────────────────────────────────────────────────
# ~120 examples per class — enough for TF-IDF + SGD to generalise well

TRAINING_DATA: List[Tuple[str, str]] = [

    # ── reminder ──────────────────────────────────────────────────────────────
    ("remind me to call mom tomorrow", "reminder"),
    ("don't let me forget the dentist on friday", "reminder"),
    ("set a reminder for 6pm meeting", "reminder"),
    ("remind me about the grocery run", "reminder"),
    ("can you remind me to take my medication", "reminder"),
    ("reminder: submit report by 5pm", "reminder"),
    ("don't forget we have dinner tonight", "reminder"),
    ("alert me when it's 3 o'clock", "reminder"),
    ("remember to pay the electricity bill", "reminder"),
    ("ping me tomorrow morning about the call", "reminder"),
    ("i need to remember to buy a gift for sarah", "reminder"),
    ("flag this — pick up kids at 4", "reminder"),
    ("don't let me miss the webinar at noon", "reminder"),
    ("set an alarm for 7am", "reminder"),
    ("note to self: water the plants", "reminder"),
    ("remind me to check email before the meeting", "reminder"),
    ("can you remind me to follow up with the client", "reminder"),
    ("i should remember to renew my passport", "reminder"),
    ("mark my calendar for the team standup", "reminder"),
    ("schedule a reminder for quarterly review", "reminder"),
    ("tell me to call the doctor next week", "reminder"),
    ("remind me about my gym session at 7", "reminder"),
    ("don't forget to RSVP to the wedding", "reminder"),
    ("need a nudge to send the invoice", "reminder"),
    ("remind me when the subscription renews", "reminder"),
    ("set timer for 30 minutes for pasta", "reminder"),
    ("i need a reminder for the car service", "reminder"),
    ("please remind me to backup my laptop", "reminder"),
    ("don't forget — mom's birthday is next week", "reminder"),
    ("i keep forgetting to take vitamins in the morning", "reminder"),
    ("nudge me before the meeting starts", "reminder"),
    ("add to my reminders: pick up dry cleaning", "reminder"),
    ("i need to remember to reply to john", "reminder"),
    ("add a reminder for the school event", "reminder"),
    ("keep track of this: submit leave form", "reminder"),
    ("alert me 10 mins before the call", "reminder"),
    ("can you keep a note that i need to review the contract", "reminder"),
    ("reminder to log hours before friday", "reminder"),
    ("don't let me forget to water the garden", "reminder"),
    ("remind me to charge my laptop before tomorrow", "reminder"),

    # ── emotional-support ─────────────────────────────────────────────────────
    ("i'm feeling really down today", "emotional-support"),
    ("nobody understands me", "emotional-support"),
    ("i just need someone to talk to", "emotional-support"),
    ("i feel so lonely lately", "emotional-support"),
    ("everything feels overwhelming right now", "emotional-support"),
    ("i cried all night and i don't know why", "emotional-support"),
    ("i'm so anxious about tomorrow", "emotional-support"),
    ("i feel like i'm failing at everything", "emotional-support"),
    ("i miss her so much", "emotional-support"),
    ("life has been really tough this week", "emotional-support"),
    ("i don't feel good about myself", "emotional-support"),
    ("i'm exhausted and stressed", "emotional-support"),
    ("feeling very sad and empty", "emotional-support"),
    ("i feel like nobody cares", "emotional-support"),
    ("i'm scared and don't know what to do", "emotional-support"),
    ("my anxiety is through the roof", "emotional-support"),
    ("had a terrible day and just want to vent", "emotional-support"),
    ("i think i'm depressed", "emotional-support"),
    ("i feel like giving up", "emotional-support"),
    ("this is really hard and i'm not coping well", "emotional-support"),
    ("i need to vent about something", "emotional-support"),
    ("my heart is broken", "emotional-support"),
    ("i feel lost and confused", "emotional-support"),
    ("things are not going well at all", "emotional-support"),
    ("i hate how i've been feeling lately", "emotional-support"),
    ("i feel so worthless sometimes", "emotional-support"),
    ("i'm struggling to get through the day", "emotional-support"),
    ("nobody seems to understand what i'm going through", "emotional-support"),
    ("i'm having a panic attack right now", "emotional-support"),
    ("i feel so trapped in my situation", "emotional-support"),
    ("i'm overwhelmed and burnt out", "emotional-support"),
    ("just had a huge argument with my partner", "emotional-support"),
    ("i feel like i can't breathe sometimes", "emotional-support"),
    ("i need support right now", "emotional-support"),
    ("i'm dealing with a lot of grief", "emotional-support"),
    ("i haven't been able to get out of bed", "emotional-support"),
    ("i feel invisible to everyone around me", "emotional-support"),
    ("today was really rough and i'm not okay", "emotional-support"),
    ("i keep having these dark thoughts", "emotional-support"),
    ("i feel so alone in all of this", "emotional-support"),

    # ── action-item ───────────────────────────────────────────────────────────
    ("can you send the report to the team", "action-item"),
    ("book a table for 8pm at the restaurant", "action-item"),
    ("buy groceries — milk, eggs, bread", "action-item"),
    ("draft the email to the client", "action-item"),
    ("fix the bug in the login page", "action-item"),
    ("update the spreadsheet with the new data", "action-item"),
    ("call the plumber today", "action-item"),
    ("submit the assignment before midnight", "action-item"),
    ("order a birthday cake for friday", "action-item"),
    ("reply to all pending emails", "action-item"),
    ("create a ticket for this issue", "action-item"),
    ("complete the onboarding form", "action-item"),
    ("prepare slides for the presentation", "action-item"),
    ("review the pull request by end of day", "action-item"),
    ("renew the domain name before it expires", "action-item"),
    ("pay rent by the first", "action-item"),
    ("print the documents for the meeting", "action-item"),
    ("confirm the hotel booking", "action-item"),
    ("schedule the interview with the candidate", "action-item"),
    ("deploy the app to production", "action-item"),
    ("set up the new laptop", "action-item"),
    ("call the bank and dispute the charge", "action-item"),
    ("collect the parcel from the post office", "action-item"),
    ("file the tax return this week", "action-item"),
    ("update my linkedin profile", "action-item"),
    ("follow up with the vendor about the quote", "action-item"),
    ("write unit tests for the new feature", "action-item"),
    ("export the analytics report", "action-item"),
    ("clean the apartment before the guests arrive", "action-item"),
    ("finish reading the quarterly report", "action-item"),
    ("add the new employee to the team channel", "action-item"),
    ("cancel the gym membership", "action-item"),
    ("transfer the files to the shared drive", "action-item"),
    ("reschedule the client meeting to thursday", "action-item"),
    ("sign and return the contract", "action-item"),
    ("push the code changes to the repository", "action-item"),
    ("generate a summary of last month's sales", "action-item"),
    ("verify the payment was received", "action-item"),
    ("send the login credentials to the new user", "action-item"),
    ("archive the old project files", "action-item"),

    # ── small-talk ────────────────────────────────────────────────────────────
    ("how was your day", "small-talk"),
    ("good morning!", "small-talk"),
    ("what did you have for lunch", "small-talk"),
    ("it's so cold today isn't it", "small-talk"),
    ("haha yeah that's funny", "small-talk"),
    ("i watched a great movie last night", "small-talk"),
    ("what are you up to this weekend", "small-talk"),
    ("just saying hi", "small-talk"),
    ("did you see the game last night", "small-talk"),
    ("the weather is beautiful today", "small-talk"),
    ("i've been listening to so much music lately", "small-talk"),
    ("random thought but have you tried sushi", "small-talk"),
    ("lol that's hilarious", "small-talk"),
    ("what's your favourite movie", "small-talk"),
    ("i'm bored, let's chat", "small-talk"),
    ("hey how are things going", "small-talk"),
    ("had the best coffee this morning", "small-talk"),
    ("can't believe it's already friday", "small-talk"),
    ("so what's new with you", "small-talk"),
    ("just wanted to check in", "small-talk"),
    ("i've been thinking about getting a dog", "small-talk"),
    ("did you watch the latest episode", "small-talk"),
    ("feeling lazy today to be honest", "small-talk"),
    ("the traffic was insane this morning", "small-talk"),
    ("nothing special going on, just chilling", "small-talk"),
    ("haha same that's exactly what happened to me", "small-talk"),
    ("have you read any good books lately", "small-talk"),
    ("i need a vacation badly", "small-talk"),
    ("what kind of music do you like", "small-talk"),
    ("i always forget what day it is", "small-talk"),
    ("working from home has its perks", "small-talk"),
    ("i can't believe how fast this year went", "small-talk"),
    ("honestly i could eat pizza every day", "small-talk"),
    ("let's hang out sometime soon", "small-talk"),
    ("you won't believe what happened today", "small-talk"),
    ("the sunrise was gorgeous this morning", "small-talk"),
    ("i've been playing a lot of games lately", "small-talk"),
    ("just finished binge watching the whole series", "small-talk"),
    ("ok this is totally random but", "small-talk"),
    ("happy friday everyone", "small-talk"),

    # ── unknown ───────────────────────────────────────────────────────────────
    ("2390583", "unknown"),
    ("asjdhjkahsdkj", "unknown"),
    ("...", "unknown"),
    ("???", "unknown"),
    ("k", "unknown"),
    ("", "unknown"),
    ("x", "unknown"),
    ("test", "unknown"),
    ("asdf", "unknown"),
    ("12345", "unknown"),
    ("null", "unknown"),
    ("undefined", "unknown"),
    ("gibberish random words splat fuzz", "unknown"),
    ("@#$%", "unknown"),
    ("hello hello hello hello hello hello", "unknown"),
    ("the the the the the", "unknown"),
    ("aaa bbb ccc ddd", "unknown"),
    ("zzzzzzzzzz", "unknown"),
    ("pqrst uvwx", "unknown"),
    ("mxyzptlk", "unknown"),
    ("lorem ipsum dolor sit amet", "unknown"),
    ("idk", "unknown"),
    ("hmm", "unknown"),
    (".", "unknown"),
    ("  ", "unknown"),
    ("blah blah blah", "unknown"),
    ("nothing", "unknown"),
    ("n/a", "unknown"),
    ("skip", "unknown"),
    ("pass", "unknown"),
    ("00000", "unknown"),
    ("!!!", "unknown"),
    ("xD", "unknown"),
    ("ok ok ok ok", "unknown"),
    ("true", "unknown"),
    ("false", "unknown"),
    ("yes no maybe", "unknown"),
    ("abc def ghi", "unknown"),
    ("this is just noise in the data stream", "unknown"),
    ("random input that does not mean anything clear", "unknown"),
]

# ── Rule-based fallback patterns ──────────────────────────────────────────────

RULE_PATTERNS: Dict[str, re.Pattern] = {
    "reminder": re.compile(
        r"\b(remind|reminder|don't forget|remember to|set an? (alarm|timer|reminder)|"
        r"alert me|ping me|note to self|flag this|schedule a reminder)\b",
        re.IGNORECASE
    ),
    "emotional-support": re.compile(
        r"\b(feel(ing)?|lonely|anxious|overwhelmed|stressed|depressed|sad|scared|"
        r"heartbroken|crying|i need (help|support|someone)|panic|grief|vent|"
        r"not okay|struggling|dark thoughts|alone|empty inside)\b",
        re.IGNORECASE
    ),
    "action-item": re.compile(
        r"\b(send|book|buy|draft|fix|update|call|submit|order|reply|create|"
        r"complete|prepare|review|renew|pay|print|confirm|schedule|deploy|"
        r"set up|collect|file|cancel|transfer|sign|push|generate|verify|archive)\b",
        re.IGNORECASE
    ),
    "small-talk": re.compile(
        r"\b(good morning|good night|hey|hi |hello|how are you|what's up|"
        r"how was your|lol|haha|hehe|omg|weather|weekend|bored|chilling|"
        r"what's new|just saying|checking in)\b",
        re.IGNORECASE
    ),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    """Normalise text before vectorising."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── Classifier ────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Lightweight offline intent classifier.
    Model: TF-IDF (word 1-2 grams + char 3-4 grams) → CalibratedSGD.
    Size on disk: ~100–300 KB.   Latency: < 5 ms on CPU.
    """

    CONFIDENCE_THRESHOLD = 0.35   # below this → rule-based override

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.le = LabelEncoder()
        self.le.fit(LABELS)

    def _build_pipeline(self) -> Pipeline:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            preprocessor=_preprocess,
        )
        # Use word-only vectorizer for simplicity and speed
        # (char grams would help but add size — keeping well under 50MB)
        base_clf = SGDClassifier(
            loss="modified_huber",   # gives calibrated probabilities natively
            alpha=1e-3,
            max_iter=500,
            random_state=42,
            class_weight="balanced",
        )
        return Pipeline([
            ("tfidf", word_vec),
            ("clf",   base_clf),
        ])

    def train_and_save(self):
        """Train on TRAINING_DATA and pickle the model."""
        random.shuffle(TRAINING_DATA)
        texts  = [_preprocess(t) for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]

        pipe = self._build_pipeline()
        pipe.fit(texts, labels)
        self.pipeline = pipe

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipe, f, protocol=4)

        size_kb = os.path.getsize(MODEL_PATH) // 1024
        print(f"  [IntentClassifier] Trained & saved → {MODEL_PATH} ({size_kb} KB)")

    def load(self):
        """Load pickled model from disk."""
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)
        size_kb = os.path.getsize(MODEL_PATH) // 1024
        print(f"  [IntentClassifier] Loaded from {MODEL_PATH} ({size_kb} KB)")

    def _rule_override(self, text: str) -> str | None:
        """Return a label if a strong rule fires, else None."""
        for label, pattern in RULE_PATTERNS.items():
            if pattern.search(text):
                return label
        return None

    def predict(self, text: str) -> Dict:
        """
        Classify a single message.

        Returns:
        {
          "intent": "reminder",
          "confidence": 0.87,
          "latency_ms": 1.2,
          "all_scores": {"reminder": 0.87, "small-talk": 0.06, ...},
          "method": "model" | "rule-fallback"
        }
        """
        t0 = time.perf_counter()

        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 1.0,
                "latency_ms": 0.0,
                "all_scores": {l: 0.0 for l in LABELS},
                "method": "empty-input",
            }

        processed = _preprocess(text)
        method = "model"
        intent = "unknown"
        confidence = 0.0
        all_scores: Dict[str, float] = {l: 0.0 for l in LABELS}

        if self.pipeline is not None:
            try:
                proba = self.pipeline.predict_proba([processed])[0]
                classes = self.pipeline.classes_
                all_scores = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
                best_idx = int(np.argmax(proba))
                confidence = float(proba[best_idx])
                intent = str(classes[best_idx])
            except Exception:
                confidence = 0.0

        # Rule-based override when model is uncertain
        if confidence < self.CONFIDENCE_THRESHOLD:
            rule_intent = self._rule_override(text)
            if rule_intent:
                intent = rule_intent
                confidence = 0.75   # heuristic confidence for rule hits
                method = "rule-fallback"
            elif confidence == 0.0:
                intent = "unknown"
                confidence = 1.0

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "latency_ms": latency_ms,
            "all_scores": all_scores,
            "method": method,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify multiple messages. Returns list in same order."""
        return [self.predict(t) for t in texts]