import re
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Tone / Mood lexicons (local, no API) ─────────────────────────────────────

TONE_LEXICON = {
    "formal":    ["therefore","however","furthermore","regarding","consequently",
                  "nevertheless","indeed","obtain","require","provide","ensure","thus"],
    "casual":    ["yeah","yep","lol","haha","gonna","wanna","kinda","sorta","nope",
                  "cool","awesome","omg","tbh","btw","idk","imo","nah"],
    "playful":   ["haha","lol","😄","😂","🎉","fun","joke","play","silly","hilarious",
                  "laugh","amazing","wow","whoa","yay","hehe"],
    "frustrated":["frustrated","annoying","ugh","terrible","horrible","hate","worst",
                  "awful","upset","angry","disappointed","ridiculous","pointless","done"],
    "curious":   ["why","how","what","wonder","curious","interesting","tell me","really",
                  "never knew","learn","discover","question","ask","explain"],
    "anxious":   ["worried","nervous","scared","afraid","anxious","stress","concern",
                  "hope","what if","maybe","unsure","don't know","overwhelmed"],
    "excited":   ["excited","thrilled","can't wait","amazing","love","incredible",
                  "fantastic","great news","so happy","finally","yes!"],
    "sad":       ["sad","miss","lonely","cry","tears","hurt","pain","lost","grief",
                  "unfortunate","depressed","heartbroken","sigh"],
}

EMOTION_LEXICON = {
    "positive": ["good","great","happy","love","enjoy","wonderful","excellent",
                 "nice","pleased","glad","thankful","grateful","blessed"],
    "negative": ["bad","sad","awful","terrible","hate","dislike","upset",
                 "angry","annoyed","frustrated","disappointed","regret"],
    "neutral":  ["okay","fine","alright","sure","maybe","perhaps","think","feel"],
}

# Trigger pattern detectors
PERSON_PATTERNS  = re.compile(r'\b(my\s+(?:sister|brother|mom|dad|mother|father|friend|boyfriend|girlfriend|husband|wife|son|daughter|colleague|boss|teacher|professor))\b', re.I)
TOPIC_PATTERNS   = re.compile(r'\b(work|job|school|college|exam|relationship|health|money|family|travel|moving|career|project|interview)\b', re.I)
EVENT_PATTERNS   = re.compile(r'\b(got|lost|found|started|ended|broke|moved|graduated|failed|passed|hired|fired|met|visited|broke up|got married)\b', re.I)


@dataclass
class DayProfile:
    day: int
    dominant_tone: str
    secondary_tone: str
    mood: str           # positive / negative / neutral
    energy: float       # 0-1 (message length + exclamation usage)
    formality: float    # 0-1
    tone_scores: dict   = field(default_factory=dict)
    triggers: list      = field(default_factory=list)
    key_topics: list    = field(default_factory=list)
    people_mentioned: list = field(default_factory=list)
    message_count: int  = 0
    avg_msg_length: float = 0.0
    sample_message: str = ""


@dataclass
class DriftEvent:
    from_day: int
    to_day: int
    from_tone: str
    to_tone: str
    drift_magnitude: float   # 0-1
    trigger_type: str        # topic | person | event | emotional
    trigger_text: str
    description: str


class PersonaDriftDetector:
    """
    Analyzes conversation messages day-by-day.
    Detects tone/mood shifts and the triggers that caused them.
    """

    def __init__(self):
        self.day_profiles: list[DayProfile] = []
        self.drift_events: list[DriftEvent] = []
        self.timeline: list[dict] = []

    # ── Score a single message ────────────────────────────────────────────────

    def _score_message(self, text: str) -> dict:
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        word_set = set(words)

        tone_scores = {}
        for tone, keywords in TONE_LEXICON.items():
            hits = sum(1 for k in keywords if k in text_lower)
            tone_scores[tone] = hits

        mood_scores = {}
        for mood, keywords in EMOTION_LEXICON.items():
            hits = sum(1 for k in keywords if k in text_lower)
            mood_scores[mood] = hits

        exclamations = text.count('!') + text.count('?')
        energy = min(1.0, (len(text) / 200 + exclamations * 0.1))

        formal_hits  = sum(1 for k in TONE_LEXICON["formal"]  if k in text_lower)
        casual_hits  = sum(1 for k in TONE_LEXICON["casual"]  if k in text_lower)
        total = formal_hits + casual_hits + 1
        formality = formal_hits / total

        return {
            "tone_scores": tone_scores,
            "mood_scores": mood_scores,
            "energy": energy,
            "formality": formality,
            "length": len(text),
        }

    # ── Extract triggers from text ────────────────────────────────────────────

    def _extract_triggers(self, texts: list[str]) -> tuple[list, list, list]:
        combined = " ".join(texts)
        people  = list(set(m.group(0) for m in PERSON_PATTERNS.finditer(combined)))
        topics  = list(set(m.group(0).lower() for m in TOPIC_PATTERNS.finditer(combined)))
        events  = list(set(m.group(0).lower() for m in EVENT_PATTERNS.finditer(combined)))
        return people[:5], topics[:5], events[:3]

    # ── Build profile for one day ─────────────────────────────────────────────

    def build_day_profile(self, day: int, messages: list[dict]) -> DayProfile:
        u1_msgs = [m["text"] for m in messages if m["spk"] == "U1"]
        if not u1_msgs:
            u1_msgs = [m["text"] for m in messages]

        agg_tone   = defaultdict(float)
        agg_mood   = defaultdict(float)
        agg_energy = []
        agg_form   = []

        for text in u1_msgs:
            s = self._score_message(text)
            for t, v in s["tone_scores"].items():
                agg_tone[t] += v
            for m, v in s["mood_scores"].items():
                agg_mood[m] += v
            agg_energy.append(s["energy"])
            agg_form.append(s["formality"])

        # Normalize
        total_tone = sum(agg_tone.values()) or 1
        tone_scores = {k: round(v / total_tone, 3) for k, v in agg_tone.items()}

        sorted_tones = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)
        dominant   = sorted_tones[0][0] if sorted_tones else "neutral"
        secondary  = sorted_tones[1][0] if len(sorted_tones) > 1 else "neutral"

        total_mood = sum(agg_mood.values()) or 1
        mood_scores = {k: v / total_mood for k, v in agg_mood.items()}
        mood = max(mood_scores, key=mood_scores.get) if mood_scores else "neutral"

        people, topics, events = self._extract_triggers(u1_msgs)

        triggers = []
        if events:
            triggers.extend([f"event: {e}" for e in events])
        if people:
            triggers.extend([f"person: {p}" for p in people])
        if topics:
            triggers.extend([f"topic: {t}" for t in topics])

        return DayProfile(
            day=day,
            dominant_tone=dominant,
            secondary_tone=secondary,
            mood=mood,
            energy=round(sum(agg_energy) / len(agg_energy), 3),
            formality=round(sum(agg_form) / len(agg_form), 3),
            tone_scores=tone_scores,
            triggers=triggers[:5],
            key_topics=topics,
            people_mentioned=people,
            message_count=len(u1_msgs),
            avg_msg_length=round(sum(len(t) for t in u1_msgs) / len(u1_msgs), 1),
            sample_message=u1_msgs[0][:120] if u1_msgs else "",
        )

    # ── Detect drift between consecutive days ─────────────────────────────────

    def _detect_drift(self, prev: DayProfile, curr: DayProfile) -> Optional[DriftEvent]:
        if prev.dominant_tone == curr.dominant_tone:
            return None

        # Measure magnitude: difference in tone score distributions
        all_tones = set(list(prev.tone_scores.keys()) + list(curr.tone_scores.keys()))
        magnitude = 0.0
        for t in all_tones:
            a = prev.tone_scores.get(t, 0)
            b = curr.tone_scores.get(t, 0)
            magnitude += abs(a - b)
        magnitude = min(1.0, magnitude / 2)

        if magnitude < 0.05:
            return None

        # Identify trigger type
        if curr.people_mentioned:
            ttype = "person"
            ttext = curr.people_mentioned[0]
        elif curr.key_topics:
            ttype = "topic"
            ttext = curr.key_topics[0]
        elif curr.mood != prev.mood:
            ttype = "emotional"
            ttext = f"mood shifted from {prev.mood} to {curr.mood}"
        else:
            ttype = "unknown"
            ttext = "no clear trigger"

        desc = (f"Tone shifted from '{prev.dominant_tone}' to '{curr.dominant_tone}' "
                f"(magnitude: {magnitude:.2f}). Likely trigger: {ttype} — {ttext}.")

        return DriftEvent(
            from_day=prev.day,
            to_day=curr.day,
            from_tone=prev.dominant_tone,
            to_tone=curr.dominant_tone,
            drift_magnitude=round(magnitude, 3),
            trigger_type=ttype,
            trigger_text=ttext,
            description=desc,
        )

    # ── Main: process all days ────────────────────────────────────────────────

    def analyze(self, days_data: list[dict]) -> dict:
        """
        Process list of {day: int, messages: [{spk, text}]}
        Returns full analysis dict.
        """
        self.day_profiles = []
        self.drift_events = []

        for entry in days_data:
            profile = self.build_day_profile(entry["day"], entry["messages"])
            self.day_profiles.append(profile)

        # Detect drifts between consecutive days
        for i in range(1, len(self.day_profiles)):
            drift = self._detect_drift(self.day_profiles[i-1], self.day_profiles[i])
            if drift:
                self.drift_events.append(drift)

        # Build timeline
        self.timeline = []
        for p in self.day_profiles:
            label = f"{p.dominant_tone}"
            if p.secondary_tone and p.secondary_tone != p.dominant_tone:
                label += f" & {p.secondary_tone}"

            self.timeline.append({
                "day": p.day,
                "label": label,
                "mood": p.mood,
                "energy": p.energy,
                "formality": p.formality,
                "triggers": p.triggers,
                "people": p.people_mentioned,
                "topics": p.key_topics,
                "msg_count": p.message_count,
                "sample": p.sample_message,
            })

        return {
            "timeline": self.timeline,
            "drift_events": [asdict(d) for d in self.drift_events],
            "day_profiles": [asdict(p) for p in self.day_profiles],
            "summary": {
                "total_days": len(self.day_profiles),
                "total_drifts": len(self.drift_events),
                "dominant_overall_tone": Counter(
                    p.dominant_tone for p in self.day_profiles
                ).most_common(1)[0][0] if self.day_profiles else "unknown",
                "most_common_trigger": Counter(
                    d.trigger_type for d in self.drift_events
                ).most_common(1)[0][0] if self.drift_events else "none",
            }
        }
