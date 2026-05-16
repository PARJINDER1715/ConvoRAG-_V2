import re
from dataclasses import dataclass, field

HIGH_EMOTION = {
    "positive": ["love","amazing","happy","excited","thrilled","wonderful",
                 "grateful","proud","blessed","fantastic","great news"],
    "negative": ["hate","hurt","cry","loss","grief","fight","broke","angry",
                 "frustrated","disappointed","upset","alone","scared","miss"],
    "neutral":  ["mentioned","said","talked","discussed","told","asked"],
}

_FALLBACK_CONTRADICTION_PAIRS = [
    ({"love","like","enjoy","happy","good"},    {"hate","dislike","upset","bad","angry"}),
    ({"close","best friend","love"},            {"fight","broke up","argument","distant"}),
    ({"well","fine","okay","healthy"},          {"sick","ill","hospital","hurt","pain"}),
    ({"together","married","dating"},           {"divorced","separated","broke up","single"}),
    ({"alive","healthy"},                       {"died","passed away","gone","lost"}),
]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    topic: str
    start_msg: int
    end_msg: int
    day: int
    relevance_score: float
    recency_score: float
    emotional_weight: float
    emotion_valence: str
    final_score: float = 0.0
    contradiction_flags: list = field(default_factory=list)


@dataclass
class ResolverResult:
    query: str
    answer: str
    ranked_chunks: list[RetrievedChunk]
    contradictions_found: bool
    contradiction_details: list[str]
    merge_strategy: str
    confidence: float


class ConflictResolver:
    W_RELEVANCE = 0.35
    W_RECENCY   = 0.40
    W_EMOTION   = 0.25

    def __init__(self, total_messages: int = 2000):
        self.total_messages = total_messages

    def _get_contradiction_pairs(self):
        try:
            from core.db import get_contradiction_pairs
            pairs = get_contradiction_pairs()
            return pairs if pairs else _FALLBACK_CONTRADICTION_PAIRS
        except Exception:
            return _FALLBACK_CONTRADICTION_PAIRS

    def _keyword_relevance(self, query, text):
        q_words = set(re.findall(r'\w+', query.lower()))
        t_words = set(re.findall(r'\w+', text.lower()))
        if not q_words:
            return 0.0
        base = len(q_words & t_words) / len(q_words | t_words)
        for word in q_words:
            if len(word) > 3 and word in text.lower():
                base += 0.1
        return min(1.0, base)

    def _emotional_weight(self, text):
        tl = text.lower()
        pos = sum(1 for w in HIGH_EMOTION["positive"] if w in tl)
        neg = sum(1 for w in HIGH_EMOTION["negative"] if w in tl)
        neu = sum(1 for w in HIGH_EMOTION["neutral"]  if w in tl)
        total = pos + neg + neu
        if total == 0:
            return 0.1, "neutral"
        weight = min(1.0, (pos + neg * 1.5) / (total + 1))
        valence = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
        return round(weight, 3), valence

    def _recency_score(self, end_msg):
        return min(1.0, end_msg / self.total_messages)

    def _detect_contradictions(self, chunks):
        contradiction_pairs = self._get_contradiction_pairs()
        contradictions = []
        texts = [(c.chunk_id, c.text.lower()) for c in chunks]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                id_a, text_a = texts[i]
                id_b, text_b = texts[j]
                for pos_set, neg_set in contradiction_pairs:
                    has_pos_a = any(w in text_a for w in pos_set)
                    has_neg_a = any(w in text_a for w in neg_set)
                    has_pos_b = any(w in text_b for w in pos_set)
                    has_neg_b = any(w in text_b for w in neg_set)
                    if (has_pos_a and has_neg_b) or (has_neg_a and has_pos_b):
                        pos_word = next((w for w in pos_set if w in text_a or w in text_b), "positive context")
                        neg_word = next((w for w in neg_set if w in text_a or w in text_b), "negative context")
                        contradictions.append(
                            f"Chunk '{id_a}' and '{id_b}' contradict: "
                            f"one mentions '{pos_word}', other mentions '{neg_word}'"
                        )
        valences = [c.emotion_valence for c in chunks]
        if "positive" in valences and "negative" in valences:
            contradictions.append(
                "Emotional valence conflict: some chunks show positive context, "
                "others show negative context for the same subject."
            )
        return list(set(contradictions))

    def _build_answer(self, query, ranked, contradictions):
        if not ranked:
            return "No relevant mentions found in the conversation history.", "none"
        top = ranked[0]
        strategy = "emotion_wins" if top.emotional_weight > 0.5 else "recency_wins"
        lines = [f"Based on the conversation history (searching: '{query}'):\n"]
        snippet = top.text[:300].replace('\n', ' ')
        lines.append(
            f"📍 Most relevant mention [Day ~{top.day}, msgs {top.start_msg}-{top.end_msg}]:\n"
            f'   "{snippet}…"\n'
            f"   Topic: {top.topic} | Sentiment: {top.emotion_valence}"
        )
        if len(ranked) > 1:
            strategy = "merged"
            lines.append(f"\n📚 Also mentioned in {len(ranked)-1} other context(s):")
            for chunk in ranked[1:3]:
                snip = chunk.text[:150].replace('\n', ' ')
                lines.append(
                    f"  • [Day ~{chunk.day}, msgs {chunk.start_msg}-{chunk.end_msg}] "
                    f"{chunk.topic}: \"{snip}…\""
                )
        if contradictions:
            lines.append(f"\n⚠️  Contradiction detected across {len(ranked)} mentions:")
            for c in contradictions[:2]:
                lines.append(f"   • {c}")
            lines.append("   → Showing most recent mention as primary answer.")
        return "\n".join(lines), strategy

    def resolve(self, query, raw_chunks):
        if not raw_chunks:
            return ResolverResult(
                query=query, answer="No relevant chunks found.",
                ranked_chunks=[], contradictions_found=False,
                contradiction_details=[], merge_strategy="none", confidence=0.0
            )
        scored = []
        for rc in raw_chunks:
            rel        = self._keyword_relevance(query, rc.get("text", ""))
            rec        = self._recency_score(rc.get("end_msg", 0))
            em_w, em_v = self._emotional_weight(rc.get("text", ""))
            final      = self.W_RELEVANCE*rel + self.W_RECENCY*rec + self.W_EMOTION*em_w
            scored.append(RetrievedChunk(
                chunk_id=str(rc.get("chunk_id", rc.get("id", "?"))),
                text=rc.get("text", ""),
                topic=rc.get("topic", "Unknown"),
                start_msg=rc.get("start_msg", rc.get("start", 0)),
                end_msg=rc.get("end_msg", rc.get("end", 0)),
                day=rc.get("day", rc.get("day_start", 0)),
                relevance_score=round(rel, 3),
                recency_score=round(rec, 3),
                emotional_weight=em_w,
                emotion_valence=em_v,
                final_score=round(final, 3),
            ))
        ranked = sorted(scored, key=lambda x: x.final_score, reverse=True)
        contradictions = self._detect_contradictions(ranked)
        for chunk in ranked:
            chunk.contradiction_flags = [c for c in contradictions if chunk.chunk_id in c]
        answer, strategy = self._build_answer(query, ranked, contradictions)
        confidence = ranked[0].final_score if ranked else 0.0
        return ResolverResult(
            query=query, answer=answer, ranked_chunks=ranked,
            contradictions_found=len(contradictions) > 0,
            contradiction_details=contradictions,
            merge_strategy=strategy, confidence=round(confidence, 3),
        )
