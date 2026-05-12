from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import blake2b
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "by",
    "for",
    "from",
    "get",
    "i",
    "in",
    "is",
    "me",
    "need",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "tool",
    "use",
    "with",
}

SYNONYMS = {
    "umbrella": ["rain", "weather", "forecast"],
    "rain": ["weather", "forecast"],
    "forecast": ["weather"],
    "hotel": ["stay", "lodging"],
    "hotels": ["stay", "lodging"],
    "stay": ["hotel", "lodging"],
    "flight": ["travel", "origin", "destination"],
    "fly": ["flight", "travel"],
    "meeting": ["calendar", "event", "attendees"],
    "meet": ["calendar", "event"],
    "calendar": ["meeting", "event"],
    "email": ["message", "recipient"],
    "note": ["email", "message"],
    "send": ["email", "message"],
    "usd": ["currency", "money", "amount"],
    "cny": ["currency", "money", "amount"],
    "eur": ["currency", "money", "amount"],
    "currency": ["money", "convert", "amount"],
    "change": ["convert", "currency"],
}


def normalize_text(text: str) -> str:
    spaced = CAMEL_RE.sub(" ", text.replace("_", " ").replace("-", " "))
    return spaced.lower()


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(text))
    cleaned = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    expanded: list[str] = []
    for token in cleaned:
        expanded.append(token)
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def term_overlap(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / math.sqrt(len(left_terms) * len(right_terms))


def top_terms(texts: Iterable[str], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return [term for term, _ in counts.most_common(limit)]


def stable_hash(token: str, buckets: int) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets
