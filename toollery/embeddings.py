from __future__ import annotations

import math
from collections import Counter

from .text import stable_hash, tokenize


class HashingTfidfEmbedder:
    """Small dependency-free semantic indexer for proxy-query matching.

    It is not a replacement for production embedding models, but it keeps this
    reference implementation runnable without external services.
    """

    def __init__(self, dimensions: int = 2048) -> None:
        self.dimensions = dimensions
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        doc_count = max(len(texts), 1)
        dfs: Counter[str] = Counter()
        for text in texts:
            dfs.update(set(tokenize(text)))
        self._idf = {
            token: math.log((doc_count + 1) / (df + 1)) + 1.0 for token, df in dfs.items()
        }
        return self

    def encode(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            idx = stable_hash(token, self.dimensions)
            sign = 1.0 if stable_hash("sign:" + token, 2) == 0 else -1.0
            vector[idx] += sign * (1.0 + math.log(count)) * self._idf.get(token, 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
