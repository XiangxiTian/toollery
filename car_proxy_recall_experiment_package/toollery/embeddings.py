from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

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

    def encode_many(self, texts: list[str], progress_callback: Any | None = None) -> list[list[float]]:
        return [self.encode(text) for text in texts]


class OpenAICompatibleEmbedder:
    """Embedding adapter for OpenAI-compatible `/embeddings` endpoints."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
        timeout: int = 120,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.dimensions = dimensions or _int_env("EMBEDDING_DIMENSIONS")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", batch_size))
        self.timeout = int(os.getenv("EMBEDDING_TIMEOUT", timeout))
        self.max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", max_retries))
        self.extra_body = extra_body if extra_body is not None else _json_env("EMBEDDING_EXTRA_BODY")

    def fit(self, texts: list[str]) -> "OpenAICompatibleEmbedder":
        return self

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str], progress_callback: Any | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
            if progress_callback:
                progress_callback(min(start + self.batch_size, len(texts)), len(texts))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY or OPENAI_API_KEY is required for embedding retrieval")
        if self.api_key.startswith("PASTE_") or self.api_key.endswith("_HERE"):
            raise RuntimeError("embedding api_key still looks like a placeholder; put your real embedding API key in config.")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        payload.update(self.extra_body)
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = self._request_with_retries(request)
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        vectors = [list(map(float, item["embedding"])) for item in items]
        return [_normalize(vector) for vector in vectors]

    def _request_with_retries(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(
                        f"Embedding request failed with HTTP {exc.code} {exc.reason}. "
                        f"model={self.model!r} base_url={self.base_url!r}. Response: {detail}"
                    ) from exc
                last_error = RuntimeError(
                    f"Embedding request failed with HTTP {exc.code} {exc.reason}. "
                    f"model={self.model!r} base_url={self.base_url!r}. Response: {detail}"
                )
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Embedding request failed after {self.max_retries + 1} attempts. "
            f"model={self.model!r} base_url={self.base_url!r}. Last error: {last_error}"
        ) from last_error


class LocalHFEmbedder:
    """Local HuggingFace embedding adapter.

    This backend is designed for locally downloaded embedding models such as
    Qwen/Qwen3-Embedding-4B. It first tries SentenceTransformers when available
    and falls back to plain Transformers with configurable pooling.
    """

    def __init__(
        self,
        model: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        pooling: str = "last",
        normalize: bool = True,
        dtype: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        text_prefix: str = "",
        implementation: str = "auto",
    ) -> None:
        self.model = model_path or model or os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_MODEL")
        if not self.model:
            raise RuntimeError("EMBEDDING_MODEL_PATH or EMBEDDING_MODEL is required for local-hf embeddings")
        self.device = device or os.getenv("EMBEDDING_DEVICE")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", batch_size))
        self.max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", max_length))
        self.pooling = os.getenv("EMBEDDING_POOLING", pooling).lower()
        self.normalize = _bool_env("EMBEDDING_NORMALIZE", normalize)
        self.dtype = dtype or os.getenv("EMBEDDING_DTYPE")
        self.trust_remote_code = _bool_env("EMBEDDING_TRUST_REMOTE_CODE", trust_remote_code)
        self.local_files_only = _bool_env("EMBEDDING_LOCAL_FILES_ONLY", local_files_only)
        self.text_prefix = os.getenv("EMBEDDING_TEXT_PREFIX", text_prefix)
        self.implementation = os.getenv("EMBEDDING_IMPLEMENTATION", implementation).lower()
        self._sentence_model: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def fit(self, texts: list[str]) -> "LocalHFEmbedder":
        return self

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str], progress_callback: Any | None = None) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            batch = texts[start : start + self.batch_size]
            if self.text_prefix:
                batch = [self.text_prefix + text for text in batch]
            if self._sentence_model is not None:
                vectors.extend(self._encode_sentence_transformers(batch))
            else:
                vectors.extend(self._encode_transformers(batch))
            if progress_callback:
                progress_callback(min(start + self.batch_size, len(texts)), len(texts))
        return vectors

    def _ensure_loaded(self) -> None:
        if self._sentence_model is not None or self._model is not None:
            return
        if self.implementation in {"auto", "sentence-transformers", "sentence_transformers", "st"}:
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    self._sentence_model = SentenceTransformer(
                        self.model,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                        local_files_only=self.local_files_only,
                    )
                except TypeError:
                    self._sentence_model = SentenceTransformer(self.model, device=self.device)
                return
            except ImportError:
                if self.implementation not in {"auto"}:
                    raise RuntimeError(
                        "sentence-transformers is required for EMBEDDING_IMPLEMENTATION=sentence-transformers"
                    )

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local HuggingFace embeddings require `transformers` and `torch`, "
                "or `sentence-transformers`. Install them and point EMBEDDING_MODEL_PATH "
                "to your downloaded model directory."
            ) from exc

        self._torch = torch
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        torch_dtype = _torch_dtype(torch, self.dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(self.model, **model_kwargs)
        if self.device:
            self._model.to(self.device)
        self._model.eval()

    def _encode_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        encoded = self._sentence_model.encode(
            texts,
            batch_size=max(1, self.batch_size),
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in encoded.tolist()]

    def _encode_transformers(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            pooled = _pool_embeddings(torch, token_embeddings, attention_mask, self.pooling)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [list(map(float, vector)) for vector in pooled.detach().cpu().tolist()]


def make_embedder_from_env() -> HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder:
    backend = os.getenv("EMBEDDING_BACKEND", "tfidf").lower()
    if backend in {"openai", "openai-compatible", "api"}:
        return OpenAICompatibleEmbedder()
    if backend in {"local", "local-hf", "hf", "huggingface", "transformers", "sentence-transformers"}:
        return LocalHFEmbedder()
    return HashingTfidfEmbedder()


def embedder_signature(embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder) -> dict[str, Any]:
    if isinstance(embedder, OpenAICompatibleEmbedder):
        return {
            "backend": "openai-compatible",
            "base_url": embedder.base_url,
            "model": embedder.model,
            "dimensions": embedder.dimensions,
            "extra_body": embedder.extra_body,
        }
    if isinstance(embedder, LocalHFEmbedder):
        return {
            "backend": "local-hf",
            "model": embedder.model,
            "max_length": embedder.max_length,
            "pooling": embedder.pooling,
            "normalize": embedder.normalize,
            "dtype": embedder.dtype,
            "text_prefix": embedder.text_prefix,
            "implementation": embedder.implementation,
        }
    return {"backend": "hashing-tfidf", "dimensions": embedder.dimensions}


def text_digest(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    return int(value)


def _json_env(name: str) -> dict[str, Any]:
    value = os.getenv(name)
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} must decode to a JSON object")
    return data


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _torch_dtype(torch: Any, dtype: str | None) -> Any | None:
    if not dtype:
        return None
    normalized = dtype.lower()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise RuntimeError(f"Unsupported EMBEDDING_DTYPE={dtype!r}; use float32, float16, bfloat16, or auto.")
    return mapping[normalized]


def _pool_embeddings(torch: Any, token_embeddings: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling in {"mean", "avg", "average"}:
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom
    if pooling == "cls":
        return token_embeddings[:, 0]
    if pooling in {"last", "eos"}:
        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(token_embeddings.shape[0], device=token_embeddings.device)
        return token_embeddings[batch_indices, lengths]
    raise RuntimeError(f"Unsupported EMBEDDING_POOLING={pooling!r}; use last, mean, or cls.")
