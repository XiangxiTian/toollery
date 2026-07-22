from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.embeddings import OpenAICompatibleEmbedder, cosine, text_digest  # noqa: E402
from toollery.text import tokenize  # noqa: E402


DEFAULT_SAMPLES = "outputs/experiments_v2/car/car_multintent_samples.jsonl"
DEFAULT_PROXY_QUERIES = "outputs/car_tools/car_tools_proxy_queries.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/experiments_v2/car_proxy_recall"
DEFAULT_OUT = f"{DEFAULT_OUTPUT_DIR}/predictions.jsonl"
DEFAULT_SUMMARY_OUT = f"{DEFAULT_OUTPUT_DIR}/summary.json"
DEFAULT_EMBEDDING_CACHE = f"{DEFAULT_OUTPUT_DIR}/proxy_query_embeddings.jsonl"


@dataclass(frozen=True)
class CarRecallSample:
    sample_id: str
    query: str
    correct_tools: list[str]
    original_query: str | None = None


class BM25ProxyRetriever:
    def __init__(self, proxy_queries_by_tool: dict[str, list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.proxy_queries_by_tool = proxy_queries_by_tool
        self.k1 = k1
        self.b = b
        self.units: list[dict[str, Any]] = []
        for tool_name, queries in proxy_queries_by_tool.items():
            for query in queries:
                text = str(query).strip()
                if text:
                    self.units.append({"tool_name": tool_name, "query": text})
        if not self.units:
            raise ValueError("No usable proxy queries were loaded")

        self.tokenized = [tokenize(unit["query"]) for unit in self.units]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        dfs: Counter[str] = Counter()
        for tokens in self.tokenized:
            dfs.update(set(tokens))
        doc_count = max(len(self.units), 1)
        self.idf = {
            term: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def retrieve(self, query: str, top_k: int, proxy_top_k: int | None = None) -> list[dict[str, Any]]:
        limit = proxy_top_k or len(self.units)
        scored_units: list[tuple[float, int]] = []
        query_terms = tokenize(query)
        for index, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            score = 0.0
            dl = self.doc_lengths[index]
            for term in query_terms:
                if term not in tf:
                    continue
                denom = tf[term] + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * tf[term] * (self.k1 + 1.0) / denom
            if score > 0.0:
                scored_units.append((score, index))

        ranked_units = heapq.nlargest(limit, scored_units, key=lambda item: item[0])
        grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for score, index in ranked_units:
            unit = self.units[index]
            grouped[str(unit["tool_name"])].append((score, str(unit["query"])))

        candidates: list[dict[str, Any]] = []
        for tool_name, hits in grouped.items():
            ranked = sorted(hits, key=lambda item: item[0], reverse=True)
            top_scores = [score for score, _ in ranked[:3]]
            score = sum(top_scores) / max(len(top_scores), 1)
            score += 0.05 * min(len(ranked), 5)
            candidates.append(
                {
                    "tool_name": tool_name,
                    "score": score,
                    "supporting_queries": [text for _, text in ranked[:3]],
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["tool_name"])))

        seen = {str(item["tool_name"]) for item in candidates}
        for tool_name, queries in self.proxy_queries_by_tool.items():
            if len(candidates) >= top_k:
                break
            if tool_name in seen:
                continue
            candidates.append(
                {
                    "tool_name": tool_name,
                    "score": 0.0,
                    "supporting_queries": [query for query in queries[:3] if query],
                }
            )
            seen.add(tool_name)
        return candidates[:top_k]


class EmbeddingProxyRetriever:
    def __init__(
        self,
        proxy_queries_by_tool: dict[str, list[str]],
        *,
        embedder: Any,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
    ) -> None:
        self.proxy_queries_by_tool = proxy_queries_by_tool
        self.embedder = embedder
        self.units: list[dict[str, str]] = []
        for tool_name, queries in proxy_queries_by_tool.items():
            for query in queries:
                text = str(query).strip()
                if text:
                    self.units.append({"tool_name": tool_name, "query": text})
        if not self.units:
            raise ValueError("No usable proxy queries were loaded")

        self.cache_path = Path(cache_path) if cache_path else None
        queries = [unit["query"] for unit in self.units]
        self.signature = {
            "version": 1,
            "embedder": _embedder_signature(embedder),
            "proxy_digest": text_digest([f"{unit['tool_name']}\0{unit['query']}" for unit in self.units]),
            "count": len(self.units),
        }
        cached_vectors = None if force_rebuild else _load_embedding_cache(self.cache_path, self.signature, len(self.units))
        if cached_vectors is not None:
            self.vectors = cached_vectors
        else:
            fit = getattr(embedder, "fit", None)
            if fit is not None:
                fit(queries)
            self.vectors = embedder.encode_many(queries)
            _write_embedding_cache(self.cache_path, self.signature, self.vectors)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_vector = self.embedder.encode(query)
        grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for unit, vector in zip(self.units, self.vectors):
            score = cosine(query_vector, vector)
            if score > 0.0:
                grouped[unit["tool_name"]].append((score, unit["query"]))

        candidates: list[dict[str, Any]] = []
        for tool_name, hits in grouped.items():
            ranked = sorted(hits, key=lambda item: item[0], reverse=True)
            top_scores = [score for score, _ in ranked[:3]]
            score = sum(top_scores) / max(len(top_scores), 1)
            score += 0.05 * min(len(ranked), 5)
            candidates.append(
                {
                    "tool_name": tool_name,
                    "score": score,
                    "supporting_queries": [text for _, text in ranked[:3]],
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["tool_name"])))

        seen = {str(item["tool_name"]) for item in candidates}
        for tool_name, queries in self.proxy_queries_by_tool.items():
            if len(candidates) >= top_k:
                break
            if tool_name in seen:
                continue
            candidates.append(
                {
                    "tool_name": tool_name,
                    "score": 0.0,
                    "supporting_queries": [query for query in queries[:3] if query],
                }
            )
            seen.add(tool_name)
        return candidates[:top_k]


def main() -> None:
    args = parse_args()
    apply_config(args)
    samples = load_samples(args.samples, limit=args.limit_samples)
    proxy_queries_by_tool = load_proxy_queries(args.proxy_queries)
    embedder = make_embedder(args.embedding_config) if _uses_embedding_method(args.methods) else None
    rows, summary = run_recall(
        samples=samples,
        proxy_queries_by_tool=proxy_queries_by_tool,
        methods=args.methods,
        top_k=args.top_k,
        bm25_proxy_top_k=args.bm25_proxy_top_k,
        embedder=embedder,
        embedding_cache=args.embedding_cache,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
    )
    write_jsonl(args.out, rows)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k recall over car proxy queries with BM25 and optional embedding-model retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON config file. Values under car_proxy_recall are used as defaults.")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES, help="JSONL samples with sample_id, query, correct_tools.")
    parser.add_argument("--proxy-queries", default=DEFAULT_PROXY_QUERIES, help="JSONL proxy queries with tool_name and query.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL predictions.")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT, help="Output JSON summary.")
    parser.add_argument("--methods", default="bm25", help="Comma-separated methods: bm25,llm. The llm method uses embedding API retrieval.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--bm25-proxy-top-k", type=int, default=200)
    parser.add_argument("--embedding-cache", default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--force-rebuild-embeddings", action="store_true", default=False)
    args = parser.parse_args(argv)
    args._cli_overrides = _cli_overrides(argv if argv is not None else sys.argv[1:])
    args.methods = _parse_methods(args.methods)
    args.embedding_config = {}
    return args


def apply_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.embedding_config = _embedding_config(config)
    values = config.get("car_proxy_recall", {})
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr) or attr in args._cli_overrides:
            continue
        if attr == "methods":
            value = _parse_methods(value)
        setattr(args, attr, value)


def load_samples(path: str | Path, limit: int | None = None) -> list[CarRecallSample]:
    samples: list[CarRecallSample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            query = str(item.get("query", "")).strip()
            if not query:
                continue
            sample_id = str(item.get("sample_id") or item.get("id") or f"sample_{idx:05d}")
            correct_tools = [str(name) for name in item.get("correct_tools", []) if str(name).strip()]
            original_query = item.get("original_query")
            samples.append(
                CarRecallSample(
                    sample_id=sample_id,
                    query=query,
                    correct_tools=correct_tools,
                    original_query=str(original_query).strip() if original_query else None,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_proxy_queries(path: str | Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            tool_name = str(item.get("tool_name") or item.get("skill_id") or item.get("name") or "").strip()
            query = str(item.get("query", "")).strip()
            if tool_name and query:
                grouped[tool_name].append(query)
    return dict(grouped)


def run_recall(
    *,
    samples: list[CarRecallSample],
    proxy_queries_by_tool: dict[str, list[str]],
    methods: list[str],
    top_k: int,
    bm25_proxy_top_k: int,
    embedder: Any | None = None,
    embedding_cache: str | Path | None = None,
    force_rebuild_embeddings: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unsupported = sorted(set(methods) - {"bm25", "llm", "embedding"})
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    retriever = BM25ProxyRetriever(proxy_queries_by_tool)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    if "bm25" in methods:
        rows = [
            _prediction_row(
                sample=sample,
                method_name="bm25",
                start_time=time.perf_counter(),
                candidates=retriever.retrieve(sample.query, top_k=top_k, proxy_top_k=bm25_proxy_top_k),
            )
            for sample in samples
        ]
        all_rows.extend(rows)
        summary["bm25"] = summarize_rows(rows, top_k=top_k)

    for method_name in [method for method in methods if method in {"llm", "embedding"}]:
        if embedder is None:
            raise RuntimeError(
                f"Method {method_name!r} requires embedding credentials in config or environment."
            )
        rows = run_embedding_recall(
            method_name=method_name,
            samples=samples,
            proxy_queries_by_tool=proxy_queries_by_tool,
            embedder=embedder,
            top_k=top_k,
            embedding_cache=embedding_cache,
            force_rebuild_embeddings=force_rebuild_embeddings,
        )
        all_rows.extend(rows)
        summary[method_name] = summarize_rows(rows, top_k=top_k)

    return all_rows, summary


def run_embedding_recall(
    *,
    method_name: str,
    samples: list[CarRecallSample],
    proxy_queries_by_tool: dict[str, list[str]],
    embedder: Any,
    top_k: int,
    embedding_cache: str | Path | None,
    force_rebuild_embeddings: bool,
) -> list[dict[str, Any]]:
    retriever = EmbeddingProxyRetriever(
        proxy_queries_by_tool,
        embedder=embedder,
        cache_path=embedding_cache,
        force_rebuild=force_rebuild_embeddings,
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        start_time = time.perf_counter()
        rows.append(
            _prediction_row(
                sample=sample,
                method_name=method_name,
                start_time=start_time,
                candidates=retriever.retrieve(sample.query, top_k=top_k),
                extra={"embedding_cache": str(Path(embedding_cache).resolve()) if embedding_cache else None},
            )
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("correct_tools")]
    hits = [row for row in labeled if row.get("top_k_hit") is True]
    first_hits = [row for row in labeled if row.get("final_success") is True]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    return {
        "samples": len(rows),
        "labeled_samples": len(labeled),
        "top_k": top_k,
        "recall_at_k": len(hits) / len(labeled) if labeled else None,
        "top_1_accuracy": len(first_hits) / len(labeled) if labeled else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def make_embedder(config: dict[str, Any]) -> OpenAICompatibleEmbedder:
    api_key = _resolve_api_key(config) or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Embedding recall requires embedding.api_key or embedding.api_key_env in config, "
            "or EMBEDDING_API_KEY/OPENAI_API_KEY."
        )
    return OpenAICompatibleEmbedder(
        model=str(config.get("model") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"),
        api_key=str(api_key),
        base_url=str(config.get("base_url") or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
        dimensions=int(config.get("dimensions") or config.get("dim")) if config.get("dimensions") or config.get("dim") else None,
        batch_size=int(config.get("batch_size") or 128),
        max_retries=int(config.get("max_retries") or 3),
        extra_body=config.get("extra_body") if isinstance(config.get("extra_body"), dict) else None,
        timeout=int(config.get("timeout") or os.getenv("EMBEDDING_TIMEOUT") or 120),
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_row(
    *,
    sample: CarRecallSample,
    method_name: str,
    start_time: float,
    candidates: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieved_tools = [str(item["tool_name"]) for item in candidates]
    correct = set(sample.correct_tools)
    row = {
        "method_name": method_name,
        "sample_id": sample.sample_id,
        "query": sample.query,
        "correct_tools": sample.correct_tools,
        "retrieved_tools": retrieved_tools,
        "retrieved_candidates": candidates,
        "top_k_hit": bool(correct & set(retrieved_tools)) if correct else None,
        "final_prediction": retrieved_tools[0] if retrieved_tools else "",
        "final_success": (retrieved_tools[0] in correct) if retrieved_tools and correct else None,
        "latency_ms": (time.perf_counter() - start_time) * 1000.0,
    }
    if sample.original_query:
        row["original_query"] = sample.original_query
    if extra:
        row.update(extra)
    return row


def _parse_methods(value: Any) -> list[str]:
    if isinstance(value, list):
        methods = [str(item).strip() for item in value]
    else:
        methods = [item.strip() for item in str(value).split(",")]
    return [method for method in methods if method]


def _cli_overrides(argv: list[str]) -> set[str]:
    out: set[str] = set()
    for item in argv:
        if item.startswith("--"):
            name = item[2:].split("=", 1)[0]
            out.add(name.replace("-", "_"))
    return out


def _embedding_config(config: dict[str, Any]) -> dict[str, Any]:
    for section in ("embedding", "openai_embedding"):
        value = config.get(section)
        if isinstance(value, dict):
            return value
    openai_value = config.get("openai")
    if isinstance(openai_value, dict):
        return openai_value
    return {}


def _resolve_api_key(config: dict[str, Any]) -> str | None:
    api_key = config.get("api_key")
    if api_key:
        return str(api_key)
    api_key_env = config.get("api_key_env")
    if not api_key_env:
        return None
    env_name = str(api_key_env)
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    if _looks_like_api_key(env_name):
        return env_name
    return None


def _looks_like_api_key(value: str) -> bool:
    return value.startswith(("sk-", "sk_", "ak-", "ak_")) or (len(value) >= 32 and "-" in value)


def _uses_embedding_method(methods: list[str]) -> bool:
    return any(method in {"llm", "embedding"} for method in methods)


def _embedder_signature(embedder: Any) -> dict[str, Any]:
    return {
        "class": type(embedder).__name__,
        "base_url": getattr(embedder, "base_url", None),
        "model": getattr(embedder, "model", None),
        "dimensions": getattr(embedder, "dimensions", None),
        "extra_body": getattr(embedder, "extra_body", None),
    }


def _load_embedding_cache(path: Path | None, signature: dict[str, Any], expected_count: int) -> list[list[float]] | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            header = json.loads(first) if first.strip() else {}
            if header.get("signature") != signature:
                return None
            vectors = [json.loads(line)["embedding"] for line in handle if line.strip()]
    except Exception:
        return None
    return vectors if len(vectors) == expected_count else None


def _write_embedding_cache(path: Path | None, signature: dict[str, Any], vectors: list[list[float]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
        for vector in vectors:
            handle.write(json.dumps({"embedding": vector}, ensure_ascii=False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
