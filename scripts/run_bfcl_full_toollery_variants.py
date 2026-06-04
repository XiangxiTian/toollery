from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.baselines import BaselinePrediction, estimate_prompt_tokens, load_generated_queries, write_json, write_jsonl  # noqa: E402
from toollery.bfcl import load_bfcl_scaled_dataset  # noqa: E402
from toollery.embeddings import HashingTfidfEmbedder, cosine  # noqa: E402
from toollery.schemas import ManualEntry, ToolSpec  # noqa: E402
from toollery.text import tokenize  # noqa: E402


class ManualBM25Index:
    def __init__(self, manual: list[ManualEntry]) -> None:
        self.manual = manual
        self.doc_lengths: list[int] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        dfs: Counter[str] = Counter()
        for idx, entry in enumerate(manual):
            counts = Counter(tokenize(entry.query))
            self.doc_lengths.append(sum(counts.values()))
            dfs.update(counts.keys())
            for term, tf in counts.items():
                self.inverted[term].append((idx, tf))
        self.doc_count = max(len(manual), 1)
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, proxy_top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            for idx, tf in self.inverted.get(term, []):
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[idx] += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
        return heapq.nlargest(proxy_top_k, scores.items(), key=lambda item: item[1])


class ManualHashingTfidfIndex:
    def __init__(self, manual: list[ManualEntry]) -> None:
        self.manual = manual
        self.embedder = HashingTfidfEmbedder()
        queries = [entry.query for entry in manual]
        self.embedder.fit(queries)
        self.vectors = [self.embedder.encode(query) for query in queries]

    def search(self, query: str, proxy_top_k: int) -> list[tuple[int, float]]:
        query_vector = self.embedder.encode(query)
        scored = [
            (idx, score)
            for idx, vector in enumerate(self.vectors)
            if (score := cosine(query_vector, vector)) > 0.0
        ]
        return heapq.nlargest(proxy_top_k, scored, key=lambda item: item[1])


def make_manual(samples: list[Any], generated_queries: dict[str, list[str]]) -> list[ManualEntry]:
    tool_names = {tool.name for sample in samples for tool in sample.tools}
    return [
        ManualEntry(query=query, tool_name=tool_name, source="bfcl_generated")
        for tool_name, queries in generated_queries.items()
        if tool_name in tool_names
        for query in queries
    ]


def retrieve(
    *,
    query: str,
    tools: list[ToolSpec],
    manual: list[ManualEntry],
    index: ManualBM25Index | ManualHashingTfidfIndex,
    top_k: int,
) -> list[str]:
    allowed_names = {tool.name for tool in tools}
    grouped: dict[str, list[float]] = defaultdict(list)
    for idx, score in index.search(query, proxy_top_k=len(manual)):
        tool_name = manual[idx].tool_name
        if tool_name in allowed_names:
            grouped[tool_name].append(score)
    candidates: list[tuple[str, float]] = []
    for tool_name, scores in grouped.items():
        ranked = sorted(scores, reverse=True)
        score = sum(ranked[:3]) / min(len(ranked), 3)
        score += 0.05 * min(len(ranked), 5)
        candidates.append((tool_name, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    retrieved = [tool_name for tool_name, _ in candidates[: min(top_k, len(tools))]]
    for tool in tools:
        if len(retrieved) >= min(top_k, len(tools)):
            break
        if tool.name not in retrieved:
            retrieved.append(tool.name)
    return retrieved


def run_method(
    *,
    method: str,
    samples: list[Any],
    manual: list[ManualEntry],
    top_k: int,
) -> list[BaselinePrediction]:
    print(f"[{method}] building verified-query index entries={len(manual)}", file=sys.stderr, flush=True)
    if method == "bm25":
        index: ManualBM25Index | ManualHashingTfidfIndex = ManualBM25Index(manual)
    elif method == "rag":
        index = ManualHashingTfidfIndex(manual)
    else:
        raise ValueError(f"unknown method: {method}")

    predictions: list[BaselinePrediction] = []
    for row_idx, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieved = retrieve(query=sample.query, tools=sample.tools, manual=manual, index=index, top_k=top_k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        tools_by_name = {tool.name: tool for tool in sample.tools}
        retrieved_tools = [tools_by_name[name] for name in retrieved if name in tools_by_name]
        correct_tools = sample.correct_tools
        final_prediction = retrieved[0] if retrieved else (sample.tools[0].name if sample.tools else "")
        predictions.append(
            BaselinePrediction(
                method_setting="full_toollery",
                method_name=f"full_toollery_{method}",
                benchmark="bfcl_scaled",
                sample_id=sample.sample_id,
                query=sample.query,
                candidate_pool_size=len(sample.tools),
                retrieved_candidates=retrieved,
                correct_candidates=correct_tools,
                top_k_hit=bool(set(retrieved) & set(correct_tools)) if correct_tools else None,
                final_prediction=final_prediction,
                final_success=final_prediction in correct_tools if correct_tools else None,
                prompt_tokens=estimate_prompt_tokens(sample.query, retrieved_tools),
                completion_tokens=None,
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        )
        if row_idx == len(samples) or row_idx % 50 == 0:
            print(f"[{method}] evaluated {row_idx}/{len(samples)} samples", file=sys.stderr, flush=True)
    return predictions


def retrieval_metrics(predictions: list[BaselinePrediction]) -> dict[str, float]:
    rows = []
    for prediction in predictions:
        ranked = prediction.retrieved_candidates
        gold = set(prediction.correct_candidates)
        if not gold:
            continue
        relevances = [1.0 if name in gold else 0.0 for name in ranked]
        ideal = [1.0] * len(gold) + [0.0] * max(0, len(ranked) - len(gold))
        rows.append(
            {
                "Hit@1": float(any(name in gold for name in ranked[:1])),
                "Recall@10": len(set(ranked[:10]) & gold) / len(gold),
                "Recall@50": len(set(ranked[:50]) & gold) / len(gold),
                "MRR@10": next((1.0 / (idx + 1) for idx, name in enumerate(ranked[:10]) if name in gold), 0.0),
                "nDCG@10": _ndcg(relevances, ideal, 10),
                "FullCoverage@10": float(gold.issubset(set(ranked[:10]))),
            }
        )
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    summary["count"] = len(rows)
    return summary


def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], ideal: list[float], k: int) -> float:
    ideal_dcg = _dcg(sorted(ideal, reverse=True), k)
    return _dcg(relevances, k) / ideal_dcg if ideal_dcg > 0.0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-data", required=True)
    parser.add_argument("--generated-queries", required=True)
    parser.add_argument("--methods", default="bm25,rag")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_bfcl_scaled_dataset(args.scaled_data)
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]
    generated_queries = load_generated_queries(args.generated_queries)
    manual = make_manual(samples, generated_queries)
    if not manual:
        raise ValueError("No generated queries matched tools in the scaled dataset")
    output_root = Path(args.output_dir)
    summary: dict[str, Any] = {}
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        predictions = run_method(method=method, samples=samples, manual=manual, top_k=args.top_k)
        method_name = f"full_toollery_{method}"
        method_dir = output_root / method_name
        write_jsonl(method_dir / "predictions.jsonl", (row.__dict__ for row in predictions))
        method_summary = {
            "method_setting": "full_toollery",
            "method_name": method_name,
            "benchmark": "bfcl_scaled",
            "scaled_data": str(Path(args.scaled_data).resolve()),
            "generated_queries": str(Path(args.generated_queries).resolve()),
            "top_k": args.top_k,
            "summary": retrieval_metrics(predictions),
        }
        write_json(method_dir / "summary.json", method_summary)
        summary[method_name] = method_summary
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
