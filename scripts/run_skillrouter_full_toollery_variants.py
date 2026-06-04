from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_skillrouter_toollery import (  # noqa: E402
    _format_path,
    _include_task,
    _load_jsonl,
    _resolve_data_root,
    _write_json,
    evaluate_predictions,
    load_skillrouter_pool,
)
from toollery.embeddings import HashingTfidfEmbedder, cosine  # noqa: E402
from toollery.io import load_manual  # noqa: E402
from toollery.schemas import ManualEntry, ToolSpec  # noqa: E402
from toollery.text import tokenize  # noqa: E402


class ManualBM25Index:
    def __init__(self, manual: list[ManualEntry]) -> None:
        self.manual = manual
        self.doc_lengths: list[int] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        dfs: Counter[str] = Counter()
        for idx, entry in enumerate(manual):
            tokens = tokenize(entry.query)
            counts = Counter(tokens)
            self.doc_lengths.append(len(tokens))
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
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for idx, tf in postings:
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[idx] += idf * tf * (self.k1 + 1.0) / denom
        return heapq.nlargest(proxy_top_k, scores.items(), key=lambda item: item[1])


class ManualHashingTfidfIndex:
    def __init__(self, manual: list[ManualEntry], dimensions: int = 2048) -> None:
        self.manual = manual
        self.embedder = HashingTfidfEmbedder(dimensions=dimensions)
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


def retrieve_tools(
    query: str,
    *,
    manual: list[ManualEntry],
    tools_by_name: dict[str, ToolSpec],
    index: ManualBM25Index | ManualHashingTfidfIndex,
    top_k: int,
    proxy_top_k: int,
) -> list[str]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for idx, score in index.search(query, proxy_top_k=max(proxy_top_k, top_k)):
        tool_name = manual[idx].tool_name
        if tool_name in tools_by_name:
            grouped[tool_name].append(score)
    candidates: list[tuple[str, float]] = []
    for tool_name, scores in grouped.items():
        ranked = sorted(scores, reverse=True)
        score = sum(ranked[:3]) / min(len(ranked), 3)
        score += 0.05 * min(len(ranked), 5)
        candidates.append((tool_name, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [tool_name for tool_name, _ in candidates[:top_k]]


def run_variant(
    *,
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    pool: list[ToolSpec],
    manual: list[ManualEntry],
    method: str,
    top_k: int,
    proxy_top_k: int,
    task_mode: str,
    limit_tasks: int | None,
) -> dict[str, list[str]]:
    print(f"[{method}] building verified-query index entries={len(manual)}", file=sys.stderr, flush=True)
    if method == "bm25":
        index: ManualBM25Index | ManualHashingTfidfIndex = ManualBM25Index(manual)
    elif method == "rag":
        index = ManualHashingTfidfIndex(manual)
    else:
        raise ValueError(f"unknown method: {method}")

    tools_by_name = {tool.name: tool for tool in pool}
    included_tasks = [task for task in tasks if _include_task(str(task["task_id"]), relevance, task_mode)]
    if limit_tasks is not None:
        included_tasks = included_tasks[:limit_tasks]
    predictions: dict[str, list[str]] = {}
    for idx, task in enumerate(included_tasks, start=1):
        task_id = str(task["task_id"])
        predictions[task_id] = retrieve_tools(
            str(task["instruction_text"]),
            manual=manual,
            tools_by_name=tools_by_name,
            index=index,
            top_k=top_k,
            proxy_top_k=proxy_top_k,
        )
        print(f"[{method}] retrieving tasks {idx}/{len(included_tasks)}", file=sys.stderr, flush=True)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillrouter-root", required=True)
    parser.add_argument("--data-root", default="data/eval_core")
    parser.add_argument("--tiers", nargs="+", default=["easy", "hard"])
    parser.add_argument("--methods", default="bm25,rag")
    parser.add_argument("--manual-out", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-out", default="{method}/retrieval/{tier}.json")
    parser.add_argument("--metrics-out", default="{method}/metrics/{tier}.json")
    parser.add_argument("--summary-out", default="summary.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--proxy-top-k", type=int, default=1000)
    parser.add_argument("--task-mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--limit-pool", type=int)
    parser.add_argument("--skill-body-chars", type=int, default=8000)
    return parser.parse_args()


def _method_path(template: str, *, method: str, tier: str, output_root: Path) -> Path:
    path = Path(template.format(method=method, tier=tier))
    return path if path.is_absolute() else output_root / path


def main() -> None:
    args = parse_args()
    skillrouter_root = Path(args.skillrouter_root)
    data_root = _resolve_data_root(skillrouter_root, args.data_root)
    output_root = Path(args.output_dir)
    tasks = _load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    summary: dict[str, Any] = {}

    for method in methods:
        method_summary: dict[str, Any] = {}
        for tier in args.tiers:
            pool = load_skillrouter_pool(data_root / tier, limit=args.limit_pool, body_chars=args.skill_body_chars)
            manual_path = _format_path(args.manual_out, tier, output_root)
            manual = load_manual(manual_path)
            predictions = run_variant(
                tasks=tasks,
                relevance=relevance,
                pool=pool,
                manual=manual,
                method=method,
                top_k=args.top_k,
                proxy_top_k=args.proxy_top_k,
                task_mode=args.task_mode,
                limit_tasks=args.limit_tasks,
            )
            predictions_path = _method_path(args.predictions_out, method=method, tier=tier, output_root=output_root)
            metrics_path = _method_path(args.metrics_out, method=method, tier=tier, output_root=output_root)
            _write_json(predictions_path, predictions)
            metrics = evaluate_predictions(
                tasks=tasks,
                relevance=relevance,
                predictions=predictions,
                pool_ids={tool.name for tool in pool},
                task_mode=args.task_mode,
            )
            _write_json(metrics_path, metrics)
            method_summary[tier] = {
                "pool_size": len(pool),
                "verified_manual_entries": len(manual),
                "predicted_tasks": len(predictions),
                "manual": str(manual_path.resolve()),
                "predictions": str(predictions_path.resolve()),
                "metrics": str(metrics_path.resolve()),
                "metrics_summary": metrics,
            }
        summary[method] = method_summary

    if args.summary_out:
        _write_json(output_root / args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
