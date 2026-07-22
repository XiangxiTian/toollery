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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.baselines import (  # noqa: E402
    RetrievalResult,
    build_tool_document,
    ensure_safe_output_path,
    estimate_prompt_tokens,
    write_json,
    write_jsonl,
)
from toollery.embeddings import HashingTfidfEmbedder, LocalHFEmbedder, OpenAICompatibleEmbedder, cosine  # noqa: E402
from toollery.llm import OpenAICompatibleLLM  # noqa: E402
from toollery.schemas import ToolSpec  # noqa: E402
from toollery.text import tokenize  # noqa: E402


@dataclass(frozen=True)
class TaichuPair:
    sample_id: str
    query: str
    expect_intent: str


@dataclass(frozen=True)
class QMetaUnit:
    tool: ToolSpec
    intent_query: str | None
    document: str
    source: str


def load_intents(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of intent labels")
    intents = [str(item).strip() for item in data if str(item).strip()]
    if len(intents) != len(set(intents)):
        duplicates = [name for name, count in Counter(intents).items() if count > 1]
        raise ValueError(f"duplicate intents found: {duplicates[:5]}")
    return intents


def load_taichu_pairs(path: str | Path) -> list[TaichuPair]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of query/intent rows")
    pairs: list[TaichuPair] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"row {idx} is not an object")
        query = str(item.get("query", "")).strip()
        intent = str(item.get("expect_intent", "")).strip()
        if not query or not intent:
            raise ValueError(f"row {idx} is missing query or expect_intent")
        pairs.append(TaichuPair(sample_id=f"taichu_{idx:05d}", query=query, expect_intent=intent))
    return pairs


def load_proxy_query_pairs(path: str | Path) -> list[TaichuPair]:
    pairs: list[TaichuPair] = []
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            tool_name = str(item.get("tool_name") or item.get("intent") or item.get("skill_id") or "").strip()
            query = str(item.get("query", "")).strip()
            if not tool_name or not query:
                continue
            pairs.append(TaichuPair(sample_id=f"proxy_{idx:05d}", query=query, expect_intent=tool_name))
    return pairs


def make_taichu_tools(intents: list[str]) -> list[ToolSpec]:
    return [
        ToolSpec(
            name=intent,
            description=_intent_description(intent),
            parameters={},
            category="taichu_intent",
        )
        for intent in intents
    ]


class QMetaBM25Index:
    """BM25 over Toollery-style intent-query documents concatenated with metadata."""

    def __init__(self, tools: list[ToolSpec], pairs: list[TaichuPair], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.units = self._make_units(tools, pairs)
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        dfs: Counter[str] = Counter()
        for idx, unit in enumerate(self.units):
            counts = Counter(tokenize(unit.document))
            self.doc_lengths.append(sum(counts.values()))
            dfs.update(counts.keys())
            for term, tf in counts.items():
                self.inverted[term].append((idx, tf))
        self.doc_count = max(len(self.units), 1)
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def search(
        self,
        query: str,
        *,
        proxy_top_k: int | None = None,
        exclude_queries: set[str] | None = None,
    ) -> list[tuple[float, QMetaUnit]]:
        excluded = exclude_queries or set()
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            for idx, tf in self.inverted.get(term, []):
                unit = self.units[idx]
                if unit.intent_query is not None and unit.intent_query in excluded:
                    continue
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[idx] += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
        limit = proxy_top_k or len(self.units)
        ranked = heapq.nlargest(limit, scores.items(), key=lambda item: item[1])
        return [(score, self.units[idx]) for idx, score in ranked if score > 0.0]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        exclude_queries: set[str] | None = None,
    ) -> list[RetrievalResult]:
        grouped: dict[str, list[tuple[float, QMetaUnit]]] = defaultdict(list)
        for score, unit in self.search(query, proxy_top_k=len(self.units), exclude_queries=exclude_queries):
            grouped[unit.tool.name].append((score, unit))

        candidates: list[tuple[str, float, str]] = []
        for tool_name, scored_units in grouped.items():
            ranked = sorted(scored_units, key=lambda item: item[0], reverse=True)
            top_scores = [score for score, _ in ranked[:3]]
            score = sum(top_scores) / max(len(top_scores), 1)
            score += 0.05 * min(len(ranked), 5)
            candidates.append((tool_name, score, ranked[0][1].document))
        candidates.sort(key=lambda item: (-item[1], item[0]))

        limit = min(top_k, len(self.tools))
        retrieved = [
            RetrievalResult(tool=self.tools_by_name[name], score=score, document=document)
            for name, score, document in candidates[:limit]
            if name in self.tools_by_name
        ]
        seen = {hit.tool.name for hit in retrieved}
        for tool in self.tools:
            if len(retrieved) >= limit:
                break
            if tool.name in seen:
                continue
            retrieved.append(RetrievalResult(tool=tool, score=0.0, document=build_tool_document(tool)))
            seen.add(tool.name)
        return retrieved

    @staticmethod
    def _make_units(tools: list[ToolSpec], pairs: list[TaichuPair]) -> list[QMetaUnit]:
        tools_by_name = {tool.name: tool for tool in tools}
        units = [
            QMetaUnit(
                tool=tool,
                intent_query=None,
                document=build_tool_document(tool),
                source="metadata_only",
            )
            for tool in tools
        ]
        for pair in pairs:
            tool = tools_by_name.get(pair.expect_intent)
            if tool is None:
                continue
            metadata = build_tool_document(tool)
            units.append(
                QMetaUnit(
                    tool=tool,
                    intent_query=pair.query,
                    document=f"{metadata}\nUSER_INTENT: {pair.query}",
                    source="intent_plus_metadata",
                )
            )
        return units


class QMetaDenseIndex:
    """Dense retrieval over Toollery query+metadata units, aggregated to intent level."""

    def __init__(
        self,
        tools: list[ToolSpec],
        pairs: list[TaichuPair],
        *,
        embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder,
        embedding_progress_callback: Any | None = None,
    ) -> None:
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.units = QMetaBM25Index._make_units(tools, pairs)
        self.embedder = embedder
        documents = [unit.document for unit in self.units]
        self.embedder.fit(documents)
        self._vectors = self.embedder.encode_many(documents, progress_callback=embedding_progress_callback)

    def encode_queries(self, queries: list[str], *, progress_callback: Any | None = None) -> list[list[float]]:
        return self.embedder.encode_many(queries, progress_callback=progress_callback)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        exclude_queries: set[str] | None = None,
    ) -> list[RetrievalResult]:
        return self.retrieve_from_vector(
            query_vector=self.embedder.encode(query),
            top_k=top_k,
            exclude_queries=exclude_queries,
        )

    def retrieve_from_vector(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        exclude_queries: set[str] | None = None,
    ) -> list[RetrievalResult]:
        excluded = exclude_queries or set()
        grouped: dict[str, list[tuple[float, QMetaUnit]]] = defaultdict(list)
        for unit, vector in zip(self.units, self._vectors):
            if unit.intent_query is not None and unit.intent_query in excluded:
                continue
            grouped[unit.tool.name].append((cosine(query_vector, vector), unit))

        candidates: list[tuple[str, float, str]] = []
        for tool_name, scored_units in grouped.items():
            ranked = sorted(scored_units, key=lambda item: item[0], reverse=True)
            top_scores = [score for score, _ in ranked[:3]]
            score = sum(top_scores) / max(len(top_scores), 1)
            score += 0.05 * min(len(ranked), 5)
            candidates.append((tool_name, score, ranked[0][1].document))
        candidates.sort(key=lambda item: (-item[1], item[0]))

        limit = min(top_k, len(self.tools))
        retrieved = [
            RetrievalResult(tool=self.tools_by_name[name], score=score, document=document)
            for name, score, document in candidates[:limit]
            if name in self.tools_by_name
        ]
        seen = {hit.tool.name for hit in retrieved}
        for tool in self.tools:
            if len(retrieved) >= limit:
                break
            if tool.name in seen:
                continue
            retrieved.append(RetrievalResult(tool=tool, score=0.0, document=build_tool_document(tool)))
            seen.add(tool.name)
        return retrieved


def run_leave_one_out(
    *,
    tools: list[ToolSpec],
    pairs: list[TaichuPair],
    proxy_pairs: list[TaichuPair] | None = None,
    top_k: int,
    limit_samples: int | None = None,
    method_name: str = "full_toollery_qmeta_bm25_zh_loo",
) -> list[dict[str, Any]]:
    eval_pairs = pairs[:limit_samples] if limit_samples is not None else pairs
    index = QMetaBM25Index(tools, proxy_pairs if proxy_pairs is not None else pairs)
    predictions: list[dict[str, Any]] = []
    for row_idx, pair in enumerate(eval_pairs, start=1):
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        hits = index.retrieve(pair.query, top_k=top_k, exclude_queries={pair.query})
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        retrieved = [hit.tool.name for hit in hits]
        scores = [float(hit.score) for hit in hits]
        rank = _rank_of(pair.expect_intent, retrieved)
        predictions.append(
            {
                "method_setting": "full_toollery",
                "method_name": method_name,
                "benchmark": "taichu_test",
                "sample_id": pair.sample_id,
                "query": pair.query,
                "candidate_pool_size": len(tools),
                "correct_intent": pair.expect_intent,
                "retrieved_candidates": retrieved,
                "scores": scores,
                "rank": rank,
                "top_1_hit": rank == 1,
                "top_k_hit": rank is not None and rank <= min(top_k, len(tools)),
                "final_prediction": retrieved[0] if retrieved else "",
                "final_success": bool(retrieved and retrieved[0] == pair.expect_intent),
                "prompt_tokens": estimate_prompt_tokens(pair.query, [hit.tool for hit in hits]),
                "retrieval_latency_ms": retrieval_latency_ms,
                "total_latency_ms": (time.perf_counter() - start) * 1000.0,
                "leave_one_out_excluded_query": pair.query,
            }
        )
        if row_idx == len(eval_pairs) or row_idx % 500 == 0:
            print(f"[taichu qmeta bm25] evaluated {row_idx}/{len(eval_pairs)} samples", file=sys.stderr, flush=True)
    return predictions


def run_dense_predictions(
    *,
    tools: list[ToolSpec],
    pairs: list[TaichuPair],
    proxy_pairs: list[TaichuPair],
    top_k: int,
    embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder,
    limit_samples: int | None = None,
    method_name: str = "full_toollery_qmeta_dense_zh",
) -> list[dict[str, Any]]:
    eval_pairs = pairs[:limit_samples] if limit_samples is not None else pairs
    index = QMetaDenseIndex(
        tools,
        proxy_pairs,
        embedder=embedder,
        embedding_progress_callback=_progress_callback("taichu dense docs"),
    )
    query_vectors = index.encode_queries(
        [pair.query for pair in eval_pairs],
        progress_callback=_progress_callback("taichu dense queries"),
    )
    predictions: list[dict[str, Any]] = []
    for row_idx, (pair, query_vector) in enumerate(zip(eval_pairs, query_vectors), start=1):
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        hits = index.retrieve_from_vector(
            query_vector=query_vector,
            top_k=top_k,
            exclude_queries={pair.query},
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        retrieved = [hit.tool.name for hit in hits]
        scores = [float(hit.score) for hit in hits]
        rank = _rank_of(pair.expect_intent, retrieved)
        predictions.append(
            {
                "method_setting": "full_toollery",
                "method_name": method_name,
                "benchmark": "taichu_test",
                "sample_id": pair.sample_id,
                "query": pair.query,
                "candidate_pool_size": len(tools),
                "correct_intent": pair.expect_intent,
                "retrieved_candidates": retrieved,
                "scores": scores,
                "rank": rank,
                "top_1_hit": rank == 1,
                "top_k_hit": rank is not None and rank <= min(top_k, len(tools)),
                "final_prediction": retrieved[0] if retrieved else "",
                "final_success": bool(retrieved and retrieved[0] == pair.expect_intent),
                "prompt_tokens": estimate_prompt_tokens(pair.query, [hit.tool for hit in hits]),
                "retrieval_latency_ms": retrieval_latency_ms,
                "total_latency_ms": (time.perf_counter() - start) * 1000.0,
                "leave_one_out_excluded_query": pair.query,
            }
        )
        if row_idx == len(eval_pairs) or row_idx % 500 == 0:
            print(f"[taichu qmeta dense] evaluated {row_idx}/{len(eval_pairs)} samples", file=sys.stderr, flush=True)
    return predictions


def generate_taichu_proxy_rows_batch(llm: Any, tools: list[ToolSpec], count: int) -> list[dict[str, Any]]:
    prompt = (
        "你在为中文意图识别/工具选择 benchmark 生成 Toollery proxy user queries。\n"
        "每个 intent 是一个可选择的工具/意图标签。请只根据 intent 的名称、描述和类别生成自然中文用户请求，"
        "不要使用任何评测集里的真实 query，也不要照抄标签文本。\n"
        "请求应该像真实用户对手机、车机、IoT/智能家居、语音助手发出的短句，可以包含口语化表达、同义改写、"
        "省略主语、轻微歧义但必须仍然指向对应 intent。\n"
        "只返回一个合法 JSON object：key 必须是原始 intent 名称，value 是数组；"
        "数组里每个元素可以是字符串，或包含 query/scenario_type/generation_notes 的 object。\n"
        f"每个 intent 恰好生成 {count} 条 query。\n\n"
        f"Intents:\n{json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False)}"
    )
    data = _extract_json_payload(
        llm._chat(
            prompt,
            stage="taichu_query_generation_batch",
            metadata={
                "intent_count": len(tools),
                "query_count": len(tools) * count,
                "intent_ids": [tool.name for tool in tools],
            },
        )
    )
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object for Taichu proxy generation")

    rows: list[dict[str, Any]] = []
    for tool in tools:
        raw_items = data.get(tool.name, [])
        if not isinstance(raw_items, list):
            raw_items = []
        for index, item in enumerate(raw_items[:count]):
            if isinstance(item, dict):
                query = str(item.get("query", "")).strip()
                scenario_type = str(item.get("scenario_type", "taichu_intent"))
                generation_notes = str(item.get("generation_notes", ""))
            else:
                query = str(item).strip()
                scenario_type = "taichu_intent"
                generation_notes = "string_item"
            rows.append(
                {
                    "stage": "candidate",
                    "skill_id": tool.name,
                    "tool_name": tool.name,
                    "query_index": index,
                    "query": query,
                    "scenario_type": scenario_type,
                    "generation_notes": generation_notes,
                    "accepted": bool(query),
                    "verifier_choice": None,
                    "rejection_reason": None if query else "empty_query",
                    "source": "OpenAICompatibleLLM",
                }
            )
    missing = [tool.name for tool in tools if sum(1 for row in rows if row["tool_name"] == tool.name and row["accepted"]) < count]
    if missing:
        raise ValueError(f"LLM returned too few usable proxy queries for intents: {missing[:10]}")
    return rows


def generate_or_load_proxy_queries(
    *,
    tools: list[ToolSpec],
    output_path: Path,
    manual_raw_path: Path,
    count: int,
    batch_size: int,
    llm: OpenAICompatibleLLM,
    force_rebuild: bool,
) -> list[TaichuPair]:
    if output_path.exists() and manual_raw_path.exists() and not force_rebuild:
        return load_proxy_query_pairs(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manual_raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = [] if force_rebuild or not manual_raw_path.exists() else _read_proxy_raw_rows(manual_raw_path)
    completed = _completed_proxy_tool_names(raw_rows, count)
    pending_tools = [tool for tool in tools if tool.name not in completed]
    with manual_raw_path.open("w", encoding="utf-8") as raw_handle:
        for row in raw_rows:
            raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for start in range(0, len(tools), max(1, batch_size)):
            batch = pending_tools[start : start + max(1, batch_size)]
            if not batch:
                continue
            batch_rows = _generate_proxy_rows_resilient(llm, batch, count)
            for row in batch_rows:
                raw_rows.append(row)
                raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_handle.flush()
            print(
                f"[taichu proxy generation] generated {len(_completed_proxy_tool_names(raw_rows, count))}/{len(tools)} intents",
                file=sys.stderr,
                flush=True,
            )
    write_normalized_proxy_queries(output_path, raw_rows)
    return load_proxy_query_pairs(output_path)


def _generate_proxy_rows_resilient(llm: Any, tools: list[ToolSpec], count: int) -> list[dict[str, Any]]:
    try:
        return generate_taichu_proxy_rows_batch(llm, tools, count=count)
    except Exception as batch_exc:
        rows: list[dict[str, Any]] = []
        for tool in tools:
            try:
                rows.extend(generate_taichu_proxy_rows_batch(llm, [tool], count=count))
            except Exception as exc:
                rows.extend(_proxy_generation_error_rows(tool.name, count, exc or batch_exc))
        return rows


def _proxy_generation_error_rows(tool_name: str, count: int, exc: Exception) -> list[dict[str, Any]]:
    return [
        {
            "stage": "candidate",
            "skill_id": tool_name,
            "tool_name": tool_name,
            "query_index": index,
            "query": "",
            "scenario_type": "generation_error",
            "generation_notes": "",
            "accepted": False,
            "verifier_choice": None,
            "rejection_reason": f"proxy_generation_failed: {type(exc).__name__}: {str(exc)[:1000]}",
            "source": "OpenAICompatibleLLM",
        }
        for index in range(count)
    ]


def _read_proxy_raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _completed_proxy_tool_names(rows: list[dict[str, Any]], expected_count: int) -> set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("accepted") is True and row.get("query"):
            tool_name = str(row.get("tool_name") or row.get("skill_id") or "")
            if tool_name:
                counts[tool_name] += 1
    return {tool_name for tool_name, row_count in counts.items() if row_count >= expected_count}


def write_normalized_proxy_queries(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("accepted") is not True or not row.get("query"):
                continue
            handle.write(
                json.dumps(
                    {
                        "tool_name": str(row.get("tool_name") or row.get("skill_id")),
                        "query": str(row["query"]),
                        "source": str(row.get("source", "OpenAICompatibleLLM")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    evaluated = [row for row in rows if row.get("correct_intent")]
    if not evaluated:
        return {"count": 0}
    ranks = [_rank_of(str(row["correct_intent"]), [str(item) for item in row.get("retrieved_candidates", [])]) for row in evaluated]
    return {
        "count": len(evaluated),
        "Hit@1": _mean(1.0 if rank == 1 else 0.0 for rank in ranks),
        "Recall@3": _mean(1.0 if rank is not None and rank <= 3 else 0.0 for rank in ranks),
        "Recall@5": _mean(1.0 if rank is not None and rank <= 5 else 0.0 for rank in ranks),
        "Recall@10": _mean(1.0 if rank is not None and rank <= 10 else 0.0 for rank in ranks),
        "Recall@50": _mean(1.0 if rank is not None and rank <= 50 else 0.0 for rank in ranks),
        "MRR@10": _mean((1.0 / rank) if rank is not None and rank <= 10 else 0.0 for rank in ranks),
        "nDCG@10": _mean((1.0 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0 for rank in ranks),
        "missing": sum(1 for rank in ranks if rank is None),
        "mean_rank": _mean(float(rank) for rank in ranks if rank is not None),
    }


def data_summary(intents: list[str], pairs: list[TaichuPair]) -> dict[str, Any]:
    counts = Counter(pair.expect_intent for pair in pairs)
    query_counts = Counter(pair.query for pair in pairs)
    missing_from_intents = sorted(set(counts) - set(intents))
    missing_labels = sorted(set(intents) - set(counts))
    return {
        "intent_count": len(intents),
        "pair_count": len(pairs),
        "labeled_intent_count": len(counts),
        "missing_from_intent_list": missing_from_intents,
        "intents_without_queries": missing_labels,
        "min_queries_per_intent": min(counts.values()) if counts else 0,
        "max_queries_per_intent": max(counts.values()) if counts else 0,
        "duplicate_query_strings": sum(1 for count in query_counts.values() if count > 1),
        "top_intents_by_query_count": [{"intent": name, "count": count} for name, count in counts.most_common(20)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing intent_list.json and query_intent_pair.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--proxy-queries", help="Optional normalized JSONL proxy queries with {tool_name, query}.")
    parser.add_argument("--generate-proxies", action="store_true", help="Generate proxy queries with an OpenAI-compatible LLM before evaluation.")
    parser.add_argument("--generated-queries-out", default="outputs/experiments_v2/taichu_test/generated_queries/deepseek_v4pro.jsonl")
    parser.add_argument("--manual-raw-out", default="outputs/experiments_v2/taichu_test/generated_queries/manual_raw_deepseek_v4pro.jsonl")
    parser.add_argument("--proxy-queries-per-intent", type=int, default=3)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--force-rebuild-proxies", action="store_true")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-extra-body-json", help="Optional JSON object merged into chat completion requests.")
    parser.add_argument("--llm-usage-log", help="Optional JSONL usage log path. API keys are not logged.")
    parser.add_argument("--dense-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dense-embedding-model-path")
    parser.add_argument("--dense-embedding-device", default="mps")
    parser.add_argument("--dense-embedding-batch-size", type=int, default=16)
    parser.add_argument("--dense-embedding-max-length", type=int, default=256)
    parser.add_argument("--dense-embedding-pooling", choices=["last", "mean", "cls"], default="last")
    parser.add_argument("--dense-embedding-dtype")
    parser.add_argument("--dense-embedding-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--dense-embedding-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dense-embedding-implementation",
        choices=["auto", "sentence-transformers", "transformers"],
        default="transformers",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments_v2/taichu_test/full_toollery_qmeta_bm25_zh_loo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_safe_output_path(output_dir)

    intents = load_intents(data_dir / "intent_list.json")
    pairs = load_taichu_pairs(data_dir / "query_intent_pair.json")
    summary_data = data_summary(intents, pairs)
    if summary_data["missing_from_intent_list"]:
        raise ValueError(f"labels missing from intent_list: {summary_data['missing_from_intent_list'][:5]}")
    tools = make_taichu_tools(intents)
    proxy_pairs, proxy_source, proxy_artifact = resolve_proxy_pairs(args, tools, pairs)
    if args.retriever == "dense":
        method_name = (
            "full_toollery_qmeta_dense_qwen3_0_6b_zh_llm_generated"
            if proxy_source == "llm_generated"
            else "full_toollery_qmeta_dense_qwen3_0_6b_zh_dataset_proxy_loo"
        )
        predictions = run_dense_predictions(
            tools=tools,
            pairs=pairs,
            proxy_pairs=proxy_pairs,
            top_k=args.top_k,
            embedder=make_dense_embedder(args),
            limit_samples=args.limit_samples,
            method_name=method_name,
        )
    else:
        method_name = (
            "full_toollery_qmeta_bm25_zh_llm_generated"
            if proxy_source == "llm_generated"
            else "full_toollery_qmeta_bm25_zh_dataset_proxy_loo"
        )
        predictions = run_leave_one_out(
            tools=tools,
            pairs=pairs,
            proxy_pairs=proxy_pairs,
            top_k=args.top_k,
            limit_samples=args.limit_samples,
            method_name=method_name,
        )

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary = {
        "method": "dense_qwen3_0_6b_query_plus_metadata_zh"
        if args.retriever == "dense"
        else "bm25_query_plus_metadata_zh",
        "method_setting": "full_toollery",
        "retrieval_unit": "metadata-only intent documents plus proxy user-intent query concatenated with intent metadata",
        "retriever": args.retriever,
        "dense_embedding": dense_embedding_summary(args) if args.retriever == "dense" else None,
        "tokenizer": "toollery.text.tokenize (Chinese phrase/ngram tokenizer plus configured synonyms)",
        "bm25": {"k1": 1.5, "b": 0.75} if args.retriever == "bm25" else None,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "proxy_source": proxy_source,
        "proxy_artifact": str(proxy_artifact.resolve()) if proxy_artifact else None,
        "proxy_query_count": len(proxy_pairs),
        "intent_count": len(intents),
        "query_intent_pair_count": len(pairs),
        "evaluated_count": len(predictions),
        "top_k": args.top_k,
        "leave_one_out": proxy_source == "dataset",
        "exact_query_exclusion": True,
        "data_summary": summary_data,
        "metrics_summary": retrieval_metrics(predictions),
        "efficiency": {
            "avg_prompt_tokens": _mean(float(row["prompt_tokens"]) for row in predictions),
            "avg_retrieval_latency_ms": _mean(float(row["retrieval_latency_ms"]) for row in predictions),
            "avg_total_latency_ms": _mean(float(row["total_latency_ms"]) for row in predictions),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "data_summary.json", summary_data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_proxy_pairs(
    args: argparse.Namespace,
    tools: list[ToolSpec],
    dataset_pairs: list[TaichuPair],
) -> tuple[list[TaichuPair], str, Path | None]:
    if args.generate_proxies:
        output_path = Path(args.generated_queries_out)
        manual_raw_path = Path(args.manual_raw_out)
        if args.llm_usage_log:
            os.environ["LLM_USAGE_LOG"] = args.llm_usage_log
        api_key = os.getenv(args.llm_api_key_env)
        if not api_key:
            raise RuntimeError(f"{args.llm_api_key_env} is required for --generate-proxies")
        extra_body = json.loads(args.llm_extra_body_json) if args.llm_extra_body_json else None
        llm = OpenAICompatibleLLM(
            model=args.llm_model,
            api_key=api_key,
            base_url=args.llm_base_url,
            extra_body=extra_body,
            timeout=args.llm_timeout,
        )
        pairs = generate_or_load_proxy_queries(
            tools=tools,
            output_path=output_path,
            manual_raw_path=manual_raw_path,
            count=args.proxy_queries_per_intent,
            batch_size=args.llm_batch_size,
            llm=llm,
            force_rebuild=args.force_rebuild_proxies,
        )
        return pairs, "llm_generated", output_path
    if args.proxy_queries:
        path = Path(args.proxy_queries)
        return load_proxy_query_pairs(path), "llm_generated", path
    return dataset_pairs, "dataset", None


def make_dense_embedder(args: argparse.Namespace) -> LocalHFEmbedder:
    return LocalHFEmbedder(
        model=args.dense_embedding_model,
        model_path=args.dense_embedding_model_path,
        device=args.dense_embedding_device,
        batch_size=args.dense_embedding_batch_size,
        max_length=args.dense_embedding_max_length,
        pooling=args.dense_embedding_pooling,
        dtype=args.dense_embedding_dtype,
        trust_remote_code=args.dense_embedding_trust_remote_code,
        local_files_only=args.dense_embedding_local_files_only,
        implementation=args.dense_embedding_implementation,
    )


def dense_embedding_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "local-hf",
        "model": args.dense_embedding_model,
        "model_path": args.dense_embedding_model_path,
        "device": args.dense_embedding_device,
        "batch_size": args.dense_embedding_batch_size,
        "max_length": args.dense_embedding_max_length,
        "pooling": args.dense_embedding_pooling,
        "dtype": args.dense_embedding_dtype,
        "local_files_only": args.dense_embedding_local_files_only,
        "implementation": args.dense_embedding_implementation,
    }


def _intent_description(intent: str) -> str:
    body = intent.split(".", 1)[1] if "." in intent else intent
    return body.replace("_", " ").replace("-", " ").strip() or intent


def _rank_of(target: str, ranked: list[str]) -> int | None:
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def _mean(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _progress_callback(label: str) -> Any:
    last_done = -1

    def callback(done: int, total: int) -> None:
        nonlocal last_done
        if total <= 0:
            return
        final = done >= total
        step = max(1, total // 20)
        if not final and done - last_done < step:
            return
        last_done = done
        print(f"[{label}] encoded {done}/{total}", file=sys.stderr, flush=True)

    return callback


def _extract_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start_object, end_object = text.find("{"), text.rfind("}")
        if start_object != -1 and end_object != -1:
            try:
                return json.loads(text[start_object : end_object + 1])
            except json.JSONDecodeError as inner_exc:
                raise ValueError(f"Invalid JSON object in LLM response: {inner_exc}") from inner_exc
        start_array, end_array = text.find("["), text.rfind("]")
        if start_array != -1 and end_array != -1:
            try:
                return json.loads(text[start_array : end_array + 1])
            except json.JSONDecodeError as inner_exc:
                raise ValueError(f"Invalid JSON array in LLM response: {inner_exc}") from inner_exc
        raise ValueError(f"No JSON payload found in LLM response: {exc}") from exc


if __name__ == "__main__":
    main()
