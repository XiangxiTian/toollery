from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .manual import synthesize_tool_manual
from .pipeline import ToolleryAgent
from .schemas import ManualEntry, ToolCandidate, ToolSpec


@dataclass(frozen=True)
class BFCLSample:
    sample_id: str
    query: str
    tools: list[ToolSpec]


@dataclass(frozen=True)
class BFCLPrediction:
    sample_id: str
    query: str
    predicted_tool: str
    candidate_tools: list[str]
    correct_tools: list[str]
    is_correct: bool | None


@dataclass(frozen=True)
class BFCLScalePrediction:
    candidate_size: int
    sample_id: str
    query: str
    predicted_tool: str
    retrieved_tools: list[str]
    scale_candidate_pool: list[str]
    correct_tools: list[str]
    is_correct: bool | None


@dataclass(frozen=True)
class BFCLScaledSample:
    candidate_size: int
    sample_id: str
    query: str
    tools: list[ToolSpec]
    correct_tools: list[str]


def load_bfcl_samples(path: str | Path) -> list[BFCLSample]:
    samples: list[BFCLSample] = []
    for item in _read_jsonl(path):
        samples.append(
            BFCLSample(
                sample_id=str(item["id"]),
                query=_extract_question_text(item.get("question", [])),
                tools=[_bfcl_function_to_tool_spec(func) for func in item.get("function", [])],
            )
        )
    return samples


def load_bfcl_answers(path: str | Path) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for item in _read_jsonl(path):
        tool_names: list[str] = []
        for call in item.get("ground_truth", []):
            if isinstance(call, dict):
                tool_names.extend(str(name) for name in call.keys())
        answers[str(item["id"])] = tool_names
    return answers


def load_bfcl_scaled_dataset(path: str | Path) -> list[BFCLScaledSample]:
    samples: list[BFCLScaledSample] = []
    for item in _read_jsonl(path):
        samples.append(
            BFCLScaledSample(
                candidate_size=int(item["candidate_size"]),
                sample_id=str(item.get("original_id", item["id"])),
                query=_extract_question_text(item.get("question", [])),
                tools=[_bfcl_function_to_tool_spec(func) for func in item.get("function", [])],
                correct_tools=[str(name) for name in item.get("correct_tools", [])],
            )
        )
    return samples


def unique_tools(samples: Iterable[BFCLSample]) -> list[ToolSpec]:
    tools_by_name: dict[str, ToolSpec] = {}
    for sample in samples:
        for tool in sample.tools:
            tools_by_name.setdefault(tool.name, tool)
    return list(tools_by_name.values())


def run_bfcl_batch(
    samples: list[BFCLSample],
    manual: list[ManualEntry] | None = None,
    answers: dict[str, list[str]] | None = None,
    tool_top_k: int = 5,
    proxy_top_k: int = 20,
    queries_per_tool: int = 8,
    distractor_count: int = 8,
    limit: int | None = None,
) -> tuple[list[BFCLPrediction], list[ManualEntry]]:
    selected_samples = samples[:limit] if limit is not None else samples
    tool_pool = unique_tools(selected_samples)
    if manual is None:
        manual = synthesize_tool_manual(
            tool_pool,
            queries_per_tool=queries_per_tool,
            distractor_count=distractor_count,
        )

    predictions: list[BFCLPrediction] = []
    for sample in selected_samples:
        candidate_names = {tool.name for tool in sample.tools}
        sample_manual = [entry for entry in manual if entry.tool_name in candidate_names]
        if sample_manual:
            agent = ToolleryAgent(
                sample.tools,
                sample_manual,
                tool_top_k=min(tool_top_k, len(sample.tools)),
                proxy_top_k=proxy_top_k,
            )
            call, candidates = agent.run(sample.query)
            candidate_tools = _candidate_names(candidates)
            predicted_tool = call.tool_name
        else:
            predicted_tool = sample.tools[0].name if sample.tools else ""
            candidate_tools = [predicted_tool] if predicted_tool else []

        correct_tools = answers.get(sample.sample_id, []) if answers else []
        is_correct = predicted_tool in correct_tools if correct_tools else None
        predictions.append(
            BFCLPrediction(
                sample_id=sample.sample_id,
                query=sample.query,
                predicted_tool=predicted_tool,
                candidate_tools=candidate_tools,
                correct_tools=correct_tools,
                is_correct=is_correct,
            )
        )

    return predictions, manual


def run_bfcl_scaled_samples(
    scaled_samples: list[BFCLScaledSample],
    manual: list[ManualEntry],
    tool_top_k: int = 5,
    proxy_top_k: int = 20,
) -> list[BFCLScalePrediction]:
    predictions: list[BFCLScalePrediction] = []
    for sample in scaled_samples:
        scaled_names = {tool.name for tool in sample.tools}
        scaled_manual = [entry for entry in manual if entry.tool_name in scaled_names]
        if scaled_manual:
            agent = ToolleryAgent(
                sample.tools,
                scaled_manual,
                tool_top_k=min(tool_top_k, len(sample.tools)),
                proxy_top_k=proxy_top_k,
            )
            call, candidates = agent.run(sample.query)
            predicted_tool = call.tool_name
            retrieved_tools = _candidate_names(candidates)
        else:
            predicted_tool = sample.tools[0].name if sample.tools else ""
            retrieved_tools = [predicted_tool] if predicted_tool else []

        predictions.append(
            BFCLScalePrediction(
                candidate_size=sample.candidate_size,
                sample_id=sample.sample_id,
                query=sample.query,
                predicted_tool=predicted_tool,
                retrieved_tools=retrieved_tools,
                scale_candidate_pool=[tool.name for tool in sample.tools],
                correct_tools=sample.correct_tools,
                is_correct=predicted_tool in sample.correct_tools if sample.correct_tools else None,
            )
        )
    return predictions


def run_bfcl_scaletool(
    samples: list[BFCLSample],
    answers: dict[str, list[str]],
    candidate_sizes: list[int],
    manual: list[ManualEntry] | None = None,
    tool_top_k: int = 5,
    proxy_top_k: int = 20,
    queries_per_tool: int = 8,
    distractor_count: int = 8,
    limit: int | None = None,
    seed: int = 11,
) -> tuple[list[BFCLScalePrediction], list[ManualEntry]]:
    """Scale BFCL candidate pools, then run Toollery on each scaled pool."""

    selected_samples = samples[:limit] if limit is not None else samples
    tool_pool = unique_tools(samples)
    tools_by_name = {tool.name: tool for tool in tool_pool}
    if manual is None:
        manual = synthesize_tool_manual(
            tool_pool,
            queries_per_tool=queries_per_tool,
            distractor_count=distractor_count,
        )

    scaled_samples = make_bfcl_scaled_samples(
        samples=selected_samples,
        answers=answers,
        tools_by_name=tools_by_name,
        candidate_sizes=candidate_sizes,
        seed=seed,
    )
    predictions = run_bfcl_scaled_samples(
        scaled_samples,
        manual=manual,
        tool_top_k=tool_top_k,
        proxy_top_k=proxy_top_k,
    )
    return predictions, manual


def make_bfcl_scaled_samples(
    samples: list[BFCLSample],
    answers: dict[str, list[str]],
    tools_by_name: dict[str, ToolSpec],
    candidate_sizes: list[int],
    seed: int = 11,
) -> list[BFCLScaledSample]:
    scaled_samples: list[BFCLScaledSample] = []
    for candidate_size in candidate_sizes:
        rng = random.Random(seed + candidate_size)
        for sample in samples:
            correct_tools = [name for name in answers.get(sample.sample_id, []) if name in tools_by_name]
            if not correct_tools:
                continue
            scaled_pool = _make_scaled_pool(
                tools_by_name=tools_by_name,
                correct_tools=correct_tools,
                size=candidate_size,
                rng=rng,
            )
            scaled_samples.append(
                BFCLScaledSample(
                    candidate_size=candidate_size,
                    sample_id=sample.sample_id,
                    query=sample.query,
                    tools=scaled_pool,
                    correct_tools=correct_tools,
                )
            )
    return scaled_samples


def save_bfcl_predictions(path: str | Path, predictions: Iterable[BFCLPrediction]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.__dict__, ensure_ascii=False) + "\n")


def save_bfcl_scale_predictions(
    path: str | Path,
    predictions: Iterable[BFCLScalePrediction],
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.__dict__, ensure_ascii=False) + "\n")


def save_bfcl_scaled_dataset(
    path: str | Path,
    samples: list[BFCLSample],
    answers: dict[str, list[str]],
    candidate_sizes: list[int],
    limit: int | None = None,
    seed: int = 11,
) -> None:
    selected_samples = samples[:limit] if limit is not None else samples
    tools_by_name = {tool.name: tool for tool in unique_tools(samples)}
    scaled_samples = make_bfcl_scaled_samples(
        samples=selected_samples,
        answers=answers,
        tools_by_name=tools_by_name,
        candidate_sizes=candidate_sizes,
        seed=seed,
    )
    with Path(path).open("w", encoding="utf-8") as handle:
        for sample in scaled_samples:
            item = {
                "id": f"{sample.sample_id}_n{sample.candidate_size}",
                "original_id": sample.sample_id,
                "candidate_size": sample.candidate_size,
                "question": [[{"role": "user", "content": sample.query}]],
                "function": [tool.to_dict() for tool in sample.tools],
                "correct_tools": sample.correct_tools,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize_predictions(predictions: list[BFCLPrediction]) -> dict[str, Any]:
    evaluated = [prediction for prediction in predictions if prediction.is_correct is not None]
    correct = sum(1 for prediction in evaluated if prediction.is_correct)
    return {
        "total": len(predictions),
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else None,
    }


def summarize_scale_predictions(predictions: list[BFCLScalePrediction]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    sizes = sorted({prediction.candidate_size for prediction in predictions})
    for candidate_size in sizes:
        group = [prediction for prediction in predictions if prediction.candidate_size == candidate_size]
        evaluated = [prediction for prediction in group if prediction.is_correct is not None]
        correct = sum(1 for prediction in evaluated if prediction.is_correct)
        summaries.append(
            {
                "candidate_size": candidate_size,
                "total": len(group),
                "evaluated": len(evaluated),
                "correct": correct,
                "accuracy": correct / len(evaluated) if evaluated else None,
            }
        )
    return summaries


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_question_text(question: Any) -> str:
    texts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str):
                texts.append(content)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(question)
    return "\n".join(texts)


def _bfcl_function_to_tool_spec(function: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=str(function["name"]),
        description=str(function.get("description", "")),
        parameters=dict(function.get("parameters", {})),
        category=None,
    )


def _candidate_names(candidates: list[ToolCandidate]) -> list[str]:
    return [candidate.tool.name for candidate in candidates]


def _make_scaled_pool(
    tools_by_name: dict[str, ToolSpec],
    correct_tools: list[str],
    size: int,
    rng: random.Random,
) -> list[ToolSpec]:
    targets = [tools_by_name[name] for name in correct_tools if name in tools_by_name]
    distractors = [tool for name, tool in tools_by_name.items() if name not in set(correct_tools)]
    sample_size = max(0, min(size - len(targets), len(distractors)))
    pool = [*targets, *rng.sample(distractors, sample_size)]
    rng.shuffle(pool)
    return pool
