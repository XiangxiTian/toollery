from __future__ import annotations

import random
from dataclasses import dataclass

from .pipeline import ToolleryAgent
from .schemas import ManualEntry, ToolSpec


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    ground_truth_tool: str


@dataclass(frozen=True)
class EvaluationResult:
    candidate_size: int
    accuracy: float
    total: int
    correct: int


def make_candidate_pool(
    tools: list[ToolSpec],
    ground_truth_tool: str,
    size: int,
    rng: random.Random,
) -> list[ToolSpec]:
    target = next(tool for tool in tools if tool.name == ground_truth_tool)
    distractors = [tool for tool in tools if tool.name != ground_truth_tool]
    sample_size = max(0, min(size - 1, len(distractors)))
    pool = [target, *rng.sample(distractors, sample_size)]
    rng.shuffle(pool)
    return pool


def evaluate_scaletool(
    tools: list[ToolSpec],
    manual: list[ManualEntry],
    cases: list[EvaluationCase],
    candidate_sizes: list[int],
    tool_top_k: int = 5,
    seed: int = 11,
) -> list[EvaluationResult]:
    """ScaleTool-style robustness test under growing distractor pools."""

    results: list[EvaluationResult] = []
    for candidate_size in candidate_sizes:
        rng = random.Random(seed + candidate_size)
        correct = 0
        total = 0
        for case in cases:
            pool = make_candidate_pool(tools, case.ground_truth_tool, candidate_size, rng)
            pool_names = {tool.name for tool in pool}
            pool_manual = [entry for entry in manual if entry.tool_name in pool_names]
            if not pool_manual:
                continue
            agent = ToolleryAgent(pool, pool_manual, tool_top_k=tool_top_k)
            call, _ = agent.run(case.query)
            correct += int(call.tool_name == case.ground_truth_tool)
            total += 1
        accuracy = correct / total if total else 0.0
        results.append(EvaluationResult(candidate_size, accuracy, total, correct))
    return results
