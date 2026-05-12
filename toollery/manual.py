from __future__ import annotations

import random

from .llm import HeuristicTeacher, HeuristicVerifier, Teacher, Verifier
from .schemas import ManualEntry, ToolSpec


def synthesize_tool_manual(
    tools: list[ToolSpec],
    teacher: Teacher | None = None,
    verifier: Verifier | None = None,
    queries_per_tool: int = 8,
    distractor_count: int = 8,
    seed: int = 7,
) -> list[ManualEntry]:
    """Algorithm 1 from the paper: synthesize, then round-trip verify."""

    teacher = teacher or HeuristicTeacher()
    verifier = verifier or HeuristicVerifier()
    rng = random.Random(seed)
    manual: list[ManualEntry] = []

    for tool in tools:
        candidate_queries = teacher.generate_queries(tool, queries_per_tool)
        for query in candidate_queries:
            distractors = _sample_distractors(tools, tool.name, distractor_count, rng)
            verification_pool = [tool, *distractors]
            rng.shuffle(verification_pool)
            selected = verifier.select_tool(query, verification_pool)
            if selected == tool.name:
                manual.append(
                    ManualEntry(
                        query=query,
                        tool_name=tool.name,
                        source=teacher.__class__.__name__,
                        verification_score=1.0,
                    )
                )
    return manual


def _sample_distractors(
    tools: list[ToolSpec],
    target_name: str,
    count: int,
    rng: random.Random,
) -> list[ToolSpec]:
    candidates = [tool for tool in tools if tool.name != target_name]
    if len(candidates) <= count:
        return candidates
    return rng.sample(candidates, count)
