from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .llm import OpenAICompatibleLLM
from .manual import synthesize_tool_manual
from .pipeline import ToolleryAgent
from .schemas import ManualEntry, ToolCandidate, ToolSpec


@dataclass(frozen=True)
class SkillBenchmarkRow:
    sample_id: str
    query: str
    skill_name: str
    correct_tools: list[str]
    scenario_type: str
    generation_notes: str
    accepted: bool | None = None
    rejection_reason: str | None = None
    verifier_choice: str | None = None


@dataclass(frozen=True)
class SkillScaledSample:
    candidate_size: int
    sample_id: str
    query: str
    tools: list[ToolSpec]
    correct_tools: list[str]


@dataclass(frozen=True)
class SkillScalePrediction:
    candidate_size: int
    sample_id: str
    query: str
    predicted_tool: str
    retrieved_tools: list[str]
    scale_candidate_pool: list[str]
    correct_tools: list[str]
    is_correct: bool | None


def load_skill_tools(root: str | Path, limit: int | None = None) -> list[ToolSpec]:
    skills_root = Path(root)
    tools: list[ToolSpec] = []
    seen: set[str] = set()
    for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        tool = _skill_dir_to_tool(skill_dir)
        if tool.name in seen:
            tool = ToolSpec(
                name=f"{tool.name}__{skill_dir.name}",
                description=tool.description,
                parameters=tool.parameters,
                category=tool.category,
            )
        seen.add(tool.name)
        tools.append(tool)
        if limit is not None and len(tools) >= limit:
            break
    return tools


def generate_skill_benchmark(
    tools: list[ToolSpec],
    skills_root: str | Path,
    llm: OpenAICompatibleLLM,
    queries_per_skill: int = 5,
    verifier_distractors: int = 8,
    seed: int = 19,
) -> tuple[list[SkillBenchmarkRow], list[SkillBenchmarkRow]]:
    raw_rows: list[SkillBenchmarkRow] = []
    verified_rows: list[SkillBenchmarkRow] = []
    rng = random.Random(seed)
    tools_by_name = {tool.name: tool for tool in tools}

    for tool in tools:
        generated = _generate_queries_for_skill(
            tool=tool,
            skill_dir=Path(skills_root) / str(tool.parameters.get("skill_dir", tool.name)),
            llm=llm,
            count=queries_per_skill,
        )
        for index, item in enumerate(generated):
            query = str(item.get("query", "")).strip()
            scenario_type = str(item.get("scenario_type", "realistic_task")).strip()
            notes = str(item.get("generation_notes", "")).strip()
            sample_id = f"{tool.name}_{index}"
            if not query:
                raw_rows.append(
                    SkillBenchmarkRow(
                        sample_id=sample_id,
                        query=query,
                        skill_name=tool.name,
                        correct_tools=[tool.name],
                        scenario_type=scenario_type,
                        generation_notes=notes,
                        accepted=False,
                        rejection_reason="empty_query",
                    )
                )
                continue

            verification_pool = [tool, *_sample_distractors(tools, tool.name, verifier_distractors, rng)]
            rng.shuffle(verification_pool)
            verifier_choice = _verify_skill_query(llm, query, verification_pool)
            accepted = verifier_choice == tool.name
            row = SkillBenchmarkRow(
                sample_id=sample_id,
                query=query,
                skill_name=tool.name,
                correct_tools=[tool.name],
                scenario_type=scenario_type,
                generation_notes=notes,
                accepted=accepted,
                rejection_reason=None if accepted else "verifier_selected_different_skill",
                verifier_choice=verifier_choice,
            )
            raw_rows.append(row)
            if accepted and tool.name in tools_by_name:
                verified_rows.append(row)
    return raw_rows, verified_rows


def make_skill_scaled_samples(
    benchmark: list[SkillBenchmarkRow],
    tools: list[ToolSpec],
    candidate_sizes: list[int],
    seed: int = 23,
) -> list[SkillScaledSample]:
    tools_by_name = {tool.name: tool for tool in tools}
    scaled_samples: list[SkillScaledSample] = []
    for candidate_size in candidate_sizes:
        rng = random.Random(seed + candidate_size)
        for row in benchmark:
            correct_tools = [name for name in row.correct_tools if name in tools_by_name]
            if not correct_tools:
                continue
            scaled_pool = _make_scaled_pool(tools_by_name, correct_tools, candidate_size, rng)
            scaled_samples.append(
                SkillScaledSample(
                    candidate_size=candidate_size,
                    sample_id=row.sample_id,
                    query=row.query,
                    tools=scaled_pool,
                    correct_tools=correct_tools,
                )
            )
    return scaled_samples


def run_skill_scaled_samples(
    scaled_samples: list[SkillScaledSample],
    manual: list[ManualEntry],
    tool_top_k: int = 3,
    proxy_top_k: int = 20,
) -> list[SkillScalePrediction]:
    predictions: list[SkillScalePrediction] = []
    for sample in scaled_samples:
        scaled_names = {tool.name for tool in sample.tools}
        sample_manual = [entry for entry in manual if entry.tool_name in scaled_names]
        if sample_manual:
            agent = ToolleryAgent(
                sample.tools,
                sample_manual,
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
            SkillScalePrediction(
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


def run_skill_scaletool(
    tools: list[ToolSpec],
    benchmark: list[SkillBenchmarkRow],
    candidate_sizes: list[int],
    manual: list[ManualEntry] | None = None,
    tool_top_k: int = 3,
    proxy_top_k: int = 20,
    manual_queries_per_tool: int = 8,
    manual_distractors: int = 8,
    seed: int = 23,
) -> tuple[list[SkillScalePrediction], list[ManualEntry], list[SkillScaledSample]]:
    if manual is None:
        manual = synthesize_tool_manual(
            tools,
            queries_per_tool=manual_queries_per_tool,
            distractor_count=manual_distractors,
        )
    scaled_samples = make_skill_scaled_samples(benchmark, tools, candidate_sizes, seed=seed)
    predictions = run_skill_scaled_samples(
        scaled_samples,
        manual=manual,
        tool_top_k=tool_top_k,
        proxy_top_k=proxy_top_k,
    )
    return predictions, manual, scaled_samples


def save_skill_tools(path: str | Path, tools: Iterable[ToolSpec]) -> None:
    payload = {"tools": [tool.to_dict() for tool in tools]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_skill_tools_file(path: str | Path) -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tools", [])
    return [ToolSpec.from_dict(item) for item in data]


def save_skill_benchmark(path: str | Path, rows: Iterable[SkillBenchmarkRow]) -> None:
    _write_jsonl(path, (row.__dict__ for row in rows))


def load_skill_benchmark(path: str | Path, verified_only: bool = False) -> list[SkillBenchmarkRow]:
    rows: list[SkillBenchmarkRow] = []
    for item in _read_jsonl(path):
        row = SkillBenchmarkRow(
            sample_id=str(item["sample_id"]),
            query=str(item["query"]),
            skill_name=str(item["skill_name"]),
            correct_tools=[str(name) for name in item.get("correct_tools", [item["skill_name"]])],
            scenario_type=str(item.get("scenario_type", "realistic_task")),
            generation_notes=str(item.get("generation_notes", "")),
            accepted=item.get("accepted"),
            rejection_reason=item.get("rejection_reason"),
            verifier_choice=item.get("verifier_choice"),
        )
        if not verified_only or row.accepted is True:
            rows.append(row)
    return rows


def save_skill_scaled_data(path: str | Path, samples: Iterable[SkillScaledSample]) -> None:
    rows = (
        {
            "id": f"{sample.sample_id}_n{sample.candidate_size}",
            "original_id": sample.sample_id,
            "candidate_size": sample.candidate_size,
            "query": sample.query,
            "function": [tool.to_dict() for tool in sample.tools],
            "correct_tools": sample.correct_tools,
        }
        for sample in samples
    )
    _write_jsonl(path, rows)


def load_skill_scaled_data(path: str | Path) -> list[SkillScaledSample]:
    samples: list[SkillScaledSample] = []
    for item in _read_jsonl(path):
        samples.append(
            SkillScaledSample(
                candidate_size=int(item["candidate_size"]),
                sample_id=str(item.get("original_id", item["id"])),
                query=str(item["query"]),
                tools=[ToolSpec.from_dict(tool) for tool in item.get("function", [])],
                correct_tools=[str(name) for name in item.get("correct_tools", [])],
            )
        )
    return samples


def save_skill_predictions(path: str | Path, predictions: Iterable[SkillScalePrediction]) -> None:
    _write_jsonl(path, (prediction.__dict__ for prediction in predictions))


def summarize_skill_predictions(predictions: list[SkillScalePrediction]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate_size in sorted({prediction.candidate_size for prediction in predictions}):
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


def _skill_dir_to_tool(skill_dir: Path) -> ToolSpec:
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="ignore")
    frontmatter, body = _split_frontmatter(skill_text)
    meta = _load_meta(skill_dir / "_meta.json")
    name = str(frontmatter.get("name") or skill_dir.name)
    description = str(frontmatter.get("description") or _first_paragraph(body) or name)
    parameters = {
        "type": "object",
        "properties": {
            "skill_dir": {"type": "string", "default": skill_dir.name},
            "skill_path": {"type": "string", "default": str(skill_dir)},
            "display_name": {"type": "string", "default": meta.get("displayName", name)},
            "owner": {"type": "string", "default": meta.get("owner", "")},
            "slug": {"type": "string", "default": meta.get("slug", skill_dir.name)},
            "has_readme": {"type": "boolean", "default": (skill_dir / "README.md").exists()},
        },
    }
    return ToolSpec(name=name, description=description, parameters=parameters, category="skill")


def _generate_queries_for_skill(
    tool: ToolSpec,
    skill_dir: Path,
    llm: OpenAICompatibleLLM,
    count: int,
) -> list[dict[str, Any]]:
    context = _skill_context(skill_dir)
    prompt = (
        "You are creating an evaluation benchmark for selecting the right AI agent skill.\n"
        "Generate realistic user requests that should trigger the target skill.\n"
        "The requests should sound like real users asking for help in practical situations.\n"
        "Include some natural ambiguity, concrete task context, and realistic constraints.\n"
        "Do not simply copy the skill description. Avoid naming the skill unless a real user would.\n"
        f"Return exactly {count} JSON objects in a JSON array. Each object must contain "
        "query, scenario_type, and generation_notes.\n\n"
        f"Target skill:\n{json.dumps(tool.to_dict(), ensure_ascii=False)}\n\n"
        f"Skill context:\n{context}"
    )
    text = llm._chat(prompt)
    data = _extract_json(text)
    if not isinstance(data, list):
        raise ValueError(f"LLM did not return a JSON list for skill {tool.name}")
    rows: list[dict[str, Any]] = []
    for item in data[:count]:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, str):
            rows.append(
                {
                    "query": item,
                    "scenario_type": "realistic_task",
                    "generation_notes": "string_response",
                }
            )
    return rows


def _verify_skill_query(
    llm: OpenAICompatibleLLM,
    query: str,
    tools: list[ToolSpec],
) -> str | None:
    names = [tool.name for tool in tools]
    prompt = (
        "You are verifying a benchmark query for skill selection.\n"
        "Choose exactly one skill that best matches the user request. "
        "Return only the skill name, or NONE if no skill fits.\n\n"
        f"User request: {query}\n\n"
        f"Candidate skills:\n{json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False)}"
    )
    answer = llm._chat(prompt).strip().strip('"').strip("'")
    if answer in names:
        return answer
    match = re.search(r"[A-Za-z0-9_.-]+", answer)
    if match and match.group(0) in names:
        return match.group(0)
    return None


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    frontmatter: dict[str, str] = {}
    lines = match.group(1).splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if ":" not in line:
            index += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {"|", ">"}:
            block: list[str] = []
            index += 1
            while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            frontmatter[key] = " ".join(part for part in block if part).strip()
            continue
        frontmatter[key] = value.strip('"')
        index += 1
    return frontmatter, text[match.end() :]


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _skill_context(skill_dir: Path) -> str:
    parts: list[str] = []
    for filename in ("SKILL.md", "README.md"):
        path = skill_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            _, body = _split_frontmatter(text)
            parts.append(f"# {filename}\n{_compact_text(body, limit=1800)}")
    return "\n\n".join(parts)[:3600]


def _first_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        stripped = re.sub(r"#+\s*", "", block).strip()
        if stripped:
            return stripped[:400]
    return ""


def _compact_text(text: str, limit: int) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped)
        if len("\n".join(lines)) >= limit:
            break
    return "\n".join(lines)[:limit]


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_array, end_array = text.find("["), text.rfind("]")
        if start_array != -1 and end_array != -1:
            return json.loads(text[start_array : end_array + 1])
        start_object, end_object = text.find("{"), text.rfind("}")
        if start_object != -1 and end_object != -1:
            return json.loads(text[start_object : end_object + 1])
    raise ValueError("No valid JSON payload found in LLM response")


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


def _make_scaled_pool(
    tools_by_name: dict[str, ToolSpec],
    correct_tools: list[str],
    size: int,
    rng: random.Random,
) -> list[ToolSpec]:
    targets = [tools_by_name[name] for name in correct_tools if name in tools_by_name]
    correct_names = set(correct_tools)
    distractors = [tool for name, tool in tools_by_name.items() if name not in correct_names]
    sample_size = max(0, min(size - len(targets), len(distractors)))
    pool = [*targets, *rng.sample(distractors, sample_size)]
    rng.shuffle(pool)
    return pool


def _candidate_names(candidates: list[ToolCandidate]) -> list[str]:
    return [candidate.tool.name for candidate in candidates]


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
