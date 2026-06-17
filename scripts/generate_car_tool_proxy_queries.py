from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import time
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_car_tools_to_bfcl import load_bfcl_tools_file  # noqa: E402
from scripts.run_skillrouter_toollery import (  # noqa: E402
    Progress,
    _extract_json,
    _sample_distractors,
    apply_llm_config,
    validate_llm_environment,
    verify_proxy_query,
)
from toollery.io import load_manual, save_manual  # noqa: E402
from toollery.llm import OpenAICompatibleLLM  # noqa: E402
from toollery.schemas import ManualEntry, ToolSpec  # noqa: E402


DEFAULT_TOOLS = "outputs/car_tools/car_tools_bfcl.json"
CAR_IMPLICIT_GUIDANCE = (
    "This is a vehicle-control and vehicle-status dataset. Generate natural Chinese user requests.\n"
    "Keep the original benchmark requirement: the query should be something a real user might ask when they need this exact tool.\n"
    "In addition to direct requests, include implicit vehicle requests when they fit the exact tool: users often describe comfort, safety, "
    "visibility, lighting, charging, seat, door/window, mirror, or environment symptoms instead of naming the API.\n"
    "Examples of implicit reasoning: '车里好热' may require air conditioning, windows, ventilation, or cabin/interior temperature; "
    "'车里有点闷' may require ventilation or windows; '太冷了' may require heating, seat heating, or closing windows; "
    "'看不清后面' may require mirrors, rear window, defogging, or camera/display status; '外面太暗' may require lights; "
    "'孩子在后排乱按' may require child lock; '手机没电了' may require wireless charging or power status.\n"
    "Only generate implicit requests that are plausible for the exact tool metadata. Do not attach unrelated vehicle actions just because an example mentions them."
)


def main() -> None:
    args = parse_args()
    args._cli_overrides = _cli_overrides(sys.argv[1:])
    apply_config(args)
    validate_llm_environment()

    tools = load_car_tool_specs(args.tools)
    if args.limit_tools is not None:
        tools = tools[: args.limit_tools]

    tools_by_name = {tool.name: tool for tool in tools}
    tool_names = sorted(tools_by_name)
    out_path = Path(args.out)
    manual_raw_path = Path(args.manual_raw_out) if args.manual_raw_out else _derive_path(out_path, ".raw.jsonl")
    manual_path = Path(args.manual_out) if args.manual_out else _derive_path(out_path, ".manual.json")

    progress = Progress("car-tool-generated")
    progress.message(f"tools={len(tool_names)} examples_for_tools=0")
    raw_rows, manual = build_or_load_car_manual(
        manual_raw_path=manual_raw_path,
        manual_path=manual_path,
        selected_skill_ids=tool_names,
        pool_by_id=tools_by_name,
        example_queries={},
        llm=RetryingLLM(
            OpenAICompatibleLLM(),
            max_retries=args.llm_max_retries,
            retry_sleep=args.llm_retry_sleep,
        ),
        proxy_queries_per_skill=args.proxy_queries_per_tool,
        verifier_distractors=args.verifier_distractors,
        verify_proxies=args.verify_proxies,
        force_rebuild=args.force_rebuild,
        seed=args.seed,
        llm_workers=args.llm_workers,
        llm_batch_size=args.llm_batch_size,
        progress=progress,
    )
    write_generated_queries(out_path, manual)

    summary = {
        "tools_file": str(Path(args.tools).resolve()),
        "out": str(out_path.resolve()),
        "manual_raw": str(manual_raw_path.resolve()),
        "manual": str(manual_path.resolve()),
        "tools": len(tool_names),
        "raw_rows": len(raw_rows),
        "generated_queries": len(manual),
        "proxy_queries_per_tool": args.proxy_queries_per_tool,
        "verify_proxies": args.verify_proxies,
    }
    if args.summary_out:
        _write_json(Path(args.summary_out), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate proxy queries for converted vehicle-control tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="skillrouter_toollery_deepseek_config.example.json")
    parser.add_argument("--tools", default=DEFAULT_TOOLS, help="Converted car tools JSON with a top-level tools array.")
    parser.add_argument("--out", required=True, help="Normalized JSONL output: one {tool_name, query} row per query.")
    parser.add_argument("--manual-raw-out", help="Raw candidate rows for resumable generation.")
    parser.add_argument("--manual-out", help="Verified/accepted manual JSON for resumable generation.")
    parser.add_argument("--summary-out", help="Optional summary JSON path.")
    parser.add_argument("--proxy-queries-per-tool", type=int, default=3)
    parser.add_argument("--verify-proxies", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verifier-distractors", type=int, default=8)
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-sleep", type=float, default=1.0)
    parser.add_argument("--limit-tools", type=int)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    apply_llm_config(config)
    values = config.get("car_tool_generated_queries", {})
    for key, value in values.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            continue
        if attr in getattr(args, "_cli_overrides", set()):
            continue
        if getattr(args, attr) in (None, parse_args_default(attr)):
            setattr(args, attr, value)


def parse_args_default(attr: str) -> Any:
    defaults = {
        "tools": DEFAULT_TOOLS,
        "proxy_queries_per_tool": 3,
        "verify_proxies": False,
        "verifier_distractors": 8,
        "llm_workers": 4,
        "llm_batch_size": 4,
        "llm_max_retries": 3,
        "llm_retry_sleep": 1.0,
        "seed": 31,
        "force_rebuild": False,
    }
    return defaults.get(attr)


def _cli_overrides(argv: list[str]) -> set[str]:
    out: set[str] = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        name = item[2:].split("=", 1)[0]
        if name.startswith("no-"):
            name = name[3:]
        out.add(name.replace("-", "_"))
    return out


class RetryingLLM:
    def __init__(
        self,
        inner: OpenAICompatibleLLM,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ) -> None:
        self.inner = inner
        self.max_retries = max(0, max_retries)
        self.retry_sleep = max(0.0, retry_sleep)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _chat(self, *args: Any, **kwargs: Any) -> str:
        attempt = 0
        while True:
            try:
                return self.inner._chat(*args, **kwargs)
            except TRANSIENT_LLM_ERRORS:
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                if self.retry_sleep:
                    time.sleep(self.retry_sleep * attempt)


TRANSIENT_LLM_ERRORS = (
    TimeoutError,
    OSError,
    http.client.RemoteDisconnected,
    urllib.error.URLError,
)


def build_or_load_car_manual(
    manual_raw_path: Path,
    manual_path: Path,
    selected_skill_ids: list[str],
    pool_by_id: dict[str, ToolSpec],
    example_queries: dict[str, list[str]],
    llm: OpenAICompatibleLLM,
    proxy_queries_per_skill: int,
    verifier_distractors: int,
    verify_proxies: bool,
    force_rebuild: bool,
    seed: int,
    llm_workers: int,
    llm_batch_size: int,
    progress: "Progress | None" = None,
) -> tuple[list[dict[str, Any]], list[ManualEntry]]:
    manual_raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = [] if force_rebuild or not manual_raw_path.exists() else list(_read_jsonl(manual_raw_path))
    rows_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing_rows:
        if row.get("stage") == "candidate":
            rows_by_skill[str(row["skill_id"])].append(row)
    accepted_counts = {
        skill_id: _accepted_count(skill_rows)
        for skill_id, skill_rows in rows_by_skill.items()
    }
    generated_skill_ids = {
        skill_id
        for skill_id, accepted_count in accepted_counts.items()
        if accepted_count >= proxy_queries_per_skill
    }
    pending_skill_ids = [
        skill_id
        for skill_id in selected_skill_ids
        if skill_id not in generated_skill_ids and skill_id in pool_by_id
    ]
    if manual_path.exists() and manual_raw_path.exists() and not force_rebuild and not pending_skill_ids:
        if progress:
            progress.message(f"reusing complete manual {manual_path}")
        return existing_rows, _rows_to_manual(existing_rows, proxy_queries_per_skill)

    rows = existing_rows[:]
    pool = list(pool_by_id.values())
    rng = random.Random(seed)
    if progress:
        progress.start("generating/verifying car manual", len(pending_skill_ids))

    with manual_raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if progress and pending_skill_ids:
            progress.message(
                f"car-specific LLM calls | workers={max(1, llm_workers)} "
                f"batch_size={max(1, llm_batch_size)} verify={verify_proxies}"
            )
        if verify_proxies or llm_workers <= 1:
            for skill_id in selected_skill_ids:
                if skill_id in generated_skill_ids or skill_id not in pool_by_id:
                    continue
                needed_count = proxy_queries_per_skill - accepted_counts.get(skill_id, 0)
                skill_rows = _generate_car_manual_rows_for_skill(
                    skill_id=skill_id,
                    pool_by_id=pool_by_id,
                    pool=pool,
                    example_queries=example_queries,
                    llm=llm,
                    proxy_queries_per_skill=needed_count,
                    verify_proxies=verify_proxies,
                    verifier_distractors=verifier_distractors,
                    rng=rng,
                )
                _append_rows(handle, rows, skill_rows)
                if progress:
                    progress.advance()
        else:
            workers = max(1, llm_workers)
            max_in_flight = workers * 4
            submitted = 0
            completed = 0
            last_heartbeat = time.monotonic()
            units = _skill_batches_by_needed_count(
                pending_skill_ids,
                accepted_counts,
                proxy_queries_per_skill,
                max(1, llm_batch_size),
            )
            iterator = iter(enumerate(units))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = set()

                def submit_next() -> bool:
                    nonlocal submitted
                    try:
                        index, (needed_count, next_skill_ids) = next(iterator)
                    except StopIteration:
                        return False
                    futures.add(
                        executor.submit(
                            _generate_car_manual_rows_for_skills,
                            skill_ids=next_skill_ids,
                            pool_by_id=pool_by_id,
                            pool=pool,
                            example_queries=example_queries,
                            llm=llm,
                            proxy_queries_per_skill=needed_count,
                            verify_proxies=False,
                            verifier_distractors=verifier_distractors,
                            rng=random.Random(seed + index),
                        )
                    )
                    submitted += len(next_skill_ids)
                    return True

                def heartbeat(force: bool = False) -> None:
                    nonlocal last_heartbeat
                    now = time.monotonic()
                    if not progress or (not force and now - last_heartbeat < 10):
                        return
                    last_heartbeat = now
                    progress.message(
                        f"LLM status | submitted={submitted}/{len(pending_skill_ids)} "
                        f"completed={completed}/{len(pending_skill_ids)} "
                        f"in_flight={len(futures)} raw_rows={len(rows)}"
                    )

                while len(futures) < max_in_flight and submit_next():
                    pass
                heartbeat(force=True)
                while futures:
                    try:
                        future = next(as_completed(futures, timeout=10))
                    except TimeoutError:
                        heartbeat(force=True)
                        continue
                    futures.remove(future)
                    skill_rows = future.result()
                    completed += len({str(row.get("skill_id")) for row in skill_rows})
                    _append_rows(handle, rows, skill_rows)
                    if progress:
                        progress.advance()
                    while len(futures) < max_in_flight and submit_next():
                        pass
                    heartbeat()

    manual = _rows_to_manual(rows, proxy_queries_per_skill)
    save_manual(manual_path, manual)
    if progress:
        progress.finish(f"manual entries={len(manual)} raw_rows={len(rows)}")
    return rows, manual


def generate_car_proxy_queries(
    llm: OpenAICompatibleLLM,
    tool: ToolSpec,
    count: int,
    examples: list[str] | None = None,
) -> list[dict[str, Any]]:
    example_block = ""
    if examples:
        example_block = (
            "\n\nExample labeled requests for style and intent guidance. Do not copy verbatim:\n"
            + json.dumps(examples, ensure_ascii=False, indent=2)
        )
    prompt = (
        f"{CAR_IMPLICIT_GUIDANCE}\n\n"
        "Return only a JSON array. Each object must contain query, scenario_type, generation_notes.\n"
        "Use scenario_type values such as direct_control_request, implicit_comfort_request, "
        "implicit_status_query, safety_request, visibility_request, or diagnostic_query.\n"
        f"Return exactly {count} objects.\n\n"
        f"Tool:\n{json.dumps(tool.to_dict(), ensure_ascii=False)}"
        f"{example_block}"
    )
    data = _extract_json(
        llm._chat(
            prompt,
            stage="car_query_generation",
            metadata={
                "skill_count": 1,
                "query_count": count,
                "skill_ids": [tool.name],
            },
        )
    )
    if not isinstance(data, list):
        raise ValueError(f"LLM did not return a JSON array for {tool.name}")
    out: list[dict[str, Any]] = []
    for item in data[:count]:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"query": item, "scenario_type": "realistic_vehicle_request", "generation_notes": "string_item"})
    if len(out) < count:
        raise ValueError(f"LLM returned {len(out)} usable proxy queries for {tool.name}, expected {count}")
    return out


def generate_car_proxy_queries_batch(
    llm: OpenAICompatibleLLM,
    tools: list[ToolSpec],
    count: int,
    examples_by_skill: dict[str, list[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    examples_by_skill = examples_by_skill or {}
    compact_tools = []
    for tool in tools:
        item = tool.to_dict()
        examples = examples_by_skill.get(tool.name, [])
        if examples:
            item["example_queries"] = examples
        compact_tools.append(item)
    prompt = (
        f"{CAR_IMPLICIT_GUIDANCE}\n\n"
        "Generate realistic proxy user queries for multiple vehicle tools.\n"
        "Return only one valid JSON object. The keys must be the exact tool names.\n"
        "Each value must be a JSON array of objects. Each object must contain query, scenario_type, generation_notes.\n"
        "Use scenario_type values such as direct_control_request, implicit_comfort_request, "
        "implicit_status_query, safety_request, visibility_request, or diagnostic_query.\n"
        f"Return exactly {count} objects per tool.\n\n"
        f"Tools:\n{json.dumps(compact_tools, ensure_ascii=False)}"
    )
    data = _extract_json(
        llm._chat(
            prompt,
            stage="car_query_generation_batch",
            metadata={
                "skill_count": len(tools),
                "query_count": len(tools) * count,
                "skill_ids": [tool.name for tool in tools],
            },
        )
    )
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object for batched car proxy generation")
    out: dict[str, list[dict[str, Any]]] = {}
    for tool in tools:
        raw_items = data.get(tool.name, [])
        if not isinstance(raw_items, list):
            raw_items = []
        items: list[dict[str, Any]] = []
        for item in raw_items[:count]:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, str):
                items.append({"query": item, "scenario_type": "realistic_vehicle_request", "generation_notes": "string_item"})
        out[tool.name] = items
    return out


def _generate_car_manual_rows_for_skills(
    skill_ids: list[str],
    pool_by_id: dict[str, ToolSpec],
    pool: list[ToolSpec],
    example_queries: dict[str, list[str]],
    llm: OpenAICompatibleLLM,
    proxy_queries_per_skill: int,
    verify_proxies: bool,
    verifier_distractors: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(skill_ids) == 1 or verify_proxies:
        rows: list[dict[str, Any]] = []
        for skill_id in skill_ids:
            rows.extend(
                _generate_car_manual_rows_for_skill(
                    skill_id=skill_id,
                    pool_by_id=pool_by_id,
                    pool=pool,
                    example_queries=example_queries,
                    llm=llm,
                    proxy_queries_per_skill=proxy_queries_per_skill,
                    verify_proxies=verify_proxies,
                    verifier_distractors=verifier_distractors,
                    rng=rng,
                )
            )
        return rows

    tools = [pool_by_id[skill_id] for skill_id in skill_ids]
    examples_by_skill = {skill_id: example_queries.get(skill_id, []) for skill_id in skill_ids}
    try:
        generated = generate_car_proxy_queries_batch(
            llm=llm,
            tools=tools,
            count=proxy_queries_per_skill,
            examples_by_skill=examples_by_skill,
        )
    except Exception as exc:
        rows: list[dict[str, Any]] = []
        for skill_id in skill_ids:
            rows.extend(_generation_error_rows(skill_id, examples_by_skill[skill_id], proxy_queries_per_skill, exc))
        return rows

    rows: list[dict[str, Any]] = []
    for skill_id in skill_ids:
        candidates = generated.get(skill_id, [])
        if len(candidates) < proxy_queries_per_skill:
            rows.extend(
                _generation_error_rows(
                    skill_id,
                    examples_by_skill[skill_id],
                    proxy_queries_per_skill,
                    ValueError(f"batch returned {len(candidates)} usable proxy queries"),
                )
            )
            continue
        rows.extend(
            _manual_rows_from_candidates(
                skill_id=skill_id,
                tool=pool_by_id[skill_id],
                pool=pool,
                candidates=candidates,
                examples_for_skill=examples_by_skill[skill_id],
                verify_proxies=False,
                verifier_distractors=verifier_distractors,
                llm=llm,
                rng=rng,
            )
        )
    return rows


def _generate_car_manual_rows_for_skill(
    skill_id: str,
    pool_by_id: dict[str, ToolSpec],
    pool: list[ToolSpec],
    example_queries: dict[str, list[str]],
    llm: OpenAICompatibleLLM,
    proxy_queries_per_skill: int,
    verify_proxies: bool,
    verifier_distractors: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    tool = pool_by_id[skill_id]
    examples_for_skill = example_queries.get(skill_id, [])
    try:
        candidates = generate_car_proxy_queries(
            llm,
            tool,
            proxy_queries_per_skill,
            examples=examples_for_skill,
        )
    except Exception as exc:
        return _generation_error_rows(skill_id, examples_for_skill, proxy_queries_per_skill, exc)
    return _manual_rows_from_candidates(
        skill_id=skill_id,
        tool=tool,
        pool=pool,
        candidates=candidates,
        examples_for_skill=examples_for_skill,
        verify_proxies=verify_proxies,
        verifier_distractors=verifier_distractors,
        llm=llm,
        rng=rng,
    )


def _manual_rows_from_candidates(
    skill_id: str,
    tool: ToolSpec,
    pool: list[ToolSpec],
    candidates: list[dict[str, Any]],
    examples_for_skill: list[str],
    verify_proxies: bool,
    verifier_distractors: int,
    llm: OpenAICompatibleLLM,
    rng: random.Random,
) -> list[dict[str, Any]]:
    skill_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        query = str(candidate.get("query", "")).strip()
        row = {
            "stage": "candidate",
            "skill_id": skill_id,
            "query_index": index,
            "query": query,
            "scenario_type": str(candidate.get("scenario_type", "realistic_vehicle_request")),
            "generation_notes": str(candidate.get("generation_notes", "")),
            "example_query_count": len(examples_for_skill),
            "example_queries": examples_for_skill,
            "accepted": False,
            "verifier_choice": None,
            "rejection_reason": None,
        }
        if not query:
            row["rejection_reason"] = "empty_query"
        elif verify_proxies:
            verification_pool = [tool, *_sample_distractors(pool, skill_id, verifier_distractors, rng)]
            rng.shuffle(verification_pool)
            verifier_choice = verify_proxy_query(llm, query, verification_pool)
            row["verifier_choice"] = verifier_choice
            row["accepted"] = verifier_choice == skill_id
            if not row["accepted"]:
                row["rejection_reason"] = "verifier_selected_different_skill"
        else:
            row["accepted"] = True
        skill_rows.append(row)
    return skill_rows


def _generation_error_rows(
    skill_id: str,
    examples_for_skill: list[str],
    count: int,
    exc: Exception,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": "candidate",
            "skill_id": skill_id,
            "query_index": index,
            "query": "",
            "scenario_type": "generation_error",
            "generation_notes": "",
            "example_query_count": len(examples_for_skill),
            "example_queries": examples_for_skill,
            "accepted": False,
            "verifier_choice": None,
            "rejection_reason": f"car_proxy_generation_failed: {type(exc).__name__}: {str(exc)[:1200]}",
        }
        for index in range(count)
    ]


def _accepted_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("accepted") is True and row.get("query"))


def _rows_to_manual(rows: list[dict[str, Any]], max_per_skill: int) -> list[ManualEntry]:
    accepted_by_skill: dict[str, int] = defaultdict(int)
    manual: list[ManualEntry] = []
    for row in rows:
        if row.get("accepted") is not True or not row.get("query"):
            continue
        skill_id = str(row["skill_id"])
        if accepted_by_skill[skill_id] >= max_per_skill:
            continue
        accepted_by_skill[skill_id] += 1
        manual.append(
            ManualEntry(
                query=str(row["query"]),
                tool_name=skill_id,
                source="CarSpecificOpenAICompatibleLLM",
                verification_score=1.0,
            )
        )
    return manual


def _append_rows(handle: Any, rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    for row in new_rows:
        rows.append(row)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _skill_batches(skill_ids: list[str], batch_size: int) -> list[list[str]]:
    return [skill_ids[index : index + batch_size] for index in range(0, len(skill_ids), batch_size)]


def _skill_batches_by_needed_count(
    skill_ids: list[str],
    accepted_counts: dict[str, int],
    target_count: int,
    batch_size: int,
) -> list[tuple[int, list[str]]]:
    ids_by_needed: dict[int, list[str]] = defaultdict(list)
    for skill_id in skill_ids:
        needed = target_count - accepted_counts.get(skill_id, 0)
        if needed > 0:
            ids_by_needed[needed].append(skill_id)
    batches: list[tuple[int, list[str]]] = []
    for needed, ids in sorted(ids_by_needed.items()):
        for batch in _skill_batches(ids, batch_size):
            batches.append((needed, batch))
    return batches


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_car_tool_specs(path: str | Path) -> list[ToolSpec]:
    return [ToolSpec.from_dict(tool) for tool in load_bfcl_tools_file(path)]


def write_generated_queries(path: str | Path, manual: list[ManualEntry]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in manual:
            handle.write(
                json.dumps(
                    {"tool_name": entry.tool_name, "query": entry.query},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _derive_path(out_path: Path, suffix: str) -> Path:
    name = out_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return out_path.with_name(name + suffix)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
