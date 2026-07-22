from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.embeddings import make_embedder_from_env
from toollery.io import load_manual, save_manual
from toollery.llm import OpenAICompatibleLLM
from toollery.retrieval import ProxyQueryIndex
from toollery.schemas import ManualEntry, ToolSpec


TIER_NAMES = {"easy", "hard"}
DEFAULT_CONFIG = Path("skillrouter_toollery_config.json")


def main() -> None:
    args = parse_args()
    cli_overrides = _cli_overrides(sys.argv[1:])
    args._cli_overrides = cli_overrides
    apply_config(args)
    apply_embedding_args(args, cli_overrides)
    skillrouter_root = Path(args.skillrouter_root)
    data_root = _resolve_data_root(skillrouter_root, args.data_root)
    output_root = Path(args.output_dir)

    tasks = _load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    validate_llm_environment()
    example_queries = build_example_queries(
        tasks=tasks,
        relevance=relevance,
        max_examples_per_skill=args.examples_per_skill,
    ) if args.example_queries_from_relevance else {}
    summary: dict[str, Any] = {}

    for tier in args.tiers:
        progress = Progress(f"{tier}")
        progress.message("loading skill pool")
        pool = load_skillrouter_pool(data_root / tier, limit=args.limit_pool, body_chars=args.skill_body_chars)
        pool_by_id = {tool.name: tool for tool in pool}
        manual_task_ids = [
            str(task["task_id"])
            for task in tasks
            if _include_task(str(task["task_id"]), relevance, args.task_mode)
        ]
        if args.limit_tasks is not None:
            manual_task_ids = manual_task_ids[: args.limit_tasks]
        selected_skill_ids = select_manual_skill_ids(
            relevance=relevance,
            pool_ids=set(pool_by_id),
            scope=args.manual_scope,
            limit=args.limit_skills,
            task_ids=manual_task_ids if args.manual_scope == "gt-related" else None,
        )
        progress.message(
            f"pool={len(pool)} manual_scope_skills={len(selected_skill_ids)} tasks={len(tasks)}"
        )
        manual_raw_path = _format_path(args.manual_raw_out, tier, output_root)
        manual_path = _format_path(args.manual_out, tier, output_root)
        predictions_path = _format_path(args.predictions_out, tier, output_root)
        metrics_path = _format_path(args.metrics_out, tier, output_root)
        embeddings_path = _format_path(args.manual_embeddings_out, tier, output_root)

        raw_rows, manual = build_or_load_manual(
            manual_raw_path=manual_raw_path,
            manual_path=manual_path,
            selected_skill_ids=selected_skill_ids,
            pool_by_id=pool_by_id,
            example_queries=example_queries,
            llm=OpenAICompatibleLLM(),
            proxy_queries_per_skill=args.proxy_queries_per_skill,
            verifier_distractors=args.verifier_distractors,
            verify_proxies=args.verify_proxies,
            force_rebuild=args.force_rebuild,
            seed=args.seed,
            llm_workers=args.llm_workers,
            llm_batch_size=args.llm_batch_size,
            progress=progress,
        )

        predictions = run_retrieval(
            tasks=tasks,
            relevance=relevance,
            pool=pool,
            manual=manual,
            top_k=args.top_k,
            proxy_top_k=args.proxy_top_k,
            task_mode=args.task_mode,
            limit_tasks=args.limit_tasks,
            embedder=make_embedder_from_env(),
            embedding_cache_path=embeddings_path,
            force_rebuild_embeddings=args.force_rebuild_embeddings,
            progress=progress,
        )
        _write_json(predictions_path, predictions)
        metrics = evaluate_predictions(
            tasks=tasks,
            relevance=relevance,
            predictions=predictions,
            pool_ids=set(pool_by_id),
            task_mode=args.task_mode,
        )
        _write_json(metrics_path, metrics)
        summary[tier] = {
            "pool_size": len(pool),
            "selected_manual_skills": len(selected_skill_ids),
            "raw_proxy_rows": len(raw_rows),
            "verified_manual_entries": len(manual),
            "predicted_tasks": len(predictions),
            "manual_raw": str(manual_raw_path.resolve()),
            "manual": str(manual_path.resolve()),
            "manual_embeddings": str(embeddings_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "metrics_summary": metrics,
        }

    if args.summary_out:
        summary_path = _format_path(args.summary_out, "summary", output_root)
        _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Toollery on SkillRouter Eval Core.")
    parser.add_argument("--config")
    parser.add_argument("--no-config", action="store_true", help="Do not load any JSON config file, including the default config.")
    parser.add_argument("--skillrouter-root")
    parser.add_argument("--data-root", default="data/eval_core")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIER_NAMES), default=["easy", "hard"])
    parser.add_argument("--manual-scope", choices=["gt-related", "full-pool", "shard"], default="gt-related")
    parser.add_argument("--proxy-queries-per-skill", type=int, default=3)
    parser.add_argument("--verify-proxies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--proxy-top-k", type=int, default=200)
    parser.add_argument("--task-mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--verifier-distractors", type=int, default=8)
    parser.add_argument("--example-queries-from-relevance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--examples-per-skill", type=int, default=3)
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--skill-body-chars", type=int, default=1800)
    parser.add_argument("--limit-pool", type=int)
    parser.add_argument("--limit-skills", type=int)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output-dir", default="outputs/toollery_skillrouter")
    parser.add_argument("--manual-raw-out", default="manual_raw_{tier}.jsonl")
    parser.add_argument("--manual-out", default="manual_verified_{tier}.json")
    parser.add_argument("--manual-embeddings-out", default="embeddings/manual_embeddings_{tier}.json")
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument(
        "--embedding-backend",
        choices=["tfidf", "openai-compatible", "local-hf"],
        help="Embedding backend for Toollery proxy-query retrieval.",
    )
    parser.add_argument("--embedding-model", help="Embedding model name for local-hf or OpenAI-compatible embeddings.")
    parser.add_argument("--embedding-model-path", help="Local embedding model directory path.")
    parser.add_argument("--embedding-device", help="Device for local-hf embeddings, e.g. cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--embedding-batch-size", type=int, help="Batch size for embedding generated or verified proxy queries.")
    parser.add_argument("--embedding-max-length", type=int, help="Max token length for local-hf embeddings.")
    parser.add_argument("--embedding-pooling", choices=["last", "mean", "cls"], help="Pooling strategy for local-hf embeddings.")
    parser.add_argument("--embedding-dtype", help="Optional dtype for local-hf loading, e.g. float16, bfloat16, float32, or auto.")
    parser.add_argument("--embedding-trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HuggingFace loaders.")
    parser.add_argument("--embedding-local-files-only", action="store_true", help="Prevent HuggingFace downloads for local-hf embeddings.")
    parser.add_argument(
        "--embedding-normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to L2-normalize local-hf embeddings.",
    )
    parser.add_argument("--embedding-text-prefix", help="Optional text prefix added before local-hf embedding.")
    parser.add_argument(
        "--embedding-implementation",
        choices=["auto", "sentence-transformers", "transformers"],
        help="Local-hf implementation preference.",
    )
    parser.add_argument("--predictions-out", default="retrieval/{tier}.json")
    parser.add_argument("--metrics-out", default="metrics/{tier}.json")
    parser.add_argument("--summary-out", default="summary.json")
    args = parser.parse_args()
    return args


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


def apply_config(args: argparse.Namespace) -> None:
    if args.no_config:
        args.config = None
        if not args.skillrouter_root:
            raise SystemExit("--skillrouter-root is required when --no-config is used.")
        return
    if not args.config and DEFAULT_CONFIG.exists():
        args.config = str(DEFAULT_CONFIG)
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        apply_llm_config(config)
        apply_embedding_config(config)

        values = config.get("skillrouter_toollery", config)
        for key, value in values.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            if attr in getattr(args, "_cli_overrides", set()):
                continue
            current = getattr(args, attr)
            if current is None or _is_parser_default(attr, current):
                setattr(args, attr, value)

    if not args.skillrouter_root:
        raise SystemExit("--skillrouter-root is required, either on the command line or in --config.")


def apply_llm_config(config: dict[str, Any]) -> None:
    llm_config = dict(config.get("openai", {}))
    if "deepseek" in config:
        llm_config.update(
            {
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-pro",
                **config["deepseek"],
            }
        )
    if "llm" in config:
        llm_config.update(config["llm"])

    provider = str(llm_config.get("provider", "")).lower()
    if provider == "deepseek":
        llm_config.setdefault("base_url", "https://api.deepseek.com")
        llm_config.setdefault("model", "deepseek-v4-pro")

    api_key = (
        llm_config.get("api_key")
        or llm_config.get("deepseek_api_key")
        or llm_config.get("DEEPSEEK_API_KEY")
    )
    api_key_env = llm_config.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env))

    for value, env_name in [
        (api_key, "LLM_API_KEY"),
        (llm_config.get("base_url"), "LLM_BASE_URL"),
        (llm_config.get("model"), "LLM_MODEL"),
        (llm_config.get("timeout"), "LLM_TIMEOUT"),
    ]:
        if value and not os.getenv(env_name):
            os.environ[env_name] = str(value)

    extra_body = llm_config.get("extra_body")
    if extra_body is not None and not os.getenv("LLM_EXTRA_BODY"):
        os.environ["LLM_EXTRA_BODY"] = json.dumps(extra_body, ensure_ascii=False)

    # Backward-compatible names for code paths that still read OPENAI_*.
    for source_name, target_name in [
        ("LLM_API_KEY", "OPENAI_API_KEY"),
        ("LLM_BASE_URL", "OPENAI_BASE_URL"),
        ("LLM_MODEL", "OPENAI_MODEL"),
    ]:
        if os.getenv(source_name) and not os.getenv(target_name):
            os.environ[target_name] = os.getenv(source_name, "")


def validate_llm_environment() -> None:
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "Toollery proxy generation requires an LLM API key. Set LLM_API_KEY "
        "or OPENAI_API_KEY, or run without --no-config using a config file "
        "that provides the LLM credentials."
    )


def apply_embedding_config(config: dict[str, Any]) -> None:
    embedding_config = config.get("embedding", {})
    if not embedding_config:
        return
    api_key = embedding_config.get("api_key")
    api_key_env = embedding_config.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env))
    for value, env_name in [
        (embedding_config.get("backend", "openai-compatible"), "EMBEDDING_BACKEND"),
        (api_key, "EMBEDDING_API_KEY"),
        (embedding_config.get("base_url"), "EMBEDDING_BASE_URL"),
        (embedding_config.get("model"), "EMBEDDING_MODEL"),
        (embedding_config.get("model_path"), "EMBEDDING_MODEL_PATH"),
        (embedding_config.get("dimensions", embedding_config.get("dim")), "EMBEDDING_DIMENSIONS"),
        (embedding_config.get("batch_size"), "EMBEDDING_BATCH_SIZE"),
        (embedding_config.get("timeout"), "EMBEDDING_TIMEOUT"),
        (embedding_config.get("max_retries"), "EMBEDDING_MAX_RETRIES"),
        (embedding_config.get("device"), "EMBEDDING_DEVICE"),
        (embedding_config.get("max_length"), "EMBEDDING_MAX_LENGTH"),
        (embedding_config.get("pooling"), "EMBEDDING_POOLING"),
        (embedding_config.get("normalize"), "EMBEDDING_NORMALIZE"),
        (embedding_config.get("dtype"), "EMBEDDING_DTYPE"),
        (embedding_config.get("trust_remote_code"), "EMBEDDING_TRUST_REMOTE_CODE"),
        (embedding_config.get("local_files_only"), "EMBEDDING_LOCAL_FILES_ONLY"),
        (embedding_config.get("text_prefix"), "EMBEDDING_TEXT_PREFIX"),
        (embedding_config.get("implementation"), "EMBEDDING_IMPLEMENTATION"),
    ]:
        if value is not None and not os.getenv(env_name):
            os.environ[env_name] = str(value)
    extra_body = embedding_config.get("extra_body")
    if extra_body is not None and not os.getenv("EMBEDDING_EXTRA_BODY"):
        os.environ["EMBEDDING_EXTRA_BODY"] = json.dumps(extra_body, ensure_ascii=False)


def apply_embedding_args(args: argparse.Namespace, cli_overrides: set[str]) -> None:
    """Apply command-line embedding options after config loading.

    CLI options intentionally override config/env values so a run command can
    switch Toollery retrieval to local-hf without creating a separate config.
    """

    for attr, env_name in [
        ("embedding_backend", "EMBEDDING_BACKEND"),
        ("embedding_model", "EMBEDDING_MODEL"),
        ("embedding_model_path", "EMBEDDING_MODEL_PATH"),
        ("embedding_device", "EMBEDDING_DEVICE"),
        ("embedding_batch_size", "EMBEDDING_BATCH_SIZE"),
        ("embedding_max_length", "EMBEDDING_MAX_LENGTH"),
        ("embedding_pooling", "EMBEDDING_POOLING"),
        ("embedding_dtype", "EMBEDDING_DTYPE"),
        ("embedding_trust_remote_code", "EMBEDDING_TRUST_REMOTE_CODE"),
        ("embedding_local_files_only", "EMBEDDING_LOCAL_FILES_ONLY"),
        ("embedding_normalize", "EMBEDDING_NORMALIZE"),
        ("embedding_text_prefix", "EMBEDDING_TEXT_PREFIX"),
        ("embedding_implementation", "EMBEDDING_IMPLEMENTATION"),
    ]:
        if attr not in cli_overrides:
            continue
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_name] = str(value)


def _is_parser_default(attr: str, value: Any) -> bool:
    defaults = {
        "data_root": "data/eval_core",
        "tiers": ["easy", "hard"],
        "manual_scope": "gt-related",
        "proxy_queries_per_skill": 3,
        "verify_proxies": True,
        "top_k": 50,
        "proxy_top_k": 200,
        "task_mode": "core",
        "verifier_distractors": 8,
        "example_queries_from_relevance": True,
        "examples_per_skill": 3,
        "llm_workers": 1,
        "llm_batch_size": 1,
        "skill_body_chars": 1800,
        "seed": 31,
        "force_rebuild": False,
        "output_dir": "outputs/toollery_skillrouter",
        "manual_raw_out": "manual_raw_{tier}.jsonl",
        "manual_out": "manual_verified_{tier}.json",
        "manual_embeddings_out": "embeddings/manual_embeddings_{tier}.json",
        "force_rebuild_embeddings": False,
        "predictions_out": "retrieval/{tier}.json",
        "metrics_out": "metrics/{tier}.json",
        "summary_out": "summary.json",
    }
    return attr in defaults and value == defaults[attr]


def load_skillrouter_pool(path: Path, limit: int | None = None, body_chars: int = 1800) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for item in _iter_jsonl(path):
        skill_id = str(item["skill_id"])
        display_name = str(item.get("name", skill_id))
        description = str(item.get("description", ""))
        body = str(item.get("body", ""))
        body_part = _compact_text(body, body_chars) if body_chars > 0 else ""
        text = "\n".join(part for part in [display_name, description, body_part] if part)
        tools.append(
            ToolSpec(
                name=skill_id,
                description=text,
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string", "default": skill_id},
                        "display_name": {"type": "string", "default": display_name},
                        "source": {"type": "string", "default": item.get("source", "")},
                    },
                },
                category="skillrouter",
            )
        )
        if limit is not None and len(tools) >= limit:
            break
    return tools


def select_manual_skill_ids(
    relevance: dict[str, Any],
    pool_ids: set[str],
    scope: str,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[str]:
    if scope in {"full-pool", "shard"}:
        selected = sorted(pool_ids)
    else:
        ids: set[str] = set()
        entries = [relevance[task_id] for task_id in task_ids or [] if task_id in relevance]
        if not entries:
            entries = list(relevance.values())
        for entry in entries:
            ids.update(entry.get("gt_skill_ids", []))
            ids.update(entry.get("core_gt_ids", []))
            ids.update(entry.get("relevance", {}).keys())
        selected = sorted(ids & pool_ids)
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_example_queries(
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    max_examples_per_skill: int = 3,
) -> dict[str, list[str]]:
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    examples: dict[str, list[str]] = defaultdict(list)
    for task_id, entry in relevance.items():
        task = tasks_by_id.get(str(task_id))
        if not task:
            continue
        query = str(task.get("instruction_text", "")).strip()
        if not query:
            continue
        skill_ids = set(entry.get("gt_skill_ids", []))
        skill_ids.update(entry.get("core_gt_ids", []))
        skill_ids.update(entry.get("relevance", {}).keys())
        for skill_id in skill_ids:
            bucket = examples[str(skill_id)]
            if len(bucket) < max_examples_per_skill:
                bucket.append(query)
    return examples


def build_or_load_manual(
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
    generated_skill_ids = {
        skill_id
        for skill_id, skill_rows in rows_by_skill.items()
        if _skill_generation_complete(skill_rows, proxy_queries_per_skill)
    }
    pending_skill_ids = [
        skill_id
        for skill_id in selected_skill_ids
        if skill_id not in generated_skill_ids and skill_id in pool_by_id
    ]
    if manual_path.exists() and manual_raw_path.exists() and not force_rebuild and not pending_skill_ids:
        if progress:
            progress.message(f"reusing complete manual {manual_path}")
        return existing_rows, load_manual(manual_path)
    if manual_path.exists() and manual_raw_path.exists() and not force_rebuild and pending_skill_ids and progress:
        progress.message(
            f"partial manual found; resuming {len(pending_skill_ids)} missing skills "
            f"({len(generated_skill_ids)} complete)"
        )
    rows = existing_rows[:]
    rng = random.Random(seed)
    pool = list(pool_by_id.values())
    if progress:
        progress.start("generating/verifying manual", len(pending_skill_ids))

    with manual_raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if progress and pending_skill_ids:
            progress.message(
                f"LLM calls are counted after skills return; workers={max(1, llm_workers)} "
                f"batch_size={max(1, llm_batch_size)}"
            )
        if verify_proxies or llm_workers <= 1:
            for skill_id in selected_skill_ids:
                if skill_id in generated_skill_ids or skill_id not in pool_by_id:
                    continue
                skill_rows = _generate_manual_rows_for_skill(
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
                _append_rows(handle, rows, skill_rows)
                if progress:
                    progress.advance()
        else:
            workers = max(1, llm_workers)
            max_in_flight = workers * 4
            submitted = 0
            completed = 0
            last_heartbeat = time.monotonic()
            units = _skill_batches(pending_skill_ids, max(1, llm_batch_size))
            iterator = iter(enumerate(units))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = set()

                def submit_next() -> bool:
                    nonlocal submitted
                    try:
                        index, next_skill_ids = next(iterator)
                    except StopIteration:
                        return False
                    futures.add(
                        executor.submit(
                            _generate_manual_rows_for_skills,
                            skill_ids=next_skill_ids,
                            pool_by_id=pool_by_id,
                            pool=pool,
                            example_queries=example_queries,
                            llm=llm,
                            proxy_queries_per_skill=proxy_queries_per_skill,
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

                try:
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
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    handle.flush()
                    partial_manual = _rows_to_manual(rows)
                    save_manual(manual_path, partial_manual)
                    if progress:
                        progress.finish(
                            f"interrupted; saved partial manual entries={len(partial_manual)} raw_rows={len(rows)}"
                        )
                    raise

    manual = _rows_to_manual(rows)
    save_manual(manual_path, manual)
    if progress:
        progress.finish(f"manual entries={len(manual)} raw_rows={len(rows)}")
    return rows, manual


def _skill_generation_complete(rows: list[dict[str, Any]], expected_count: int) -> bool:
    if len(rows) < expected_count:
        return False
    return any(row.get("accepted") is True and row.get("query") for row in rows)


def _append_rows(handle: Any, rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    for row in new_rows:
        rows.append(row)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _rows_to_manual(rows: list[dict[str, Any]]) -> list[ManualEntry]:
    return [
        ManualEntry(
            query=str(row["query"]),
            tool_name=str(row["skill_id"]),
            source="OpenAICompatibleLLM",
            verification_score=1.0,
        )
        for row in rows
        if row.get("accepted") is True and row.get("query")
    ]


def _skill_batches(skill_ids: list[str], batch_size: int) -> list[list[str]]:
    return [skill_ids[index : index + batch_size] for index in range(0, len(skill_ids), batch_size)]


def _generate_manual_rows_for_skills(
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
                _generate_manual_rows_for_skill(
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
        generated = generate_proxy_queries_batch(
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


def _generate_manual_rows_for_skill(
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
        candidates = generate_proxy_queries(
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
            "rejection_reason": f"proxy_generation_failed: {type(exc).__name__}: {str(exc)[:1200]}",
        }
        for index in range(count)
    ]


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
            "scenario_type": str(candidate.get("scenario_type", "realistic_task")),
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


def generate_proxy_queries(
    llm: OpenAICompatibleLLM,
    tool: ToolSpec,
    count: int,
    examples: list[str] | None = None,
) -> list[dict[str, Any]]:
    example_block = ""
    if examples:
        example_block = (
            "\n\nExample benchmark tasks that were labeled relevant for this skill. "
            "Use them only as style and intent guidance; do not copy them verbatim:\n"
            + json.dumps(examples, ensure_ascii=False, indent=2)
        )
    prompt = (
        "Generate realistic proxy user queries for a skill-routing benchmark.\n"
        "The queries should be things real users might ask when they need this exact skill.\n"
        "Use practical task context, constraints, and natural wording. Avoid copying the skill text.\n"
        "If examples are provided, make the generated queries similarly realistic but shorter and not duplicated.\n"
        "Return only a JSON array. Each object must contain query, scenario_type, generation_notes.\n"
        f"Return exactly {count} objects.\n\n"
        f"Skill:\n{json.dumps(tool.to_dict(), ensure_ascii=False)}"
        f"{example_block}"
    )
    data = _extract_json(
        llm._chat(
            prompt,
            stage="query_generation",
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
            out.append({"query": item, "scenario_type": "realistic_task", "generation_notes": "string_item"})
    if len(out) < count:
        raise ValueError(f"LLM returned {len(out)} usable proxy queries for {tool.name}, expected {count}")
    return out


def generate_proxy_queries_batch(
    llm: OpenAICompatibleLLM,
    tools: list[ToolSpec],
    count: int,
    examples_by_skill: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    compact_tools = []
    for tool in tools:
        item = tool.to_dict()
        examples = examples_by_skill.get(tool.name, [])
        if examples:
            item["example_queries"] = examples
        compact_tools.append(item)
    prompt = (
        "Generate realistic proxy user queries for multiple skills in a skill-routing benchmark.\n"
        "For each skill, write things real users might ask when they need that exact skill.\n"
        "Use practical task context, constraints, and natural wording. Avoid copying skill text verbatim.\n"
        "Return only one valid JSON object. The keys must be the exact skill names.\n"
        "Each value must be a JSON array of objects. Each object must contain query, scenario_type, generation_notes.\n"
        f"Return exactly {count} objects per skill.\n\n"
        f"Skills:\n{json.dumps(compact_tools, ensure_ascii=False)}"
    )
    data = _extract_json(
        llm._chat(
            prompt,
            stage="query_generation_batch",
            metadata={
                "skill_count": len(tools),
                "query_count": len(tools) * count,
                "skill_ids": [tool.name for tool in tools],
            },
        )
    )
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object for batched proxy generation")
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
                items.append({"query": item, "scenario_type": "realistic_task", "generation_notes": "string_item"})
        out[tool.name] = items
    return out


def verify_proxy_query(llm: OpenAICompatibleLLM, query: str, tools: list[ToolSpec]) -> str | None:
    names = {tool.name for tool in tools}
    prompt = (
        "Choose exactly one SkillRouter skill id that best matches the user request.\n"
        "Return only the skill_id, or NONE if no skill fits.\n\n"
        f"User request:\n{query}\n\n"
        f"Candidate skills:\n{json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False)}"
    )
    answer = llm._chat(
        prompt,
        stage="verification",
        metadata={
            "skill_count": len(tools),
            "query_count": 1,
            "candidate_skill_ids": [tool.name for tool in tools],
        },
    ).strip().strip('"').strip("'")
    if answer in names:
        return answer
    for name in names:
        if name in answer:
            return name
    match = re.search(r"[A-Za-z0-9_./-]+", answer)
    if match and match.group(0) in names:
        return match.group(0)
    return None


def run_retrieval(
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    pool: list[ToolSpec],
    manual: list[ManualEntry],
    top_k: int,
    proxy_top_k: int,
    task_mode: str,
    limit_tasks: int | None,
    embedder: Any | None = None,
    embedding_cache_path: Path | None = None,
    force_rebuild_embeddings: bool = False,
    progress: "Progress | None" = None,
) -> dict[str, list[str]]:
    if not manual:
        raise ValueError("Verified manual is empty; cannot run Toollery retrieval.")
    if progress:
        progress.message(f"building proxy index with embedder={type(embedder).__name__ if embedder else 'default'}")
        if embedding_cache_path:
            progress.message(f"manual embedding cache: {embedding_cache_path}")

    last_embedding_progress = 0
    embedding_progress_started = False

    def embedding_progress(done: int, total: int) -> None:
        nonlocal last_embedding_progress, embedding_progress_started
        if not progress or total <= 0:
            return
        if not embedding_progress_started:
            progress.start("embedding manual queries", total)
            embedding_progress_started = True
        if done > last_embedding_progress:
            progress.advance(done - last_embedding_progress)
            last_embedding_progress = done

    index = ProxyQueryIndex(
        pool,
        manual,
        embedder=embedder,
        embedding_cache_path=embedding_cache_path,
        force_rebuild_embeddings=force_rebuild_embeddings,
        embedding_progress_callback=embedding_progress,
    )
    if progress and embedding_progress_started:
        progress.finish("embedding cache ready")
    predictions: dict[str, list[str]] = {}
    count = 0
    included_tasks = [task for task in tasks if _include_task(task["task_id"], relevance, task_mode)]
    if limit_tasks is not None:
        included_tasks = included_tasks[:limit_tasks]
    if progress:
        progress.start("retrieving tasks", len(included_tasks))
    for task in included_tasks:
        task_id = task["task_id"]
        candidates = index.retrieve_tools(
            str(task["instruction_text"]),
            tool_top_k=top_k,
            proxy_top_k=max(proxy_top_k, top_k),
        )
        predictions[task_id] = [candidate.tool.name for candidate in candidates]
        count += 1
        if progress:
            progress.advance()
    if progress:
        progress.finish(f"predicted_tasks={count}")
    return predictions


def evaluate_predictions(
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    predictions: dict[str, list[str]],
    pool_ids: set[str],
    task_mode: str,
) -> dict[str, Any]:
    results_by_stratum: dict[str, list[dict[str, float]]] = {"all": [], "single": [], "multi": []}
    for task in tasks:
        task_id = task["task_id"]
        rel_entry = relevance.get(task_id, {})
        if task_mode == "core":
            if rel_entry.get("task_type") == "generic_only":
                continue
            gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
        elif task_mode == "single":
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
            if len(gt_ids) != 1:
                continue
        else:
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
        gt_ids_in_pool = gt_ids & pool_ids
        if not gt_ids_in_pool or task_id not in predictions:
            continue
        ranked_ids = predictions[task_id]
        tier_relevance = {
            key: float(value)
            for key, value in rel_entry.get("relevance", {}).items()
            if key in pool_ids
        }
        metrics = compute_all_metrics(ranked_ids, gt_ids_in_pool, tier_relevance or None)
        results_by_stratum["all"].append(metrics)
        if len(gt_ids) == 1:
            results_by_stratum["single"].append(metrics)
        else:
            results_by_stratum["multi"].append(metrics)
    return {key: aggregate_metrics(value) for key, value in results_by_stratum.items() if value}


def compute_all_metrics(
    ranked_ids: list[str],
    gt_skill_ids: set[str],
    relevance_map: dict[str, float] | None = None,
) -> dict[str, float]:
    if relevance_map:
        relevances = [float(relevance_map.get(rid, 0.0)) for rid in ranked_ids]
        all_relevance_values = list(relevance_map.values())
    else:
        relevances = [1.0 if rid in gt_skill_ids else 0.0 for rid in ranked_ids]
        all_relevance_values = [1.0] * len(gt_skill_ids) + [0.0] * max(0, len(ranked_ids) - len(gt_skill_ids))
    return {
        "nDCG@1": ndcg_at_k(relevances, all_relevance_values, 1),
        "nDCG@3": ndcg_at_k(relevances, all_relevance_values, 3),
        "nDCG@10": ndcg_at_k(relevances, all_relevance_values, 10),
        "Hit@1": hit_at_k(ranked_ids, gt_skill_ids, 1),
        "Precision@3": precision_at_k(ranked_ids, gt_skill_ids, 3),
        "MRR@10": mrr_at_k(ranked_ids, gt_skill_ids, 10),
        "Recall@10": recall_at_k(ranked_ids, gt_skill_ids, 10),
        "Recall@20": recall_at_k(ranked_ids, gt_skill_ids, 20),
        "Recall@50": recall_at_k(ranked_ids, gt_skill_ids, 50),
        "FullCoverage@3": full_coverage_at_k(ranked_ids, gt_skill_ids, 3),
        "FullCoverage@5": full_coverage_at_k(ranked_ids, gt_skill_ids, 5),
        "FullCoverage@10": full_coverage_at_k(ranked_ids, gt_skill_ids, 10),
    }


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    out = {
        key: sum(metrics[key] for metrics in metrics_list) / len(metrics_list)
        for key in metrics_list[0]
    }
    out["count"] = len(metrics_list)
    return out


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for index, ranked_id in enumerate(ranked_ids[:k]):
        if ranked_id in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(ranked_id in relevant_ids for ranked_id in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return sum(1 for ranked_id in ranked_ids[:k] if ranked_id in relevant_ids) / k if k > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def full_coverage_at_k(ranked_ids: list[str], required_ids: set[str], k: int) -> float:
    return 1.0 if required_ids.issubset(set(ranked_ids[:k])) else 0.0


def _include_task(task_id: str, relevance: dict[str, Any], task_mode: str) -> bool:
    rel_entry = relevance.get(task_id, {})
    if task_mode == "core":
        return rel_entry.get("task_type") != "generic_only"
    if task_mode == "single":
        return len(rel_entry.get("gt_skill_ids", [])) == 1
    return True


def _resolve_data_root(skillrouter_root: Path, data_root: str) -> Path:
    path = Path(data_root)
    return path if path.is_absolute() else skillrouter_root / path


def _format_path(template: str, tier: str, output_root: Path) -> Path:
    path = Path(template.format(tier=tier))
    return path if path.is_absolute() else output_root / path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    paths = _jsonl_paths(path)
    for item_path in paths:
        opener = gzip.open if item_path.name.endswith(".gz") else open
        with opener(item_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _jsonl_paths(path: Path) -> list[Path]:
    if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
        return [path]
    if path.is_dir():
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and (item.name.endswith(".jsonl") or item.name.endswith(".jsonl.gz"))
        )
    raise FileNotFoundError(f"Path not found: {path}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start_array, end_array = text.find("["), text.rfind("]")
        if start_array != -1 and end_array != -1:
            try:
                return json.loads(text[start_array : end_array + 1])
            except json.JSONDecodeError as inner_exc:
                snippet = text[start_array : min(end_array + 1, start_array + 1000)]
                raise ValueError(f"Invalid JSON array in LLM response: {inner_exc}; snippet={snippet!r}") from inner_exc
        start_object, end_object = text.find("{"), text.rfind("}")
        if start_object != -1 and end_object != -1:
            try:
                return json.loads(text[start_object : end_object + 1])
            except json.JSONDecodeError as inner_exc:
                snippet = text[start_object : min(end_object + 1, start_object + 1000)]
                raise ValueError(f"Invalid JSON object in LLM response: {inner_exc}; snippet={snippet!r}") from inner_exc
        snippet = text[:1000]
        raise ValueError(f"No JSON payload found in LLM response: {exc}; snippet={snippet!r}") from exc


def _compact_text(text: str, limit: int) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped)
        if len("\n".join(lines)) >= limit:
            break
    return "\n".join(lines)[:limit]


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


class Progress:
    def __init__(self, prefix: str, width: int = 28) -> None:
        self.prefix = prefix
        self.width = width
        self.label = ""
        self.total = 0
        self.current = 0

    def message(self, text: str) -> None:
        print(f"[{self.prefix}] {text}", file=sys.stderr, flush=True)

    def start(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(total, 0)
        self.current = 0
        self._render()

    def advance(self, step: int = 1) -> None:
        self.current += step
        if self.total <= 0 or self.current == self.total or self.current % max(1, self.total // 100) == 0:
            self._render()

    def finish(self, suffix: str = "") -> None:
        if self.total:
            self.current = self.total
        self._render(final=True, suffix=suffix)

    def _render(self, final: bool = False, suffix: str = "") -> None:
        if self.total <= 0:
            line = f"[{self.prefix}] {self.label}: 0/0"
        else:
            ratio = min(max(self.current / self.total, 0.0), 1.0)
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            line = f"[{self.prefix}] {self.label}: [{bar}] {self.current}/{self.total} {ratio * 100:5.1f}%"
        if suffix:
            line += f" | {suffix}"
        end = "\n" if final else "\r"
        print(line, file=sys.stderr, end=end, flush=True)


if __name__ == "__main__":
    main()
