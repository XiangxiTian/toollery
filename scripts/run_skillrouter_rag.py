from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_skillrouter_toollery import (  # noqa: E402
    DEFAULT_CONFIG,
    TIER_NAMES,
    Progress,
    _format_path,
    _is_parser_default,
    _iter_jsonl,
    _load_jsonl,
    _resolve_data_root,
    evaluate_predictions,
)
from toollery.rag_baseline import EmbeddingConfig, make_rag_retriever  # noqa: E402
from toollery.schemas import ToolSpec  # noqa: E402


def main() -> None:
    args = parse_args()
    apply_config(args)
    if args.backend in {"raganything", "lightrag"}:
        _require_embedding_config(args)
    skillrouter_root = Path(args.skillrouter_root)
    data_root = _resolve_data_root(skillrouter_root, args.data_root)
    output_root = Path(args.output_dir)

    tasks = _load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}

    for tier in args.tiers:
        progress = Progress(f"{tier}/rag")
        progress.message("loading skill pool")
        pool = load_skillrouter_rag_pool(data_root / tier, limit=args.limit_pool)
        pool_ids = {tool.name for tool in pool}
        predictions_path = _format_path(args.predictions_out, tier, output_root)
        metrics_path = _format_path(args.metrics_out, tier, output_root)

        retriever = make_rag_retriever(
            args.backend,
            str(Path(args.rag_working_dir) / tier),
            embedding_config=args.embedding_config,
        )
        try:
            predictions = run_rag_retrieval(
                tasks=tasks,
                relevance=relevance,
                pool=pool,
                top_k=args.top_k,
                task_mode=args.task_mode,
                limit_tasks=args.limit_tasks,
                retriever=retriever,
                progress=progress,
            )
        finally:
            close = getattr(retriever, "close", None)
            if close is not None:
                close()
        _write_json(predictions_path, predictions)
        metrics = evaluate_predictions(
            tasks=tasks,
            relevance=relevance,
            predictions=predictions,
            pool_ids=pool_ids,
            task_mode=args.task_mode,
        )
        _write_json(metrics_path, metrics)
        summary[tier] = {
            "backend": args.backend,
            "pool_size": len(pool),
            "predicted_tasks": len(predictions),
            "predictions": str(predictions_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "metrics_summary": metrics,
        }

    if args.summary_out:
        summary_path = _format_path(args.summary_out, "summary", output_root)
        _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate name-description RAG on SkillRouter Eval Core.")
    parser.add_argument("--config")
    parser.add_argument("--skillrouter-root")
    parser.add_argument("--data-root", default="data/eval_core")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIER_NAMES), default=["easy", "hard"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--task-mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--limit-pool", type=int)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--backend", choices=("raganything", "tfidf", "lightrag"), default="raganything")
    parser.add_argument("--rag-working-dir", default=".raganything_skillrouter")
    parser.add_argument("--output-dir", default="outputs/rag_skillrouter")
    parser.add_argument("--predictions-out", default="retrieval/{tier}.json")
    parser.add_argument("--metrics-out", default="metrics/{tier}.json")
    parser.add_argument("--summary-out", default="summary.json")
    return parser.parse_args()


def _llm_config(config: dict[str, Any]) -> dict[str, Any]:
    for section in ("openai", "deepseek", "llm"):
        value = config.get(section)
        if isinstance(value, dict):
            return value
    return {}


def _embedding_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("embedding")
    if isinstance(value, dict):
        return value
    value = config.get("openai_embedding")
    if isinstance(value, dict):
        return value
    openai_value = config.get("openai")
    if isinstance(openai_value, dict):
        return openai_value
    return {}


def apply_config(args: argparse.Namespace) -> None:
    if not args.config and DEFAULT_CONFIG.exists():
        args.config = str(DEFAULT_CONFIG)
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        openai_config = _llm_config(config)
        embedding_config = _embedding_config(config)
        args.embedding_config = _resolve_embedding_config(embedding_config)
        api_key = openai_config.get("api_key")
        api_key_env = openai_config.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = str(api_key)
        for key, env_name in {
            "base_url": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
        }.items():
            value = openai_config.get(key)
            if value and not os.getenv(env_name):
                os.environ[env_name] = str(value)
        if openai_config.get("base_url") and not os.getenv("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = str(openai_config["base_url"])

        has_rag_config = "skillrouter_rag" in config
        values = config.get("skillrouter_rag", config.get("skillrouter_toollery", config))
        for key, value in values.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            if not has_rag_config and attr in {
                "backend",
                "rag_working_dir",
                "output_dir",
                "predictions_out",
                "metrics_out",
                "summary_out",
            }:
                continue
            current = getattr(args, attr)
            if current is None or _is_parser_default(attr, current) or _is_rag_parser_default(attr, current):
                setattr(args, attr, value)

    if not args.skillrouter_root:
        raise SystemExit("--skillrouter-root is required, either on the command line or in --config.")
    if not hasattr(args, "embedding_config"):
        args.embedding_config = _resolve_embedding_config({})


def _resolve_embedding_config(config: dict[str, Any]) -> EmbeddingConfig:
    api_key = _resolve_api_key(config)
    return EmbeddingConfig(
        backend=str(config.get("backend") or "openai-compatible"),
        api_key=str(api_key) if api_key else None,
        base_url=str(config.get("base_url") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
        model=str(config.get("model") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"),
        dim=int(config.get("dim") or config.get("embedding_dim") or 1536),
        timeout=int(config["timeout"]) if config.get("timeout") else None,
        device=str(config["device"]) if config.get("device") else None,
        batch_size=int(config.get("batch_size") or 8),
        max_length=int(config.get("max_length") or 512),
        pooling=str(config.get("pooling") or "last"),
        dtype=str(config["dtype"]) if config.get("dtype") else None,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=bool(config.get("local_files_only", False)),
    )


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
    return value.startswith(("sk-", "sk_", "ak-", "ak_")) or len(value) >= 32 and "-" in value


def _require_embedding_config(args: argparse.Namespace) -> None:
    config = args.embedding_config
    if not config.api_key:
        raise SystemExit(
            "RAG-Anything backend requires an embedding API key. "
            "Add an `embedding` section with `api_key` or `api_key_env` to the config."
        )


def run_rag_retrieval(
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    pool: list[Any],
    top_k: int,
    task_mode: str,
    limit_tasks: int | None,
    retriever: Any,
    progress: Progress | None = None,
) -> dict[str, list[str]]:
    from scripts.run_skillrouter_toollery import _include_task

    predictions: dict[str, list[str]] = {}
    included_tasks = [task for task in tasks if _include_task(task["task_id"], relevance, task_mode)]
    if limit_tasks is not None:
        included_tasks = included_tasks[:limit_tasks]
    if progress:
        progress.start("retrieving tasks with RAG", len(included_tasks))
    for task in included_tasks:
        task_id = str(task["task_id"])
        hits = retriever.retrieve(str(task["instruction_text"]), pool, min(top_k, len(pool)))
        predictions[task_id] = [hit.tool.name for hit in hits]
        if progress:
            progress.advance()
    if progress:
        progress.finish(f"predicted_tasks={len(predictions)}")
    return predictions


def load_skillrouter_rag_pool(path: Path, limit: int | None = None) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for item in _iter_jsonl(path):
        skill_id = str(item["skill_id"])
        display_name = str(item.get("name", skill_id))
        description = str(item.get("description", ""))
        text = "\n".join(part for part in [display_name, description] if part)
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


def _is_rag_parser_default(attr: str, value: Any) -> bool:
    defaults = {
        "backend": "raganything",
        "rag_working_dir": ".raganything_skillrouter",
        "output_dir": "outputs/rag_skillrouter",
        "predictions_out": "retrieval/{tier}.json",
        "metrics_out": "metrics/{tier}.json",
        "summary_out": "summary.json",
    }
    return attr in defaults and value == defaults[attr]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
