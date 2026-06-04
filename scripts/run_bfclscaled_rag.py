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

from scripts.run_skillrouter_toollery import Progress, _cli_overrides, _format_path, _write_json  # noqa: E402
from toollery.bfcl import (  # noqa: E402
    BFCLScalePrediction,
    load_bfcl_scaled_dataset,
    save_bfcl_scale_predictions,
    summarize_scale_predictions,
)
from toollery.rag_baseline import (  # noqa: E402
    EmbeddingConfig,
    compare_scale_predictions,
    load_bfcl_scale_predictions,
    make_rag_retriever,
    save_comparison,
)


DEFAULT_CONFIG = Path("bfclscaled_toollery_config.json")


def main() -> None:
    args = parse_args()
    args._cli_overrides = _cli_overrides(sys.argv[1:])
    apply_config(args)
    if args.backend in {"raganything", "lightrag"}:
        _require_embedding_config(args)

    output_root = Path(args.output_dir)
    predictions_path = _format_path(args.predictions_out, "bfclscaled", output_root)
    metrics_path = _format_path(args.metrics_out, "bfclscaled", output_root)
    summary_path = _format_path(args.summary_out, "summary", output_root) if args.summary_out else None
    compare_path = _format_path(args.compare_out, "bfclscaled", output_root) if args.compare_out else None

    progress = Progress("bfclscaled/rag")
    progress.message("loading scaled BFCL data")
    scaled_samples = load_bfcl_scaled_dataset(args.scaled_data)
    if args.limit_samples is not None:
        scaled_samples = scaled_samples[: args.limit_samples]

    retriever = make_rag_retriever(
        args.backend,
        args.rag_working_dir,
        embedding_config=args.embedding_config,
    )
    try:
        predictions = run_bfclscaled_rag_retrieval(
            scaled_samples=scaled_samples,
            top_k=args.top_k,
            retriever=retriever,
            progress=progress,
        )
    finally:
        close = getattr(retriever, "close", None)
        if close is not None:
            close()

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    save_bfcl_scale_predictions(predictions_path, predictions)
    metrics = summarize_scale_predictions(predictions)
    _write_json(metrics_path, metrics)

    summary: dict[str, Any] = {
        "backend": args.backend,
        "metric": "top_k_hit",
        "top_k": args.top_k,
        "scaled_samples": len(scaled_samples),
        "predicted_samples": len(predictions),
        "scaled_data": str(Path(args.scaled_data).resolve()),
        "predictions": str(predictions_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "metrics_summary": metrics,
    }

    if args.compare_to:
        comparison = compare_scale_predictions(
            load_bfcl_scale_predictions(args.compare_to),
            predictions,
        )
        if compare_path is not None:
            compare_path.parent.mkdir(parents=True, exist_ok=True)
            save_comparison(compare_path, comparison)
            summary["comparison"] = str(compare_path.resolve())
        summary["comparison_rows"] = len(comparison)
        summary["compare_to"] = str(Path(args.compare_to).resolve())

    if summary_path is not None:
        _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate name-description RAG on BFCL scaled tool pools.")
    parser.add_argument("--config")
    parser.add_argument("--scaled-data")
    parser.add_argument("--compare-to")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=("raganything", "tfidf", "lightrag"), default="raganything")
    parser.add_argument("--rag-working-dir", default=".raganything_bfclscaled")
    parser.add_argument("--output-dir", default="outputs/rag_bfclscaled")
    parser.add_argument("--predictions-out", default="retrieval/bfclscaled_rag_predictions.jsonl")
    parser.add_argument("--metrics-out", default="metrics/bfclscaled_rag_metrics.json")
    parser.add_argument("--compare-out", default="comparison/bfclscaled_toollery_vs_rag.jsonl")
    parser.add_argument("--summary-out", default="summary.json")
    parser.add_argument("--limit-samples", type=int)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    if not args.config and DEFAULT_CONFIG.exists():
        args.config = str(DEFAULT_CONFIG)
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        openai_config = _llm_config(config)
        embedding_config = _embedding_config(config)
        args.embedding_config = _resolve_embedding_config(embedding_config)
        _apply_llm_env(openai_config)

        has_rag_config = "bfclscaled_rag" in config
        values = config.get("bfclscaled_rag", config.get("bfclscaled_toollery", config))
        for key, value in values.items():
            attr = key.replace("-", "_")
            if not hasattr(args, attr):
                continue
            if attr in getattr(args, "_cli_overrides", set()):
                continue
            if not has_rag_config and attr in {
                "backend",
                "rag_working_dir",
                "output_dir",
                "predictions_out",
                "metrics_out",
                "compare_out",
                "summary_out",
            }:
                continue
            current = getattr(args, attr)
            if current is None or _is_rag_parser_default(attr, current):
                setattr(args, attr, value)

    if not args.scaled_data:
        raise SystemExit("--scaled-data is required, either on the command line or in --config.")
    if not hasattr(args, "embedding_config"):
        args.embedding_config = _resolve_embedding_config({})


def run_bfclscaled_rag_retrieval(
    *,
    scaled_samples: list[Any],
    top_k: int,
    retriever: Any,
    progress: Progress | None = None,
) -> list[BFCLScalePrediction]:
    global_pool = _unique_tools(scaled_samples)
    global_top_k = len(global_pool)
    if progress:
        progress.message(f"global_pool={len(global_pool)}; filtering each query to its scaled candidate pool")
    predictions: list[BFCLScalePrediction] = []
    if progress:
        progress.start("retrieving scaled samples with RAG", len(scaled_samples))
    for sample in scaled_samples:
        hits = retriever.retrieve(sample.query, global_pool, global_top_k)
        candidate_names = {tool.name for tool in sample.tools}
        retrieved_tools: list[str] = []
        for hit in hits:
            if hit.tool.name in candidate_names:
                retrieved_tools.append(hit.tool.name)
                if len(retrieved_tools) >= min(top_k, len(sample.tools)):
                    break
        predicted_tool = retrieved_tools[0] if retrieved_tools else (sample.tools[0].name if sample.tools else "")
        predictions.append(
            BFCLScalePrediction(
                candidate_size=sample.candidate_size,
                sample_id=sample.sample_id,
                query=sample.query,
                predicted_tool=predicted_tool,
                retrieved_tools=retrieved_tools,
                scale_candidate_pool=[tool.name for tool in sample.tools],
                correct_tools=sample.correct_tools,
                is_correct=bool(set(retrieved_tools) & set(sample.correct_tools)) if sample.correct_tools else None,
            )
        )
        if progress:
            progress.advance()
    if progress:
        progress.finish(f"predicted_samples={len(predictions)}")
    return predictions


def _unique_tools(scaled_samples: list[Any]) -> list[Any]:
    tools_by_name: dict[str, Any] = {}
    for sample in scaled_samples:
        for tool in sample.tools:
            tools_by_name.setdefault(tool.name, tool)
    return list(tools_by_name.values())


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


def _apply_llm_env(config: dict[str, Any]) -> None:
    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env))
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)
    for key, env_name in {
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
    }.items():
        value = config.get(key)
        if value and not os.getenv(env_name):
            os.environ[env_name] = str(value)
    if config.get("base_url") and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = str(config["base_url"])


def _require_embedding_config(args: argparse.Namespace) -> None:
    config = args.embedding_config
    if not config.api_key:
        raise SystemExit(
            "RAG-Anything backend requires an embedding API key. "
            "Add an `embedding` section with `api_key` or `api_key_env` to the config."
        )


def _is_rag_parser_default(attr: str, value: Any) -> bool:
    defaults = {
        "top_k": 3,
        "backend": "raganything",
        "rag_working_dir": ".raganything_bfclscaled",
        "output_dir": "outputs/rag_bfclscaled",
        "predictions_out": "retrieval/bfclscaled_rag_predictions.jsonl",
        "metrics_out": "metrics/bfclscaled_rag_metrics.json",
        "compare_out": "comparison/bfclscaled_toollery_vs_rag.jsonl",
        "summary_out": "summary.json",
        "limit_samples": None,
    }
    return attr in defaults and value == defaults[attr]


if __name__ == "__main__":
    main()
