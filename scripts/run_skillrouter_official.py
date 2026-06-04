from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_skillrouter_toollery import (  # noqa: E402
    TIER_NAMES,
    _format_path,
    _resolve_data_root,
    evaluate_predictions,
)
from toollery.baselines import ensure_safe_output_path, load_generated_queries, write_json  # noqa: E402


def main() -> None:
    args = parse_args()
    ensure_safe_output_path(args.output_dir)
    skillrouter_root = Path(args.skillrouter_root)
    data_root = _resolve_data_root(skillrouter_root, args.data_root)
    output_root = Path(args.output_dir)
    method_setting = "query_augmented_baseline" if args.query_augmented else "raw_baseline"
    method_name = "skillrouter_query_augmented" if args.query_augmented else "skillrouter_raw"

    official_data_root = data_root
    if args.query_augmented:
        official_data_root = materialize_query_augmented_data(
            data_root=data_root,
            output_root=output_root / "query_augmented_data",
            tiers=args.tiers,
            generated_queries=load_generated_queries(args.generated_queries),
        )

    predictions_root = output_root / method_name
    retrieval_dir = predictions_root / "retrieval"
    if args.run_export:
        if not args.encoder_model_or_path:
            raise SystemExit("--encoder-model-or-path is required when --run-export is set.")
        cmd = [
            sys.executable,
            "-m",
            "src.export_retrieval",
            "--encoder_model_or_path",
            args.encoder_model_or_path,
            "--data_root",
            str(official_data_root),
            "--output_dir",
            str(predictions_root),
            "--top_k",
            str(args.top_k),
            "--tiers",
            *args.tiers,
        ]
        subprocess.run(cmd, cwd=skillrouter_root, check=True)

    tasks = list(_iter_jsonl(official_data_root / "tasks.jsonl"))
    relevance = json.loads((official_data_root / "relevance.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    for tier in args.tiers:
        predictions_path = retrieval_dir / f"{tier}.json"
        if not predictions_path.exists():
            raise SystemExit(f"Missing official SkillRouter predictions: {predictions_path}")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        pool_ids = {str(item["skill_id"]) for item in _iter_jsonl(official_data_root / tier)}
        metrics = evaluate_predictions(
            tasks=tasks,
            relevance=relevance,
            predictions=predictions,
            pool_ids=pool_ids,
            task_mode=args.task_mode,
        )
        payload = {
            "method_setting": method_setting,
            "method_name": method_name,
            "benchmark": "skillrouter",
            "tier": tier,
            "official": True,
            "predictions": str(predictions_path.resolve()),
            "metrics": metrics,
        }
        metrics_path = _format_path(args.metrics_out, tier, predictions_root)
        write_json(metrics_path, payload)
        summary[tier] = payload
    write_json(predictions_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap official SkillRouter retrieval outputs in Toollery experiment format.")
    parser.add_argument("--skillrouter-root", required=True)
    parser.add_argument("--data-root", default="data/eval_core")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIER_NAMES), default=["easy", "hard"])
    parser.add_argument("--output-dir", default="outputs/experiments_v2/skillrouter/official")
    parser.add_argument("--task-mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--metrics-out", default="metrics/{tier}.json")
    parser.add_argument("--run-export", action="store_true")
    parser.add_argument("--encoder-model-or-path")
    parser.add_argument("--query-augmented", action="store_true")
    parser.add_argument("--generated-queries")
    return parser.parse_args()


def materialize_query_augmented_data(
    *,
    data_root: Path,
    output_root: Path,
    tiers: list[str],
    generated_queries: dict[str, list[str]],
) -> Path:
    ensure_safe_output_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_root / "tasks.jsonl", output_root / "tasks.jsonl")
    shutil.copyfile(data_root / "relevance.json", output_root / "relevance.json")
    for tier in tiers:
        tier_out = output_root / tier
        with tier_out.open("w", encoding="utf-8") as handle:
            for item in _iter_jsonl(data_root / tier):
                queries = generated_queries.get(str(item.get("skill_id", "")), [])
                if queries:
                    body = str(item.get("body", ""))
                    item["body"] = body + "\n\nGenerated user queries:\n" + "\n".join(f"- {query}" for query in queries)
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_root


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    paths = _jsonl_paths(path)
    for item_path in paths:
        opener = gzip.open if item_path.name.endswith(".gz") else open
        with opener(item_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


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


if __name__ == "__main__":
    main()

