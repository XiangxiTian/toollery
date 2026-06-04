from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_skillrouter_toollery import (  # noqa: E402
    Progress,
    apply_llm_config,
    build_or_load_manual,
    validate_llm_environment,
)
from toollery.bfcl import BFCLSample, load_bfcl_answers, load_bfcl_samples, unique_tools  # noqa: E402
from toollery.llm import OpenAICompatibleLLM  # noqa: E402
from toollery.schemas import ManualEntry  # noqa: E402


def main() -> None:
    args = parse_args()
    apply_config(args)
    validate_llm_environment()

    samples = load_bfcl_samples(args.data)
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]
    tools = unique_tools(samples)
    if args.limit_tools is not None:
        tools = tools[: args.limit_tools]
    tools_by_name = {tool.name: tool for tool in tools}
    tool_names = sorted(tools_by_name)

    answers = load_bfcl_answers(args.answers) if args.answers else {}
    examples = build_example_queries(
        samples=samples,
        answers=answers,
        max_examples_per_tool=args.examples_per_tool,
    )

    out_path = Path(args.out)
    manual_raw_path = Path(args.manual_raw_out) if args.manual_raw_out else _derive_path(out_path, ".raw.jsonl")
    manual_path = Path(args.manual_out) if args.manual_out else _derive_path(out_path, ".manual.json")

    progress = Progress("bfcl-generated")
    progress.message(f"tools={len(tool_names)} samples={len(samples)} examples_for_tools={len(examples)}")
    raw_rows, manual = build_or_load_manual(
        manual_raw_path=manual_raw_path,
        manual_path=manual_path,
        selected_skill_ids=tool_names,
        pool_by_id=tools_by_name,
        example_queries=examples,
        llm=OpenAICompatibleLLM(),
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
        "data": str(Path(args.data).resolve()),
        "answers": str(Path(args.answers).resolve()) if args.answers else None,
        "out": str(out_path.resolve()),
        "manual_raw": str(manual_raw_path.resolve()),
        "manual": str(manual_path.resolve()),
        "tools": len(tool_names),
        "samples": len(samples),
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
        description="Generate normalized BFCL tool-level proxy queries with the Toollery LLM prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="skillrouter_toollery_deepseek_config.example.json")
    parser.add_argument("--data", required=True, help="BFCL JSONL data file, e.g. BFCL_v4_multiple.json.")
    parser.add_argument("--answers", help="Optional BFCL possible_answer JSONL file for example-query conditioning.")
    parser.add_argument("--out", required=True, help="Normalized JSONL output: one {tool_name, query} row per generated query.")
    parser.add_argument("--manual-raw-out", help="Raw candidate rows for resumable generation.")
    parser.add_argument("--manual-out", help="Verified/accepted manual JSON for resumable generation.")
    parser.add_argument("--summary-out", help="Optional summary JSON path.")
    parser.add_argument("--proxy-queries-per-tool", type=int, default=3)
    parser.add_argument("--verify-proxies", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verifier-distractors", type=int, default=8)
    parser.add_argument("--examples-per-tool", type=int, default=3)
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--limit-tools", type=int)
    parser.add_argument("--limit-samples", type=int)
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
    values = config.get("bfcl_generated_queries", {})
    for key, value in values.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) in (None, parse_args_default(attr)):
            setattr(args, attr, value)


def parse_args_default(attr: str) -> Any:
    defaults = {
        "proxy_queries_per_tool": 3,
        "verify_proxies": False,
        "verifier_distractors": 8,
        "examples_per_tool": 3,
        "llm_workers": 4,
        "llm_batch_size": 4,
        "seed": 31,
        "force_rebuild": False,
    }
    return defaults.get(attr)


def build_example_queries(
    samples: list[BFCLSample],
    answers: dict[str, list[str]],
    max_examples_per_tool: int,
) -> dict[str, list[str]]:
    if max_examples_per_tool <= 0 or not answers:
        return {}
    examples: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        query = sample.query.strip()
        if not query:
            continue
        for tool_name in answers.get(sample.sample_id, []):
            bucket = examples[str(tool_name)]
            if len(bucket) < max_examples_per_tool:
                bucket.append(query)
    return dict(examples)


def write_generated_queries(path: Path, manual: list[ManualEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
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
