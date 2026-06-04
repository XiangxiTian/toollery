from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bfcl import (
    load_bfcl_answers,
    load_bfcl_samples,
    load_bfcl_scaled_dataset,
    run_bfcl_batch,
    run_bfcl_scaled_samples,
    run_bfcl_scaletool,
    save_bfcl_predictions,
    save_bfcl_scaled_dataset,
    save_bfcl_scale_predictions,
    summarize_predictions,
    summarize_scale_predictions,
    unique_tools,
)
from .io import load_manual, load_tools, save_manual
from .manual import synthesize_tool_manual
from .pipeline import ToolleryAgent
from .rag_baseline import (
    EmbeddingConfig,
    compare_scale_predictions,
    load_bfcl_scale_predictions,
    load_skill_scale_predictions,
    make_rag_retriever,
    run_bfcl_rag_scaled_samples,
    run_skill_rag_scaled_samples,
    save_comparison,
)
from .scaletool import EvaluationCase, evaluate_scaletool
from .skills import (
    generate_skill_benchmark,
    load_skill_benchmark,
    load_skill_scaled_data,
    load_skill_tools as parse_skill_tools,
    load_skill_tools_file,
    run_skill_scaled_samples,
    run_skill_scaletool,
    save_skill_benchmark,
    save_skill_predictions,
    save_skill_scaled_data,
    save_skill_tools,
    summarize_skill_predictions,
)
from .llm import OpenAICompatibleLLM


def main() -> None:
    parser = argparse.ArgumentParser(prog="toollery", description="Toollery workflow in Python")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manual", help="offline synthesis + verification")
    build.add_argument("--tools", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--queries-per-tool", type=int, default=8)
    build.add_argument("--distractors", type=int, default=8)

    query = subparsers.add_parser("query", help="online proxy-query matching + final selection")
    query.add_argument("--tools", required=True)
    query.add_argument("--manual", required=True)
    query.add_argument("--q", required=True)
    query.add_argument("--top-k", type=int, default=5)

    bench = subparsers.add_parser("benchmark", help="ScaleTool-style candidate growth test")
    bench.add_argument("--tools", required=True)
    bench.add_argument("--manual", required=True)
    bench.add_argument("--cases", required=True)
    bench.add_argument("--sizes", default="2,5,10,20,50,100")
    bench.add_argument("--top-k", type=int, default=5)

    bfcl = subparsers.add_parser("bfcl-batch", help="run Toollery over a BFCL JSONL dataset")
    bfcl.add_argument("--data", required=True)
    bfcl.add_argument("--answers")
    bfcl.add_argument("--out", required=True)
    bfcl.add_argument("--manual-out")
    bfcl.add_argument("--top-k", type=int, default=5)
    bfcl.add_argument("--proxy-top-k", type=int, default=20)
    bfcl.add_argument("--queries-per-tool", type=int, default=8)
    bfcl.add_argument("--distractors", type=int, default=8)
    bfcl.add_argument("--limit", type=int)
    bfcl.add_argument("--force-rebuild", action="store_true")

    bfcl_scale = subparsers.add_parser(
        "bfcl-scaletool",
        help="expand BFCL candidate pools with ScaleTool, then run Toollery",
    )
    bfcl_scale.add_argument("--data", required=True)
    bfcl_scale.add_argument("--answers", required=True)
    bfcl_scale.add_argument("--out", required=True)
    bfcl_scale.add_argument("--manual-out")
    bfcl_scale.add_argument("--scaled-data-out")
    bfcl_scale.add_argument("--sizes", default="2,5,10,20,50,100")
    bfcl_scale.add_argument("--top-k", type=int, default=5)
    bfcl_scale.add_argument("--proxy-top-k", type=int, default=20)
    bfcl_scale.add_argument("--queries-per-tool", type=int, default=8)
    bfcl_scale.add_argument("--distractors", type=int, default=8)
    bfcl_scale.add_argument("--limit", type=int)
    bfcl_scale.add_argument("--seed", type=int, default=11)
    bfcl_scale.add_argument("--force-rebuild", action="store_true")

    bfcl_rag = subparsers.add_parser(
        "bfcl-rag-baseline",
        help="run a name-description RAG baseline over scaled BFCL candidate pools",
    )
    bfcl_rag.add_argument("--config")
    bfcl_rag.add_argument("--scaled-data", required=True)
    bfcl_rag.add_argument("--out", required=True)
    bfcl_rag.add_argument("--compare-to")
    bfcl_rag.add_argument("--compare-out")
    bfcl_rag.add_argument("--top-k", type=int, default=3)
    bfcl_rag.add_argument("--backend", choices=("raganything", "tfidf", "lightrag"), default="raganything")
    bfcl_rag.add_argument("--rag-working-dir", default=".raganything_bfcl")

    skill_scale = subparsers.add_parser(
        "skill-scaletool",
        help="generate an LLM skill benchmark, scale candidate pools, then run Toollery",
    )
    skill_scale.add_argument("--config")
    skill_scale.add_argument("--skills-root")
    skill_scale.add_argument("--tools-out")
    skill_scale.add_argument("--raw-benchmark-out")
    skill_scale.add_argument("--benchmark-out")
    skill_scale.add_argument("--manual-out")
    skill_scale.add_argument("--scaled-data-out")
    skill_scale.add_argument("--out")
    skill_scale.add_argument("--queries-per-skill", type=int, default=5)
    skill_scale.add_argument("--sizes", default="2,5,10,20,50,100,200")
    skill_scale.add_argument("--top-k", type=int, default=3)
    skill_scale.add_argument("--proxy-top-k", type=int, default=20)
    skill_scale.add_argument("--manual-queries-per-tool", type=int, default=8)
    skill_scale.add_argument("--manual-distractors", type=int, default=8)
    skill_scale.add_argument("--verifier-distractors", type=int, default=8)
    skill_scale.add_argument("--limit", type=int)
    skill_scale.add_argument("--seed", type=int, default=23)
    skill_scale.add_argument("--force-rebuild", action="store_true")

    skill_rag = subparsers.add_parser(
        "skill-rag-baseline",
        help="run a name-description RAG baseline over scaled skill candidate pools",
    )
    skill_rag.add_argument("--scaled-data", required=True)
    skill_rag.add_argument("--out", required=True)
    skill_rag.add_argument("--compare-to")
    skill_rag.add_argument("--compare-out")
    skill_rag.add_argument("--top-k", type=int, default=3)
    skill_rag.add_argument("--backend", choices=("raganything", "tfidf", "lightrag"), default="raganything")
    skill_rag.add_argument("--rag-working-dir", default=".raganything_skills")

    args = parser.parse_args()
    if args.command == "build-manual":
        tools = load_tools(args.tools)
        manual = synthesize_tool_manual(
            tools,
            queries_per_tool=args.queries_per_tool,
            distractor_count=args.distractors,
        )
        save_manual(args.out, manual)
        print(f"wrote {len(manual)} verified proxy queries to {Path(args.out).resolve()}")
    elif args.command == "query":
        tools = load_tools(args.tools)
        manual = load_manual(args.manual)
        agent = ToolleryAgent(tools, manual, tool_top_k=args.top_k)
        call, candidates = agent.run(args.q)
        print(
            json.dumps(
                {
                    "tool_call": call.__dict__,
                    "candidates": [
                        {
                            "tool": item.tool.name,
                            "score": item.score,
                            "supporting_queries": [hit.__dict__ for hit in item.supporting_queries],
                        }
                        for item in candidates
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "benchmark":
        tools = load_tools(args.tools)
        manual = load_manual(args.manual)
        cases = _load_cases(args.cases)
        sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
        results = evaluate_scaletool(tools, manual, cases, sizes, tool_top_k=args.top_k)
        print(
            json.dumps(
                [result.__dict__ for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "bfcl-batch":
        samples = load_bfcl_samples(args.data)
        answers = load_bfcl_answers(args.answers) if args.answers else None
        reused_manual = _should_reuse(args.manual_out, args.force_rebuild)
        manual = _maybe_load_manual(args.manual_out, args.force_rebuild)
        predictions, manual = run_bfcl_batch(
            samples,
            manual=manual,
            answers=answers,
            tool_top_k=args.top_k,
            proxy_top_k=args.proxy_top_k,
            queries_per_tool=args.queries_per_tool,
            distractor_count=args.distractors,
            limit=args.limit,
        )
        save_bfcl_predictions(args.out, predictions)
        if args.manual_out:
            save_manual(args.manual_out, manual)
        summary = summarize_predictions(predictions)
        summary["manual_size"] = len(manual)
        summary["output"] = str(Path(args.out).resolve())
        if args.manual_out:
            summary["manual_output"] = str(Path(args.manual_out).resolve())
            summary["reused_manual"] = reused_manual
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "bfcl-scaletool":
        samples = load_bfcl_samples(args.data)
        answers = load_bfcl_answers(args.answers)
        sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
        reused_manual = _should_reuse(args.manual_out, args.force_rebuild)
        manual = _maybe_load_manual(args.manual_out, args.force_rebuild)
        reused_scaled_data = (
            bool(args.scaled_data_out)
            and Path(args.scaled_data_out).exists()
            and not args.force_rebuild
        )
        if reused_scaled_data:
            scaled_samples = load_bfcl_scaled_dataset(args.scaled_data_out)
            if manual is None:
                manual = synthesize_tool_manual(
                    unique_tools(samples),
                    queries_per_tool=args.queries_per_tool,
                    distractor_count=args.distractors,
                )
            predictions = run_bfcl_scaled_samples(
                scaled_samples,
                manual=manual,
                tool_top_k=args.top_k,
                proxy_top_k=args.proxy_top_k,
            )
        else:
            predictions, manual = run_bfcl_scaletool(
                samples,
                answers=answers,
                candidate_sizes=sizes,
                manual=manual,
                tool_top_k=args.top_k,
                proxy_top_k=args.proxy_top_k,
                queries_per_tool=args.queries_per_tool,
                distractor_count=args.distractors,
                limit=args.limit,
                seed=args.seed,
            )
        save_bfcl_scale_predictions(args.out, predictions)
        if args.manual_out:
            save_manual(args.manual_out, manual)
        if args.scaled_data_out and not reused_scaled_data:
            save_bfcl_scaled_dataset(
                args.scaled_data_out,
                samples=samples,
                answers=answers,
                candidate_sizes=sizes,
                limit=args.limit,
                seed=args.seed,
            )
        summary = {
            "manual_size": len(manual),
            "sizes": summarize_scale_predictions(predictions),
            "output": str(Path(args.out).resolve()),
        }
        if args.manual_out:
            summary["manual_output"] = str(Path(args.manual_out).resolve())
            summary["reused_manual"] = reused_manual
        if args.scaled_data_out:
            summary["scaled_data_output"] = str(Path(args.scaled_data_out).resolve())
            summary["reused_scaled_data"] = reused_scaled_data
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "bfcl-rag-baseline":
        _apply_bfcl_rag_config(args)
        scaled_samples = load_bfcl_scaled_dataset(args.scaled_data)
        if args.backend in {"raganything", "lightrag"}:
            _require_embedding_config(args)
        retriever = make_rag_retriever(
            args.backend,
            args.rag_working_dir,
            embedding_config=args.embedding_config,
        )
        try:
            predictions = run_bfcl_rag_scaled_samples(
                scaled_samples,
                top_k=args.top_k,
                retriever=retriever,
            )
        finally:
            close = getattr(retriever, "close", None)
            if close is not None:
                close()
        save_bfcl_scale_predictions(args.out, predictions)
        summary = {
            "backend": args.backend,
            "metric": "top_k_hit",
            "sizes": summarize_scale_predictions(predictions),
            "scaled_data": str(Path(args.scaled_data).resolve()),
            "output": str(Path(args.out).resolve()),
        }
        if args.compare_to:
            comparison = compare_scale_predictions(
                load_bfcl_scale_predictions(args.compare_to),
                predictions,
            )
            if args.compare_out:
                save_comparison(args.compare_out, comparison)
                summary["comparison_output"] = str(Path(args.compare_out).resolve())
            summary["comparison_rows"] = len(comparison)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "skill-scaletool":
        _apply_skill_config(args)
        _require_skill_args(args)
        sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]

        reused_tools = _should_reuse(args.tools_out, args.force_rebuild)
        if reused_tools:
            tools = load_skill_tools_file(args.tools_out)
            if args.limit is not None:
                tools = tools[: args.limit]
        else:
            tools = parse_skill_tools(args.skills_root, limit=args.limit)
            save_skill_tools(args.tools_out, tools)

        reused_raw = _should_reuse(args.raw_benchmark_out, args.force_rebuild)
        reused_benchmark = _should_reuse(args.benchmark_out, args.force_rebuild)
        if reused_benchmark:
            benchmark = load_skill_benchmark(args.benchmark_out, verified_only=True)
            raw_benchmark = load_skill_benchmark(args.raw_benchmark_out) if reused_raw else []
        else:
            if reused_raw:
                raw_benchmark = load_skill_benchmark(args.raw_benchmark_out)
                benchmark = [row for row in raw_benchmark if row.accepted is True]
                save_skill_benchmark(args.benchmark_out, benchmark)
            else:
                llm = OpenAICompatibleLLM()
                raw_benchmark, benchmark = generate_skill_benchmark(
                    tools,
                    skills_root=args.skills_root,
                    llm=llm,
                    queries_per_skill=args.queries_per_skill,
                    verifier_distractors=args.verifier_distractors,
                    seed=args.seed,
                )
                save_skill_benchmark(args.raw_benchmark_out, raw_benchmark)
                save_skill_benchmark(args.benchmark_out, benchmark)

        reused_manual = _should_reuse(args.manual_out, args.force_rebuild)
        manual = _maybe_load_manual(args.manual_out, args.force_rebuild)

        reused_scaled_data = _should_reuse(args.scaled_data_out, args.force_rebuild)
        if reused_scaled_data:
            scaled_samples = load_skill_scaled_data(args.scaled_data_out)
            if manual is None:
                manual = synthesize_tool_manual(
                    tools,
                    queries_per_tool=args.manual_queries_per_tool,
                    distractor_count=args.manual_distractors,
                )
            predictions = run_skill_scaled_samples(
                scaled_samples,
                manual=manual,
                tool_top_k=args.top_k,
                proxy_top_k=args.proxy_top_k,
            )
        else:
            predictions, manual, scaled_samples = run_skill_scaletool(
                tools,
                benchmark,
                candidate_sizes=sizes,
                manual=manual,
                tool_top_k=args.top_k,
                proxy_top_k=args.proxy_top_k,
                manual_queries_per_tool=args.manual_queries_per_tool,
                manual_distractors=args.manual_distractors,
                seed=args.seed,
            )
            save_skill_scaled_data(args.scaled_data_out, scaled_samples)

        save_manual(args.manual_out, manual)
        save_skill_predictions(args.out, predictions)
        summary = {
            "skills": len(tools),
            "raw_benchmark_rows": len(raw_benchmark),
            "verified_benchmark_rows": len(benchmark),
            "manual_size": len(manual),
            "sizes": summarize_skill_predictions(predictions),
            "tools_output": str(Path(args.tools_out).resolve()),
            "raw_benchmark_output": str(Path(args.raw_benchmark_out).resolve()),
            "benchmark_output": str(Path(args.benchmark_out).resolve()),
            "manual_output": str(Path(args.manual_out).resolve()),
            "scaled_data_output": str(Path(args.scaled_data_out).resolve()),
            "output": str(Path(args.out).resolve()),
            "reused_tools": reused_tools,
            "reused_raw_benchmark": reused_raw,
            "reused_benchmark": reused_benchmark,
            "reused_manual": reused_manual,
            "reused_scaled_data": reused_scaled_data,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "skill-rag-baseline":
        scaled_samples = load_skill_scaled_data(args.scaled_data)
        if args.backend in {"raganything", "lightrag"}:
            _require_openai_env()
        retriever = make_rag_retriever(args.backend, args.rag_working_dir)
        predictions = run_skill_rag_scaled_samples(
            scaled_samples,
            top_k=args.top_k,
            retriever=retriever,
        )
        save_skill_predictions(args.out, predictions)
        summary = {
            "backend": args.backend,
            "metric": "top_k_hit",
            "sizes": summarize_skill_predictions(predictions),
            "scaled_data": str(Path(args.scaled_data).resolve()),
            "output": str(Path(args.out).resolve()),
        }
        if args.compare_to:
            comparison = compare_scale_predictions(
                load_skill_scale_predictions(args.compare_to),
                predictions,
            )
            if args.compare_out:
                save_comparison(args.compare_out, comparison)
                summary["comparison_output"] = str(Path(args.compare_out).resolve())
            summary["comparison_rows"] = len(comparison)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_cases(path: str) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [EvaluationCase(str(item["query"]), str(item["ground_truth_tool"])) for item in data]


def _maybe_load_manual(path: str | None, force_rebuild: bool):
    if path and Path(path).exists() and not force_rebuild:
        return load_manual(path)
    return None


def _should_reuse(path: str | None, force_rebuild: bool) -> bool:
    return bool(path) and Path(path).exists() and not force_rebuild


def _apply_skill_config(args) -> None:
    if not args.config:
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if "openai" in config:
        _apply_env_defaults(config["openai"])
    values = config.get("skill_scaletool", config)
    for key, value in values.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            continue
        current = getattr(args, attr)
        if current is None or _is_parser_default(attr, current):
            if attr == "sizes" and isinstance(value, list):
                value = ",".join(str(item) for item in value)
            setattr(args, attr, value)


def _apply_bfcl_rag_config(args) -> None:
    if not args.config:
        args.embedding_config = _resolve_embedding_config({})
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    llm_config = _llm_config(config)
    embedding_config = _embedding_config(config)
    if llm_config:
        _apply_env_defaults(llm_config)
    args.embedding_config = _resolve_embedding_config(embedding_config)

    values = config.get("bfcl_rag", config)
    for key, value in values.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            continue
        current = getattr(args, attr)
        if current is None or _is_bfcl_rag_parser_default(attr, current):
            setattr(args, attr, value)


def _llm_config(config: dict) -> dict:
    for section in ("openai", "deepseek", "llm"):
        value = config.get(section)
        if isinstance(value, dict):
            return value
    return {}


def _embedding_config(config: dict) -> dict:
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


def _resolve_embedding_config(config: dict) -> EmbeddingConfig:
    api_key = _resolve_api_key(config)
    return EmbeddingConfig(
        backend=str(config.get("backend") or "openai-compatible"),
        api_key=str(api_key) if api_key else None,
        base_url=str(
            config.get("base_url")
            or _env("OPENAI_API_BASE")
            or _env("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ),
        model=str(config.get("model") or _env("EMBEDDING_MODEL") or "text-embedding-3-small"),
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


def _resolve_api_key(config: dict) -> str | None:
    api_key = config.get("api_key")
    if api_key:
        return str(api_key)
    api_key_env = config.get("api_key_env")
    if not api_key_env:
        return None
    env_name = str(api_key_env)
    env_value = _env(env_name)
    if env_value:
        return env_value
    if _looks_like_api_key(env_name):
        return env_name
    return None


def _looks_like_api_key(value: str) -> bool:
    return value.startswith(("sk-", "sk_", "ak-", "ak_")) or len(value) >= 32 and "-" in value


def _require_embedding_config(args) -> None:
    config = getattr(args, "embedding_config", None)
    if not config or not config.api_key:
        raise SystemExit(
            "RAG-Anything backend requires an embedding API key. "
            "Add an `embedding` section with `api_key` or `api_key_env` to the config."
        )


def _env(name: str) -> str | None:
    import os

    return os.getenv(name)


def _apply_env_defaults(config: dict) -> None:
    import os

    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env))
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)
    mapping = {
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
    }
    for key, env_name in mapping.items():
        value = config.get(key)
        if value and not os.getenv(env_name):
            os.environ[env_name] = str(value)
    if config.get("base_url") and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = str(config["base_url"])


def _require_openai_env() -> None:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "RAG-Anything backend requires OPENAI_API_KEY. "
            "Set openai.api_key in config, or export the env var named by openai.api_key_env."
        )
    if not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _is_parser_default(attr: str, value) -> bool:
    defaults = {
        "queries_per_skill": 5,
        "sizes": "2,5,10,20,50,100,200",
        "top_k": 3,
        "proxy_top_k": 20,
        "manual_queries_per_tool": 8,
        "manual_distractors": 8,
        "verifier_distractors": 8,
        "seed": 23,
        "force_rebuild": False,
    }
    return attr in defaults and value == defaults[attr]


def _is_bfcl_rag_parser_default(attr: str, value) -> bool:
    defaults = {
        "top_k": 3,
        "backend": "raganything",
        "rag_working_dir": ".raganything_bfcl",
    }
    return attr in defaults and value == defaults[attr]


def _require_skill_args(args) -> None:
    required = [
        "skills_root",
        "tools_out",
        "raw_benchmark_out",
        "benchmark_out",
        "manual_out",
        "scaled_data_out",
        "out",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(
            "skill-scaletool missing required settings: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
            + ". Provide them on the command line or in --config."
        )


if __name__ == "__main__":
    main()
