from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_manual, load_tools, save_manual
from .manual import synthesize_tool_manual
from .pipeline import ToolleryAgent
from .scaletool import EvaluationCase, evaluate_scaletool


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


def _load_cases(path: str) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [EvaluationCase(str(item["query"]), str(item["ground_truth_tool"])) for item in data]


if __name__ == "__main__":
    main()
