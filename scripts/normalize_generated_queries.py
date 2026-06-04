from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.baselines import normalize_generated_queries  # noqa: E402


def main() -> None:
    args = parse_args()
    grouped = normalize_generated_queries(
        source_path=args.source,
        out_path=args.out,
        accepted_only=args.accepted_only,
    )
    summary = {
        "source": str(Path(args.source).resolve()),
        "out": str(Path(args.out).resolve()),
        "accepted_only": args.accepted_only,
        "tools_or_skills": len(grouped),
        "queries": sum(len(items) for items in grouped.values()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Toollery manual_raw JSONL into generated-query rows.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--accepted-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

