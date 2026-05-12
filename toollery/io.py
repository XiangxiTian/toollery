from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import ManualEntry, ToolSpec


def load_tools(path: str | Path) -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tools", [])
    return [ToolSpec.from_dict(item) for item in data]


def save_tools(path: str | Path, tools: Iterable[ToolSpec]) -> None:
    payload = {"tools": [tool.to_dict() for tool in tools]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manual(path: str | Path) -> list[ManualEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("manual", [])
    return [ManualEntry.from_dict(item) for item in data]


def save_manual(path: str | Path, manual: Iterable[ManualEntry]) -> None:
    payload = {"manual": [entry.to_dict() for entry in manual]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
