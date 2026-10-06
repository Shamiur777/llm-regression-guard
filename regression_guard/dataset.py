"""Golden dataset: JSONL with id, input, expected and/or criteria, tags, weight."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scorers import SCORERS


class DatasetError(ValueError):
    """The dataset file is malformed. The message names the line."""


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: Any
    criteria: list
    tags: tuple
    weight: float


def _parse_line(n: int, raw: str) -> Case:
    def bad(msg):
        raise DatasetError(f"line {n}: {msg}")

    try:
        row = json.loads(raw)
    except ValueError as e:
        bad(f"invalid JSON ({e})")
    if not isinstance(row, dict):
        bad("each line must be a JSON object")
    cid = row.get("id")
    if not isinstance(cid, str) or not cid.strip():
        bad("'id' must be a non-empty string")
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        bad("'input' must be a non-empty string")
    if "expected" not in row and not row.get("criteria"):
        bad("needs 'expected' or 'criteria'")
    tags = row.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        bad("'tags' must be a list of non-empty strings")
    weight = row.get("weight", 1)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
        bad("'weight' must be a positive number")
    criteria = row.get("criteria") or [{"type": "exact"}]
    if not isinstance(criteria, list) or not all(isinstance(c, dict) for c in criteria):
        bad("'criteria' must be a list of objects")
    for c in criteria:
        if c.get("type") not in SCORERS:
            bad(f"unknown criterion type {c.get('type')!r} (known: {', '.join(sorted(SCORERS))})")
        if c["type"] == "regex":
            try:
                re.compile(c.get("pattern", ""))
            except re.error as e:
                bad(f"bad regex ({e})")
            if "pattern" not in c:
                bad("regex criterion needs 'pattern'")
        if c["type"] == "json_schema" and not isinstance(c.get("schema"), dict):
            bad("json_schema criterion needs a 'schema' object")
        if c["type"] == "custom" and ":" not in str(c.get("callable", "")):
            bad("custom criterion needs 'callable' as 'module:function'")
        if c["type"] == "llm_judge" and not c.get("rubric"):
            bad("llm_judge criterion needs a 'rubric'")
    return Case(cid, row["input"], row.get("expected"), criteria, tuple(tags), float(weight))


def load_dataset(path) -> list[Case]:
    """Load and validate; raises DatasetError with the offending line number."""
    cases, seen = [], {}
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        case = _parse_line(n, raw)
        if case.id in seen:
            raise DatasetError(f"line {n}: duplicate id {case.id!r} (first seen on line {seen[case.id]})")
        seen[case.id] = n
        cases.append(case)
    if not cases:
        raise DatasetError("dataset is empty")
    return cases
