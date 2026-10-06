"""Run a feature over a dataset, aggregate scores, and store/load baselines."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .scorers import score_case


@dataclass
class CaseResult:
    id: str
    input: str
    expected: object
    output: str
    tags: list
    weight: float
    score: float
    passed: bool
    details: list


def run_feature(feature, cases) -> list[CaseResult]:
    results = []
    for c in cases:
        try:
            output = feature.generate(c.input)
        except Exception as e:  # provider outage counts as a failed case, loudly
            output = ""
            results.append(CaseResult(c.id, c.input, c.expected, output, list(c.tags), c.weight,
                                      0.0, False, [f"provider error: {type(e).__name__}: {e}"]))
            continue
        s = score_case(c, output, feature.judge)
        results.append(CaseResult(c.id, c.input, c.expected, output, list(c.tags), c.weight,
                                  s.score, s.passed, s.details))
    return results


def _agg(rows) -> dict:
    total = sum(r.weight for r in rows)
    return {
        "n": len(rows),
        "score": sum(r.weight * r.score for r in rows) / total,
        "pass_rate": sum(r.weight for r in rows if r.passed) / total,
    }


def aggregate(results: list[CaseResult]) -> dict:
    """Weighted overall metrics plus the same per tag."""
    tags = sorted({t for r in results for t in r.tags})
    return {
        "overall": _agg(results),
        "tags": {t: _agg([r for r in results if t in r.tags]) for t in tags},
    }


def save_baseline(path, feature, results) -> dict:
    doc = {
        "meta": {**feature.describe(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "metrics": aggregate(results),
        "results": [asdict(r) for r in results],
    }
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def load_baseline(path) -> tuple[dict, list[CaseResult]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"baseline not found: {p} (create it with: python -m regression_guard baseline)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc["meta"], [CaseResult(**r) for r in doc["results"]]
