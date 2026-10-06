import json
from pathlib import Path

import pytest

from regression_guard.dataset import DatasetError, load_dataset

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, *rows, raw=None):
    p = tmp_path / "d.jsonl"
    p.write_text(raw if raw is not None else "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_bundled_dataset_loads():
    cases = load_dataset(ROOT / "data/golden_triage.jsonl")
    assert 60 <= len(cases) <= 100
    assert len({c.id for c in cases}) == len(cases)


def test_defaults(tmp_path):
    (c,) = load_dataset(write(tmp_path, {"id": "a", "input": "hi", "expected": "yo"}))
    assert c.weight == 1.0 and c.tags == () and c.criteria == [{"type": "exact"}]


@pytest.mark.parametrize("row, message", [
    ({"input": "x", "expected": "y"}, "'id'"),
    ({"id": "a", "input": "", "expected": "y"}, "'input'"),
    ({"id": "a", "input": "x"}, "expected"),
    ({"id": "a", "input": "x", "expected": "y", "weight": 0}, "weight"),
    ({"id": "a", "input": "x", "expected": "y", "weight": True}, "weight"),
    ({"id": "a", "input": "x", "expected": "y", "tags": "oops"}, "tags"),
    ({"id": "a", "input": "x", "criteria": [{"type": "vibes"}]}, "unknown criterion"),
    ({"id": "a", "input": "x", "criteria": [{"type": "regex", "pattern": "("}]}, "bad regex"),
    ({"id": "a", "input": "x", "criteria": [{"type": "json_schema"}]}, "schema"),
    ({"id": "a", "input": "x", "criteria": [{"type": "custom", "callable": "nocolon"}]}, "callable"),
    ({"id": "a", "input": "x", "criteria": [{"type": "llm_judge"}]}, "rubric"),
])
def test_validation_errors_name_the_line(tmp_path, row, message):
    good = {"id": "ok", "input": "x", "expected": "y"}
    with pytest.raises(DatasetError, match=r"line 2.*" + message):
        load_dataset(write(tmp_path, good, row))


def test_invalid_json_and_non_object(tmp_path):
    with pytest.raises(DatasetError, match="line 1: invalid JSON"):
        load_dataset(write(tmp_path, raw="{nope"))
    with pytest.raises(DatasetError, match="line 1: each line must be a JSON object"):
        load_dataset(write(tmp_path, raw="[1, 2]"))


def test_duplicate_ids_and_empty_file(tmp_path):
    row = {"id": "a", "input": "x", "expected": "y"}
    with pytest.raises(DatasetError, match="duplicate id 'a'"):
        load_dataset(write(tmp_path, row, row))
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(write(tmp_path, raw="\n\n"))
