"""Scorers. Each takes (spec, case, output, judge) and returns (passed, detail)."""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field

_MISSING = object()


class ScorerError(Exception):
    """Misconfiguration (not a model failure): bad spec, missing judge, etc."""


def extract_json(text: str):
    """Parse the output as JSON; if there is chatter around it, take the first {...} object."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    break
        start = text.find("{", start + 1)
    return _MISSING


def _actual(spec: dict, output: str):
    """The value being judged: a JSON field if spec has `field`, else the raw output."""
    name = spec.get("field")
    if name is None:
        return output.strip()
    data = extract_json(output)
    if data is _MISSING or not isinstance(data, dict) or name not in data:
        return _MISSING
    return data[name]


def _wanted(spec: dict, case):
    if "value" in spec:
        return spec["value"]
    exp = case.expected
    if spec.get("field") and isinstance(exp, dict):
        return exp.get(spec["field"], _MISSING)
    return _MISSING if exp is None else exp


def exact(spec, case, output, judge=None):
    got, want = _actual(spec, output), _wanted(spec, case)
    if got is _MISSING:
        return False, f"field {spec.get('field')!r} not found in output"
    if want is _MISSING:
        raise ScorerError(f"case {case.id}: no expected value for exact match")
    if spec.get("case_insensitive") and isinstance(got, str) and isinstance(want, str):
        got, want = got.lower(), want.lower()
    return got == want, f"got {got!r}, want {want!r}"


def contains(spec, case, output, judge=None):
    got = _actual(spec, output)
    if got is _MISSING:
        return False, f"field {spec.get('field')!r} not found in output"
    needles = spec.get("any_of") or [_wanted(spec, case)]
    text = str(got).lower()
    hit = any(str(n).lower() in text for n in needles)
    return hit, f"looking for any of {needles!r}"


def regex(spec, case, output, judge=None):
    got = _actual(spec, output)
    if got is _MISSING:
        return False, f"field {spec.get('field')!r} not found in output"
    ok = re.search(spec["pattern"], str(got)) is not None
    return ok, f"pattern {spec['pattern']!r}"


def _check_schema(value, schema, path="$"):
    """Small JSON-schema subset: type, enum, required, properties, minLength, minimum, maximum."""
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    t = schema.get("type")
    if t:
        ok = isinstance(value, types[t]) and not (t in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return f"{path}: expected {t}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: {value!r} not in enum"
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return f"{path}: shorter than minLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}: missing required {key!r}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                err = _check_schema(value[key], sub, f"{path}.{key}")
                if err:
                    return err
    return None


def json_schema(spec, case, output, judge=None):
    """Strict: the whole output must be JSON (chatter around it fails), then match the schema."""
    try:
        data = json.loads(output)
    except ValueError:
        return False, "output is not valid JSON on its own"
    err = _check_schema(data, spec["schema"])
    return err is None, err or "schema ok"


def numeric(spec, case, output, judge=None):
    got, want = _actual(spec, output), _wanted(spec, case)
    if got is _MISSING:
        return False, f"field {spec.get('field')!r} not found in output"
    try:
        got, want = float(got), float(want)
    except (TypeError, ValueError):
        return False, f"not numeric: got {got!r}, want {want!r}"
    tol = float(spec.get("tolerance", 0))
    return abs(got - want) <= tol, f"got {got}, want {want} +/- {tol}"


def custom(spec, case, output, judge=None):
    """`callable` is 'package.module:function'; it gets (input, output, expected)."""
    try:
        mod, _, fn = spec["callable"].partition(":")
        func = getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError, KeyError) as e:
        raise ScorerError(f"cannot load custom scorer {spec.get('callable')!r}: {e}") from e
    try:
        res = func(case.input, output, case.expected)
    except Exception as e:  # a crashing check is a failed check, not a crashed run
        return False, f"custom scorer raised {type(e).__name__}: {e}"
    return (res if isinstance(res, tuple) else (bool(res), "custom check"))


def llm_judge(spec, case, output, judge=None):
    if judge is None:
        raise ScorerError("llm_judge criterion used but no judge provider is configured")
    system = ("You are a strict grader. Read the rubric, the input and the answer. "
              "Reply with PASS or FAIL on the first line, then one short reason.")
    user = (f"Rubric: {spec['rubric']}\n\nInput:\n{case.input}\n\nExpected (may be empty):\n"
            f"{case.expected}\n\nAnswer:\n{output}")
    verdict = judge.complete(system, user).strip()
    return verdict.upper().startswith("PASS"), verdict.splitlines()[0] if verdict else "empty verdict"


SCORERS = {
    "exact": exact, "contains": contains, "regex": regex, "json_schema": json_schema,
    "numeric": numeric, "custom": custom, "llm_judge": llm_judge,
}


@dataclass
class CaseScore:
    score: float
    passed: bool
    details: list[str] = field(default_factory=list)


def score_case(case, output: str, judge=None) -> CaseScore:
    """Score = fraction of criteria passed; the case passes only if all do."""
    results = []
    for spec in case.criteria:
        fn = SCORERS.get(spec["type"])
        if fn is None:
            raise ScorerError(f"unknown scorer type {spec['type']!r}")
        ok, detail = fn(spec, case, output, judge)
        results.append((ok, f"{spec['type']}{'.' + spec['field'] if spec.get('field') else ''}: {detail}"))
    passed = sum(1 for ok, _ in results if ok)
    return CaseScore(
        score=passed / len(results),
        passed=passed == len(results),
        details=[d for ok, d in results if not ok],
    )
