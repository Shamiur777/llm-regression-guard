import pytest

from regression_guard.dataset import Case
from regression_guard.providers import ScriptedProvider
from regression_guard.runner import CaseResult, aggregate
from regression_guard.scorers import _MISSING, ScorerError, extract_json, score_case

SCHEMA = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer", "minimum": 1}}}


def case(criteria, expected=None, cid="c", tags=(), weight=1.0):
    return Case(cid, "input text", expected, criteria, tuple(tags), weight)


def passes(criteria, output, expected=None, judge=None):
    return score_case(case(criteria, expected), output, judge).passed


def test_exact_raw_and_case_insensitive():
    assert passes([{"type": "exact"}], "  Paris \n", "Paris")
    assert not passes([{"type": "exact"}], "paris", "Paris")
    assert passes([{"type": "exact", "case_insensitive": True}], "paris", "Paris")


def test_exact_on_json_field_even_with_chatter():
    out = 'Sure! {"label": "spam"} hope that helps'
    assert passes([{"type": "exact", "field": "label"}], out, {"label": "spam"})
    assert not passes([{"type": "exact", "field": "label"}], out, {"label": "ham"})
    assert not passes([{"type": "exact", "field": "missing"}], out, {"missing": 1})


def test_exact_without_expected_is_a_config_error():
    with pytest.raises(ScorerError):
        passes([{"type": "exact", "field": "a"}], '{"a": 1}', expected=None)


def test_contains_and_regex():
    assert passes([{"type": "contains", "any_of": ["refund", "credit"]}], "We will CREDIT you")
    assert not passes([{"type": "contains", "value": "refund"}], "nothing here")
    assert passes([{"type": "regex", "pattern": r"order #\d{4}"}], "see order #1234")
    assert not passes([{"type": "regex", "pattern": r"order #\d{4}"}], "see order #12")


def test_json_schema_is_strict_about_chatter():
    spec = [{"type": "json_schema", "schema": SCHEMA}]
    assert passes(spec, '{"n": 2}')
    assert not passes(spec, 'Here: {"n": 2}')
    assert not passes(spec, '{"n": 0}')          # minimum
    assert not passes(spec, '{"n": "2"}')        # type
    assert not passes(spec, "{}")                # required
    assert not passes(spec, '{"n": true}')       # bool is not an integer


def test_numeric_tolerance():
    spec = [{"type": "numeric", "field": "x", "tolerance": 0.5}]
    assert passes(spec, '{"x": 10.4}', {"x": 10})
    assert not passes(spec, '{"x": 10.6}', {"x": 10})
    assert not passes(spec, '{"x": "abc"}', {"x": 10})


def test_custom_callable_and_failure_modes():
    spec = [{"type": "custom", "callable": "regression_guard.examples.triage:reply_is_concise"}]
    assert passes(spec, '{"reply": "short"}')
    assert not passes(spec, '{"reply": "' + "x" * 300 + '"}')
    assert not passes(spec, "no json at all")
    with pytest.raises(ScorerError):
        passes([{"type": "custom", "callable": "nope.module:fn"}], "x")


def test_llm_judge_uses_provider_and_requires_one():
    spec = [{"type": "llm_judge", "rubric": "polite?"}]
    judge = ScriptedProvider(["PASS\nvery polite"])
    assert passes(spec, "Thank you!", judge=judge)
    assert "polite?" in judge.calls[0][1] and "Thank you!" in judge.calls[0][1]
    assert not passes(spec, "go away", judge=ScriptedProvider(["FAIL rude"]))
    with pytest.raises(ScorerError, match="no judge"):
        passes(spec, "x")


def test_case_score_is_fraction_of_criteria():
    c = case([{"type": "contains", "value": "a"}, {"type": "contains", "value": "b"}])
    s = score_case(c, "a only")
    assert s.score == 0.5 and not s.passed and len(s.details) == 1


def test_extract_json_handles_nested_and_garbage():
    assert extract_json('x {"a": {"b": 1}} y') == {"a": {"b": 1}}
    assert extract_json("no braces") is _MISSING
    assert extract_json("{broken json") is _MISSING


def result(cid, passed, score, tags, weight=1.0):
    return CaseResult(cid, "i", None, "o", list(tags), weight, score, passed, [])


def test_aggregation_weights_overall_and_per_tag():
    rows = [result("a", True, 1.0, ["x", "y"], 3), result("b", False, 0.0, ["x"], 1),
            result("c", False, 0.5, ["y"], 1)]
    m = aggregate(rows)
    assert m["overall"]["score"] == pytest.approx((3 + 0 + 0.5) / 5)
    assert m["overall"]["pass_rate"] == pytest.approx(3 / 5)
    assert m["tags"]["x"] == {"n": 2, "score": pytest.approx(0.75), "pass_rate": pytest.approx(0.75)}
    assert m["tags"]["y"]["score"] == pytest.approx((3 + 0.5) / 4)
