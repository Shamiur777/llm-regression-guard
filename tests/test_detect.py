from dataclasses import replace

import pytest

from regression_guard.detect import Rules, compare, load_rules, mcnemar_p
from regression_guard.runner import CaseResult


def run(passed_flags, tags_for=lambda i: ["all"]):
    return [CaseResult(f"c{i:03d}", f"input {i}", {"k": i}, f"out {i}", tags_for(i), 1.0,
                       1.0 if ok else 0.0, ok, [] if ok else ["wrong"])
            for i, ok in enumerate(passed_flags)]


def flipped(base_flags, down=(), up=()):
    out = list(base_flags)
    for i in down:
        out[i] = False
    for i in up:
        out[i] = True
    return out


def test_mcnemar_p_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) > 0.5
    assert mcnemar_p(10, 0) == pytest.approx(1 / 1024)
    assert mcnemar_p(0, 10) == 1.0  # improvement is never "significant" as a regression


def test_identical_runs_do_not_alert():
    base = run([True] * 90 + [False] * 10)
    cmp = compare(base, base)
    assert not cmp.regressed and cmp.overall["delta"] == 0 and not cmp.flips


def test_noise_does_not_alert_but_would_without_guard():
    base_flags = [True] * 90 + [False] * 10
    # 8 cases got worse, 7 got better: more than the flip cap, overall drop 1 point
    cur_flags = flipped(base_flags, down=range(8), up=range(90, 97))
    base, cur = run(base_flags), run(cur_flags)
    cmp = compare(base, cur, Rules(max_overall_drop=0.0))
    assert len(cmp.flips) == 8 and len(cmp.fixes) == 7
    assert not cmp.regressed
    # same data with the guard off trips the flip cap and the zero-tolerance drop rule
    unguarded = compare(base, cur, Rules(max_overall_drop=0.0, stat_guard=False))
    assert {f.rule for f in unguarded.findings} == {"overall_drop", "case_flips"}


def test_real_drop_alerts_and_reports_worst_flips():
    base_flags = [True] * 100
    cur = run(flipped(base_flags, down=range(20)))
    cmp = compare(run(base_flags), cur)
    assert cmp.regressed
    assert {f.rule for f in cmp.findings} >= {"overall_drop", "case_flips"}
    assert len(cmp.flips) == 20 and cmp.p_overall < 0.001


def test_small_drop_below_threshold_is_not_alerted():
    base_flags = [True] * 100
    cmp = compare(run(base_flags), run(flipped(base_flags, down=range(4))))
    assert not cmp.regressed  # 4 points < 5 point threshold and 4 flips <= cap


def test_per_tag_drop_alerts_when_overall_is_fine():
    tags = lambda i: ["small"] if i < 10 else ["rest"]
    base_flags = [True] * 100
    cur = run(flipped(base_flags, down=range(4)), tags)  # 4 of 10 in tag "small" fail
    cmp = compare(run(base_flags, tags), cur)
    assert [f.rule for f in cmp.findings] == ["tag_drop"]
    assert "small" in cmp.findings[0].message


def test_tiny_tags_are_ignored():
    tags = lambda i: ["tiny"] if i < 3 else ["rest"]
    base_flags = [True] * 100
    cmp = compare(run(base_flags, tags), run(flipped(base_flags, down=range(3)), tags))
    assert not cmp.regressed


def test_improvements_are_reported_not_alerted():
    base_flags = [False] * 20 + [True] * 80
    cmp = compare(run(base_flags), run([True] * 100))
    assert not cmp.regressed
    assert any("overall score improved" in i for i in cmp.improvements)
    assert any("20 case(s) flipped fail->pass" in i for i in cmp.improvements)


def test_mismatched_ids_are_skipped_and_disjoint_is_an_error():
    base = run([True] * 10)
    cur = run([True] * 9)  # c009 missing
    cmp = compare(base, cur)
    assert cmp.compared == 9 and cmp.skipped == ["c009"]
    with pytest.raises(ValueError, match="no case ids"):
        compare(base, [replace(r, id="z" + r.id) for r in base])


def test_load_rules(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"max_flips": 2}')
    assert load_rules(p).max_flips == 2 and load_rules(p).alpha == 0.05
    p.write_text('{"max_flipz": 2}')
    with pytest.raises(ValueError, match="unknown rule"):
        load_rules(p)
    assert load_rules(tmp_path / "missing.json") == Rules()
