"""Compare a run to the baseline and decide whether it is a regression."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from math import comb
from pathlib import Path

from .runner import aggregate


@dataclass
class Rules:
    max_overall_drop: float = 0.05   # weighted score points, 0.05 = 5 points
    max_tag_drop: float = 0.10
    max_flips: int = 5               # pass->fail cases tolerated
    min_tag_cases: int = 5           # ignore tags smaller than this
    stat_guard: bool = True          # require the paired test to reject "just noise"
    alpha: float = 0.05              # overall and flip rules
    tag_alpha: float = 0.10          # per tag (small n, so looser)


def load_rules(path=None) -> Rules:
    if path is None or not Path(path).is_file():
        return Rules()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(Rules)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown rule(s) in {path}: {', '.join(sorted(unknown))}")
    return Rules(**raw)


def mcnemar_p(worse: int, better: int) -> float:
    """One-sided exact McNemar: chance of at least `worse` pass->fail flips among the
    worse+better discordant cases if flips were equally likely either way (pure noise)."""
    n = worse + better
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(worse, n + 1)) / 2 ** n


@dataclass
class Finding:
    rule: str
    message: str


@dataclass
class Comparison:
    regressed: bool
    findings: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    flips: list = field(default_factory=list)       # (baseline, current) pass->fail
    fixes: list = field(default_factory=list)       # fail->pass
    p_overall: float = 1.0
    compared: int = 0
    skipped: list = field(default_factory=list)     # ids not in both runs


def _paired(base, cur):
    worse = [(b, c) for b, c in zip(base, cur) if b.passed and not c.passed]
    better = [(b, c) for b, c in zip(base, cur) if not b.passed and c.passed]
    return worse, better


def compare(baseline, current, rules: Rules | None = None) -> Comparison:
    rules = rules or Rules()
    bmap, cmap = {r.id: r for r in baseline}, {r.id: r for r in current}
    ids = [r.id for r in baseline if r.id in cmap]
    if not ids:
        raise ValueError("baseline and current run share no case ids")
    base, cur = [bmap[i] for i in ids], [cmap[i] for i in ids]
    skipped = sorted(set(bmap) ^ set(cmap))

    bm, cm = aggregate(base), aggregate(cur)
    worse, better = _paired(base, cur)
    p_all = mcnemar_p(len(worse), len(better))
    out = Comparison(False, overall={"baseline": bm["overall"], "current": cm["overall"],
                                     "delta": cm["overall"]["score"] - bm["overall"]["score"]},
                     flips=worse, fixes=better, p_overall=p_all, compared=len(ids), skipped=skipped)

    def significant(p, alpha):
        return (not rules.stat_guard) or p <= alpha

    delta_all = out.overall["delta"]
    if -delta_all >= rules.max_overall_drop and significant(p_all, rules.alpha):
        out.findings.append(Finding("overall_drop", f"overall score fell {-delta_all:.1%} "
            f"({bm['overall']['score']:.1%} -> {cm['overall']['score']:.1%}), p={p_all:.4f}"))
    elif delta_all >= rules.max_overall_drop:
        out.improvements.append(f"overall score improved {delta_all:.1%}")
    for tag, b in bm["tags"].items():
        c = cm["tags"][tag]
        delta = c["score"] - b["score"]
        tw, tb = _paired([x for x in base if tag in x.tags], [x for x in cur if tag in x.tags])
        p = mcnemar_p(len(tw), len(tb))
        out.tags[tag] = {"baseline": b, "current": c, "delta": delta, "p": p}
        if b["n"] < rules.min_tag_cases:
            continue
        if -delta >= rules.max_tag_drop and significant(p, rules.tag_alpha):
            out.findings.append(Finding("tag_drop", f"tag '{tag}' fell {-delta:.1%} "
                f"({b['score']:.1%} -> {c['score']:.1%}, n={b['n']}), p={p:.4f}"))
        elif delta >= rules.max_tag_drop:
            out.improvements.append(f"tag '{tag}' improved {delta:.1%} ({b['score']:.1%} -> {c['score']:.1%})")
    if len(worse) > rules.max_flips and significant(p_all, rules.alpha):
        out.findings.append(Finding("case_flips", f"{len(worse)} cases flipped pass->fail "
            f"(cap {rules.max_flips}) against {len(better)} fail->pass, p={p_all:.4f}"))
    if better:
        out.improvements.append(f"{len(better)} case(s) flipped fail->pass")

    out.flips.sort(key=lambda bc: (-bc[1].weight, bc[1].score))
    out.regressed = bool(out.findings)
    return out
