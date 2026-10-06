"""Markdown report for a comparison."""
from __future__ import annotations


def _clip(text, n=160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def render_markdown(cmp, meta: dict, worst: int = 8) -> str:
    status = "REGRESSION DETECTED" if cmp.regressed else "No regression"
    o = cmp.overall
    lines = [
        f"# Regression check: {meta.get('feature', 'feature')} - {status}",
        "",
        f"- Prompt `{meta.get('prompt_sha', '?')}` vs baseline `{meta.get('baseline_prompt_sha', '?')}`; "
        f"{cmp.compared} cases compared.",
        f"- Overall score: {o['baseline']['score']:.1%} -> {o['current']['score']:.1%} "
        f"({o['delta']:+.1%}); pass rate {o['baseline']['pass_rate']:.1%} -> {o['current']['pass_rate']:.1%}.",
        f"- Paired test (exact McNemar, one-sided): {len(cmp.flips)} pass->fail vs {len(cmp.fixes)} "
        f"fail->pass, p={cmp.p_overall:.4f}.",
    ]
    if meta.get("simulated"):
        lines.append("- Outputs come from the SIMULATED mock provider, not a real model.")
    if cmp.skipped:
        lines.append(f"- Not compared (id in only one run): {', '.join(cmp.skipped[:10])}")
    lines += ["", "## Findings", ""]
    lines += [f"- **{f.rule}**: {f.message}" for f in cmp.findings] or ["- None. Changes are within thresholds or noise."]
    if cmp.improvements:
        lines += ["", "## Improvements", ""] + [f"- {i}" for i in cmp.improvements]
    lines += ["", "## Tags", "", "| Tag | n | Baseline | Current | Delta | p |", "|---|---|---|---|---|---|"]
    for tag, t in sorted(cmp.tags.items(), key=lambda kv: kv[1]["delta"]):
        lines.append(f"| {tag} | {t['baseline']['n']} | {t['baseline']['score']:.1%} | "
                     f"{t['current']['score']:.1%} | {t['delta']:+.1%} | {t['p']:.3f} |")
    if cmp.flips:
        lines += ["", f"## Worst flips (pass -> fail), top {min(worst, len(cmp.flips))} of {len(cmp.flips)}", ""]
        for _, c in cmp.flips[:worst]:
            lines += [f"### `{c.id}` (weight {c.weight:g}, tags: {', '.join(c.tags)})",
                      f"- Input: {_clip(c.input)}",
                      f"- Expected: `{_clip(c.expected)}`",
                      f"- Actual: `{_clip(c.output)}`",
                      f"- Why: {_clip('; '.join(c.details) or 'n/a')}", ""]
    return "\n".join(lines).rstrip() + "\n"
