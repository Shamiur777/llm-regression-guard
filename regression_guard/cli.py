"""Command line: python -m regression_guard run|baseline|check|demo."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from . import slack
from .dataset import DatasetError, load_dataset
from .detect import compare, load_rules
from .feature import load_feature
from .providers import MockProvider
from .report import render_markdown
from .runner import aggregate, load_baseline, run_feature, save_baseline
from .scorers import ScorerError

REPO_ROOT = Path(__file__).resolve().parent.parent
EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config/feature.json")
    p.add_argument("--rules", default="config/rules.json")
    p.add_argument("--baseline", default="baseline.json")
    p.add_argument("--root", default=".", help="paths in the config are relative to this")
    p.add_argument("--prompt", action="append", help="use this prompt file instead of the configured one")
    p.add_argument("--seed", type=int, help="override the mock provider seed")
    p.add_argument("--noise", type=float, help="override the mock provider noise rate")


def _load(args):
    feature = load_feature(args.config, args.root, args.prompt, args.seed, args.noise)
    return feature, load_dataset(feature.dataset)


def _meta(feature) -> dict:
    return {**feature.describe(), "simulated": isinstance(feature.provider, MockProvider)}


def _print_metrics(metrics: dict) -> None:
    o = metrics["overall"]
    print(f"overall: score {o['score']:.1%}, pass rate {o['pass_rate']:.1%} ({o['n']} cases)")
    for tag, m in metrics["tags"].items():
        print(f"  {tag:<16} n={m['n']:<3} score {m['score']:.1%}")


def cmd_run(args) -> int:
    feature, cases = _load(args)
    results = run_feature(feature, cases)
    _print_metrics(aggregate(results))
    if args.out:
        Path(args.out).write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    return EXIT_OK


def cmd_baseline(args) -> int:
    feature, cases = _load(args)
    doc = save_baseline(args.baseline, feature, run_feature(feature, cases))
    _print_metrics(doc["metrics"])
    print(f"baseline written to {args.baseline}")
    return EXIT_OK


def check(feature, cases, baseline_path, rules):
    """Run the candidate and compare it to the stored baseline. Returns (comparison, meta)."""
    base_meta, base_results = load_baseline(baseline_path)
    cmp = compare(base_results, run_feature(feature, cases), rules)
    meta = {**_meta(feature), "baseline_prompt_sha": base_meta.get("prompt_sha")}
    return cmp, meta


def alert(cmp, meta, dry_run: bool) -> None:
    payload = slack.build_payload(cmp, meta, run_url=os.environ.get("RUN_URL"))
    if dry_run:
        print("--- Slack payload (dry run) ---")
        print(json.dumps(payload, indent=2))
    elif os.environ.get(slack.ENV_VAR):
        try:
            print(f"Slack alert sent (HTTP {slack.send(payload)})")
        except Exception as e:  # alert failure must not hide the regression result
            print(f"warning: Slack alert failed: {e}", file=sys.stderr)
    else:
        print(f"{slack.ENV_VAR} not set; Slack alert skipped")


def cmd_check(args) -> int:
    feature, cases = _load(args)
    cmp, meta = check(feature, cases, args.baseline, load_rules(args.rules))
    report = render_markdown(cmp, meta)
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    if cmp.regressed:
        alert(cmp, meta, args.dry_run)
    return EXIT_REGRESSION if cmp.regressed else EXIT_OK


def cmd_demo(args) -> int:
    root = REPO_ROOT
    cfg, rules = root / "config/feature.json", load_rules(root / "config/rules.json")
    bad_prompt = [root / "prompts/examples/triage_v2_degraded.txt"]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "baseline.json"
        feature = load_feature(cfg, root)
        cases = load_dataset(feature.dataset)
        print("== SIMULATED demo: outputs come from a mock provider, not a real model ==\n")
        print(f"1. Baseline from prompt v1 ({len(cases)} golden cases)")
        _print_metrics(save_baseline(base, feature, run_feature(feature, cases))["metrics"])

        print("\n2. Re-run v1 with different sampling noise (seed 1): should NOT alert")
        cmp, meta = check(load_feature(cfg, root, seed=1), cases, base, rules)
        print(f"   overall {cmp.overall['delta']:+.1%}, flips {len(cmp.flips)} vs fixes {len(cmp.fixes)}, "
              f"p={cmp.p_overall:.3f} -> {'REGRESSION' if cmp.regressed else 'ok'}")

        print("\n3. Candidate prompt v2 (degraded): should alert\n")
        cmp, meta = check(load_feature(cfg, root, bad_prompt), cases, base, rules)
        print(render_markdown(cmp, meta, worst=4))
        alert(cmp, meta, dry_run=True)
    return EXIT_OK if cmp.regressed else EXIT_ERROR  # demo "succeeds" when v2 is caught


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="regression_guard", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run the feature and print metrics")
    _common(p)
    p.add_argument("--out", help="write per-case results as JSON")
    p = sub.add_parser("baseline", help="run and store baseline.json")
    _common(p)
    p = sub.add_parser("check", help="compare a run to the baseline; exit 1 on regression")
    _common(p)
    p.add_argument("--report", help="write the Markdown report here")
    p.add_argument("--dry-run", action="store_true", help="print the Slack payload instead of sending")
    sub.add_parser("demo", help="v1 passes, degraded v2 is caught (simulated)")
    args = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "baseline": cmd_baseline, "check": cmd_check, "demo": cmd_demo}[args.cmd](args)
    except (DatasetError, ScorerError, FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR
