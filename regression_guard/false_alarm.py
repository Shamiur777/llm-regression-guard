"""False-alarm check (SIMULATED): rerun the UNCHANGED prompt with different noise seeds and
count how often the detector alarms, with and without the statistical guard. Also counts
how often the degraded prompt is caught.

    python -m regression_guard.false_alarm --runs 200
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .cli import REPO_ROOT
from .dataset import load_dataset
from .detect import compare, load_rules
from .feature import load_feature
from .runner import run_feature


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--noise", type=float, default=None, help="override noise rate (default: config)")
    args = ap.parse_args(argv)

    cfg = REPO_ROOT / "config/feature.json"
    rules = load_rules(REPO_ROOT / "config/rules.json")
    no_guard = replace(rules, stat_guard=False)
    base_feature = load_feature(cfg, REPO_ROOT, seed=0, noise=args.noise)
    cases = load_dataset(base_feature.dataset)
    baseline = run_feature(base_feature, cases)
    bad = [REPO_ROOT / "prompts/examples/triage_v2_degraded.txt"]

    guarded = unguarded = caught = 0
    for seed in range(1, args.runs + 1):
        same = run_feature(load_feature(cfg, REPO_ROOT, seed=seed, noise=args.noise), cases)
        guarded += compare(baseline, same, rules).regressed
        unguarded += compare(baseline, same, no_guard).regressed
        degraded = run_feature(load_feature(cfg, REPO_ROOT, bad, seed=seed, noise=args.noise), cases)
        caught += compare(baseline, degraded, rules).regressed

    noise = args.noise if args.noise is not None else json.loads(cfg.read_text())["provider"]["noise"]
    print(f"SIMULATED, {len(cases)} cases, noise={noise}, {args.runs} runs per row")
    print(f"unchanged prompt, stat guard ON : {guarded}/{args.runs} false alarms")
    print(f"unchanged prompt, stat guard OFF: {unguarded}/{args.runs} false alarms")
    print(f"degraded prompt v2, guard ON    : caught {caught}/{args.runs}")


if __name__ == "__main__":
    main()
