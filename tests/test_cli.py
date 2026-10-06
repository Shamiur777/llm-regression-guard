import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from regression_guard.cli import main
from regression_guard.feature import load_feature
from regression_guard.providers import AnthropicProvider, MockProvider

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "prompts/examples/triage_v2_degraded.txt"


def base_args(tmp_path):
    return ["--root", str(ROOT), "--config", str(ROOT / "config/feature.json"),
            "--rules", str(ROOT / "config/rules.json"), "--baseline", str(tmp_path / "baseline.json")]


@pytest.fixture(autouse=True)
def no_slack_env(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)


def test_baseline_then_check_unchanged_exits_zero(tmp_path, capsys):
    assert main(["baseline", *base_args(tmp_path)]) == 0
    doc = json.loads((tmp_path / "baseline.json").read_text())
    assert doc["metrics"]["overall"]["n"] == len(doc["results"]) >= 60
    assert main(["check", *base_args(tmp_path), "--seed", "1"]) == 0  # sampling noise only
    assert "No regression" in capsys.readouterr().out


def test_check_degraded_prompt_exits_one_with_report_and_dry_run(tmp_path, capsys):
    main(["baseline", *base_args(tmp_path)])
    report = tmp_path / "report.md"
    code = main(["check", *base_args(tmp_path), "--prompt", str(V2), "--report", str(report), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1
    assert "REGRESSION DETECTED" in report.read_text(encoding="utf-8")
    assert "Slack payload (dry run)" in out and '"blocks"' in out


def test_check_sends_to_webhook_env_only_on_regression(tmp_path, monkeypatch, capsys):
    sent = []
    monkeypatch.setattr("regression_guard.slack.send", lambda payload, url=None: sent.append(payload) or 200)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "http://example.invalid/hook")
    main(["baseline", *base_args(tmp_path)])
    assert main(["check", *base_args(tmp_path)]) == 0 and sent == []
    assert main(["check", *base_args(tmp_path), "--prompt", str(V2)]) == 1 and len(sent) == 1


def test_slack_failure_does_not_change_exit_code(tmp_path, monkeypatch, capsys):
    def boom(payload, url=None):
        raise OSError("network down")
    monkeypatch.setattr("regression_guard.slack.send", boom)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "http://example.invalid/hook")
    main(["baseline", *base_args(tmp_path)])
    assert main(["check", *base_args(tmp_path), "--prompt", str(V2)]) == 1
    assert "Slack alert failed" in capsys.readouterr().err


def test_missing_baseline_and_bad_dataset_exit_two(tmp_path, capsys):
    assert main(["check", *base_args(tmp_path)]) == 2
    assert "baseline not found" in capsys.readouterr().err
    cfg = json.loads((ROOT / "config/feature.json").read_text())
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n')
    cfg["dataset"] = str(bad)
    cfg_path = tmp_path / "feature.json"
    cfg_path.write_text(json.dumps(cfg))
    args = base_args(tmp_path)
    args[args.index("--config") + 1] = str(cfg_path)
    assert main(["run", *args]) == 2
    assert "line 1" in capsys.readouterr().err


def test_run_prints_metrics(tmp_path, capsys):
    assert main(["run", *base_args(tmp_path)]) == 0
    assert "overall: score" in capsys.readouterr().out


def test_demo_catches_v2(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "SIMULATED" in out and "should NOT alert" in out and "REGRESSION DETECTED" in out


def test_prompt_change_changes_simulated_outputs():
    good = load_feature(ROOT / "config/feature.json", ROOT, noise=0)
    bad = load_feature(ROOT / "config/feature.json", ROOT, [V2], noise=0)
    ticket = "Please cancel my subscription before the next renewal."
    assert json.loads(good.generate(ticket))["category"] == "cancellation"
    assert json.loads(bad.generate(ticket))["category"] == "other"
    assert isinstance(good.provider, MockProvider)


def test_anthropic_provider_with_fake_client(monkeypatch):
    calls = {}

    class Messages:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi "),
                                            SimpleNamespace(type="text", text="there")])

    p = AnthropicProvider(client=SimpleNamespace(messages=Messages()))
    assert p.complete("sys", "user") == "hi there"
    assert calls["system"] == "sys" and calls["messages"][0]["content"] == "user"
    assert calls["model"] == "claude-sonnet-5-5"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        AnthropicProvider().complete("s", "u")
