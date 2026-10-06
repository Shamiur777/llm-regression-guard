"""Slack incoming-webhook alert: build blocks, send with urllib."""
from __future__ import annotations

import json
import os
import urllib.request

ENV_VAR = "SLACK_WEBHOOK_URL"


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_payload(cmp, meta: dict, run_url: str | None = None, worst: int = 3) -> dict:
    o = cmp.overall
    shown = [f"- {_esc(f.message)}" for f in cmp.findings[:6]]
    if len(cmp.findings) > 6:
        shown.append(f"- ...and {len(cmp.findings) - 6} more in the full report")
    title = f"LLM regression detected: {meta.get('feature', 'feature')}"
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Overall score*\n{o['baseline']['score']:.1%} -> {o['current']['score']:.1%} ({o['delta']:+.1%})"},
            {"type": "mrkdwn", "text": f"*Flips*\n{len(cmp.flips)} pass->fail, {len(cmp.fixes)} fail->pass (p={cmp.p_overall:.4f})"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": "*Findings*\n" + "\n".join(shown)}},
    ]
    if cmp.flips:
        rows = [f"- `{_esc(c.id)}`: {_esc(' '.join(c.input.split())[:90])}" for _, c in cmp.flips[:worst]]
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Worst flips*\n" + "\n".join(rows)}})
    ctx = f"Prompt {meta.get('prompt_sha', '?')} vs baseline {meta.get('baseline_prompt_sha', '?')}"
    if meta.get("simulated"):
        ctx += " | simulated mock provider"
    if run_url:
        ctx += f" | <{run_url}|CI run>"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": ctx}]})
    return {"text": f"{title} ({o['baseline']['score']:.1%} -> {o['current']['score']:.1%})", "blocks": blocks}


def send(payload: dict, url: str | None = None, timeout: float = 10.0) -> int:
    """POST the payload; returns the HTTP status. The URL comes from the argument or $SLACK_WEBHOOK_URL."""
    url = url or os.environ.get(ENV_VAR)
    if not url:
        raise RuntimeError(f"{ENV_VAR} is not set")
    if not url.startswith(("https://", "http://")):
        raise ValueError("webhook URL must be http(s)")
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme checked above
        return resp.status
