"""Support-ticket triage: SIMULATED model behaviour plus a custom scorer.

The simulator is not a language model. It reads the prompt text it is given
(keyword lists per category, priority words, a "JSON only" instruction, a reply
instruction) and behaves accordingly, so editing the prompt genuinely changes
the outputs. That is what lets the demo show a regression without an API key.
"""
from __future__ import annotations

import json
import random
import re

ALL_CATEGORIES = ["billing", "login", "shipping", "bug", "cancellation", "feature_request", "other"]
SLA_HOURS = {"high": 4, "normal": 24, "low": 72}
REPLIES = {
    "billing": "Thanks for contacting billing support. We are reviewing your invoice and will confirm any refund within one business day.",
    "login": "Sorry you cannot get in. We have sent steps to reset your password and restore access to your account.",
    "shipping": "Sorry about the wait. We are checking the tracking details with the courier and the delivery status will follow shortly.",
    "bug": "Thanks for reporting this bug. An engineer is looking into a fix and we will update you soon.",
    "cancellation": "We are sorry to see you go. We have started your cancel request and will confirm by email.",
    "feature_request": "Thanks for the suggestion. We have passed this feature idea to the product team for the roadmap.",
    "other": "Thanks for your message. A member of our team will take a look and get back to you.",
}
_LINE = re.compile(r"^([a-z_]+):\s*(.+)$")


def parse_prompt(system: str) -> dict:
    """Pull the 'knowledge' the simulated model gets from the prompt text."""
    categories: dict[str, list[str]] = {}
    priority: dict[str, list[str]] = {}
    mode = None
    for line in system.splitlines():
        line = line.strip()
        if line.startswith("##"):
            mode = line.lstrip("# ").lower()
            continue
        m = _LINE.match(line)
        if not m or mode not in ("categories", "priority"):
            continue
        words = [w.strip().lower() for w in m.group(2).split(",") if w.strip()]
        (categories if mode == "categories" else priority)[m.group(1)] = words
    low = system.lower()
    return {
        "categories": categories,
        "priority": priority,
        "json_only": "respond with json only" in low,
        "wants_reply": "short reply" in low,
    }


def make_simulator(noise: float = 0.0):
    """Build a handler for MockProvider. `noise` is the chance a category is wrong,
    standing in for sampling nondeterminism."""

    def handler(system: str, user: str, rng: random.Random) -> str:
        spec = parse_prompt(system)
        text = user.lower()
        best, best_hits = "other", 0
        for name, words in spec["categories"].items():
            hits = sum(1 for w in words if w in text)
            if hits > best_hits:
                best, best_hits = name, hits
        if noise and rng.random() < noise:
            best = rng.choice([c for c in ALL_CATEGORIES if c != best])
        prio = "normal"
        if any(w in text for w in spec["priority"].get("high", [])):
            prio = "high"
        elif any(w in text for w in spec["priority"].get("low", [])):
            prio = "low"
        body = json.dumps({
            "category": best,
            "priority": prio,
            "sla_hours": SLA_HOURS[prio],
            "reply": REPLIES[best] if spec["wants_reply"] else "",
        })
        # Without a "JSON only" instruction the simulated model gets chatty on longer tickets.
        if not spec["json_only"] and len(user.split()) > 20:
            return "Sure, here is the classification you asked for: " + body
        return body

    return handler


def reply_is_concise(case_input: str, output: str, expected) -> tuple[bool, str]:
    """Example custom scorer: reply must be short and contain no template placeholders."""
    try:
        reply = json.loads(output[output.index("{"):]).get("reply", "")
    except (ValueError, AttributeError):
        return False, "no parsable JSON"
    ok = len(reply) <= 220 and "{" not in reply
    return ok, f"reply length {len(reply)}"
