import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from regression_guard import slack
from regression_guard.detect import compare
from regression_guard.report import render_markdown
from test_detect import flipped, run


@pytest.fixture
def regression():
    base_flags = [True] * 50
    cmp = compare(run(base_flags), run(flipped(base_flags, down=range(15))))
    assert cmp.regressed
    return cmp, {"feature": "demo feature", "prompt_sha": "abc", "baseline_prompt_sha": "def", "simulated": True}


@pytest.fixture
def webhook():
    """Local HTTP server that records POSTs."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, self.headers["Content-Type"], json.loads(body)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}/services/T/B/X", seen
    server.shutdown()
    server.server_close()


def test_payload_structure_and_escaping(regression):
    cmp, meta = regression
    p = slack.build_payload(cmp, meta, run_url="https://ci.example/run/1")
    assert p["text"].startswith("LLM regression detected")
    types = [b["type"] for b in p["blocks"]]
    assert types[0] == "header" and types[-1] == "context"
    flat = json.dumps(p)
    assert "simulated" in flat and "https://ci.example/run/1" in flat
    assert p["blocks"][0]["text"]["text"].endswith("demo feature")


def test_mrkdwn_values_are_escaped():
    from regression_guard.detect import Comparison, Finding
    cmp = Comparison(True, findings=[Finding("r", "a -> b <@channel>")],
                     overall={"baseline": {"score": 1.0}, "current": {"score": 0.5}, "delta": -0.5})
    text = slack.build_payload(cmp, {})["blocks"][2]["text"]["text"]
    assert "&gt;" in text and "<@channel>" not in text


def test_payload_caps_findings():
    from regression_guard.detect import Comparison, Finding
    cmp = Comparison(True, findings=[Finding("tag_drop", f"tag {i} fell") for i in range(10)],
                     overall={"baseline": {"score": 1.0}, "current": {"score": 0.5}, "delta": -0.5})
    text = slack.build_payload(cmp, {})["blocks"][2]["text"]["text"]
    assert text.count("tag ") == 6 and "4 more" in text


def test_send_posts_json_to_local_server(regression, webhook):
    url, seen = webhook
    cmp, meta = regression
    status = slack.send(slack.build_payload(cmp, meta), url=url)
    assert status == 200
    path, ctype, body = seen[0]
    assert path == "/services/T/B/X" and ctype == "application/json"
    assert body["blocks"][0]["type"] == "header"


def test_send_reads_env_and_refuses_without_url(regression, webhook, monkeypatch):
    url, seen = webhook
    cmp, meta = regression
    monkeypatch.setenv("SLACK_WEBHOOK_URL", url)
    assert slack.send(slack.build_payload(cmp, meta)) == 200 and len(seen) == 1
    monkeypatch.delenv("SLACK_WEBHOOK_URL")
    with pytest.raises(RuntimeError, match="not set"):
        slack.send({})
    with pytest.raises(ValueError):
        slack.send({}, url="file:///etc/passwd")


def test_markdown_report_lists_worst_flips(regression):
    cmp, meta = regression
    md = render_markdown(cmp, meta, worst=3)
    assert "REGRESSION DETECTED" in md and "SIMULATED" in md
    assert md.count("### `c") == 3 and "Expected:" in md and "Actual:" in md and "Input:" in md
