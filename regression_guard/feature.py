"""A Feature = prompt file(s) + model config + a Provider."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .providers import Provider, build_provider


@dataclass
class Feature:
    name: str
    prompt_files: list
    system_prompt: str
    model: dict
    provider: Provider
    dataset: Path
    judge: Provider | None = None

    def generate(self, user_input: str) -> str:
        return self.provider.complete(self.system_prompt, user_input)

    @property
    def prompt_sha(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> dict:
        return {"feature": self.name, "prompt_files": [str(p) for p in self.prompt_files],
                "prompt_sha": self.prompt_sha, "model": self.model,
                "provider": type(self.provider).__name__}


def load_feature(config_path, root=".", prompt_override=None, seed=None, noise=None) -> Feature:
    """Read a feature config. Paths inside it are relative to `root`.

    `prompt_override` swaps the prompt files (used to test a candidate prompt).
    """
    root = Path(root)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    files = [Path(p) for p in (prompt_override or [root / p for p in cfg["prompt_files"]])]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"prompt file(s) not found: {', '.join(missing)}")
    text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in files)
    judge = build_provider(cfg["judge"]) if cfg.get("judge") else None
    return Feature(
        name=cfg["name"],
        prompt_files=files,
        system_prompt=text,
        model=cfg.get("model", {}),
        provider=build_provider(cfg.get("provider", {}), seed=seed, noise=noise),
        dataset=root / cfg["dataset"],
        judge=judge,
    )
