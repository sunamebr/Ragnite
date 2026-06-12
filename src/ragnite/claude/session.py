"""Invoke-mode session state and configuration.

``.ragnite/session.json`` — mutable runtime state (active flag + counters).
``.ragnite/config.toml``  — user-edited knobs ([invoke] section).
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_STATS = {
    "prompts_enriched": 0,
    "episodes_learned": 0,
    "files_reindexed": 0,
    "cache_invalidations": 0,
    "searches_redirected": 0,
}


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk up from ``start`` looking for .ragnite, then .git; fallback: start."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".ragnite").is_dir():
            return candidate
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


class SessionState:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / ".ragnite" / "session.json"
        self.data: dict[str, Any] = {
            "active": False,
            "installed_at": None,
            "activated_at": None,
            "stats": dict(DEFAULT_STATS),
        }
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.update(loaded)
                self.data["stats"] = {**DEFAULT_STATS, **loaded.get("stats", {})}
            except (json.JSONDecodeError, OSError):
                pass  # corrupted state file -> start fresh, never crash a hook

    @property
    def active(self) -> bool:
        return bool(self.data.get("active"))

    def set_active(self, active: bool) -> None:
        self.data["active"] = active
        self.data["activated_at"] = time.time() if active else self.data.get("activated_at")
        self.save()

    def bump(self, stat: str, n: int = 1) -> None:
        self.data.setdefault("stats", dict(DEFAULT_STATS))
        self.data["stats"][stat] = self.data["stats"].get(stat, 0) + n
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


@dataclass
class InvokeConfig:
    strict: bool = False  # strict=True lets PreToolUse deny broad searches answered "direct"
    budget_tokens: int = 1200  # context budget per prompt injection
    min_confidence: float = 0.25  # below this, inject nothing (avoid noise)
    max_briefing_decisions: int = 8
    learn_from_bash: bool = True


def load_invoke_config(root: str | Path) -> InvokeConfig:
    config_file = Path(root) / ".ragnite" / "config.toml"
    cfg = InvokeConfig()
    if not config_file.exists():
        return cfg
    try:
        section = tomllib.loads(config_file.read_text(encoding="utf-8")).get("invoke", {})
    except (tomllib.TOMLDecodeError, OSError):
        return cfg
    cfg.strict = bool(section.get("strict", cfg.strict))
    cfg.budget_tokens = int(section.get("budget_tokens", cfg.budget_tokens))
    cfg.min_confidence = float(section.get("min_confidence", cfg.min_confidence))
    cfg.max_briefing_decisions = int(section.get("max_briefing_decisions", cfg.max_briefing_decisions))
    cfg.learn_from_bash = bool(section.get("learn_from_bash", cfg.learn_from_bash))
    return cfg
