"""Prompt template registry: loads .txt templates, caches, renders with variables."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_CACHE: dict[str, str] = {}


def _load_template(name: str) -> str:
    if name not in _CACHE:
        path = _TEMPLATE_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        _CACHE[name] = path.read_text(encoding="utf-8")
    return _CACHE[name]


def render(name: str, **kwargs: str) -> str:
    """Load and render a prompt template with the given variables."""
    template = _load_template(name)
    return template.format(**kwargs)


def get_raw(name: str) -> str:
    """Get raw template without rendering."""
    return _load_template(name)


def clear_cache() -> None:
    _CACHE.clear()
