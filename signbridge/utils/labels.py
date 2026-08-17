"""TensorFlow Object Detection label-map helpers."""

import re
from collections.abc import Iterable

ITEM = re.compile(r"item\s*\{\s*name:\s*['\"]([^'\"]+)['\"]\s*id:\s*(\d+)\s*\}", re.S)


def render_label_map(labels: Iterable[str]) -> str:
    normalized = [label.strip().lower() for label in labels]
    if not normalized or any(not label for label in normalized):
        raise ValueError("at least one non-empty label is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must be unique")
    return "\n".join(
        f"item {{\n  name: '{label}'\n  id: {index}\n}}" for index, label in enumerate(normalized, 1)
    ) + "\n"


def parse_label_map(text: str) -> dict[int, str]:
    parsed = {int(identifier): name for name, identifier in ITEM.findall(text)}
    if not parsed:
        raise ValueError("label map contains no valid items")
    return parsed
