"""Validation for LabelImg Pascal VOC XML files."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class Box:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def read_voc(path: Path, allowed_labels: set[str]) -> tuple[tuple[int, int], list[Box]]:
    root = ElementTree.parse(path).getroot()
    width, height = int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid image dimensions")
    boxes = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip().lower()
        coords = [int(obj.findtext(f"bndbox/{key}", "-1")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if label not in allowed_labels:
            raise ValueError(f"{path}: unknown label {label!r}")
        if not (0 <= coords[0] < coords[2] <= width and 0 <= coords[1] < coords[3] <= height):
            raise ValueError(f"{path}: bounding box is outside the image")
        boxes.append(Box(label, *coords))
    if not boxes:
        raise ValueError(f"{path}: annotation has no objects")
    return (width, height), boxes
