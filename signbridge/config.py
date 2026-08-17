"""Single source of truth for labels and TensorFlow workspace paths."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "Tensorflow" / "workspace"
LABELS = ("hello", "thanks", "yes", "no", "iloveyou")


@dataclass(frozen=True)
class Paths:
    images: Path = WORKSPACE / "images"
    annotations: Path = WORKSPACE / "annotations"
    models: Path = WORKSPACE / "models"
    pretrained: Path = WORKSPACE / "pre-trained-models"
    exported: Path = WORKSPACE / "exported-models"

    @property
    def label_map(self) -> Path:
        return self.annotations / "label_map.pbtxt"


PATHS = Paths()
