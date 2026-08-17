"""Create the label map and validate LabelImg XML before TFRecord conversion."""

import argparse
from pathlib import Path

from signbridge.config import LABELS, PATHS
from signbridge.utils.annotations import read_voc
from signbridge.utils.labels import render_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="validate every XML annotation")
    args = parser.parse_args()
    for path in (PATHS.images / "train", PATHS.images / "test", PATHS.annotations, PATHS.models, PATHS.pretrained, PATHS.exported):
        path.mkdir(parents=True, exist_ok=True)
    PATHS.label_map.write_text(render_label_map(LABELS), encoding="utf-8")
    if args.validate:
        xml_files = list(PATHS.images.glob("**/*.xml"))
        for xml in xml_files:
            read_voc(xml, set(LABELS))
        print(f"Validated {len(xml_files)} Pascal VOC annotations")
    print(f"Label map: {PATHS.label_map}")


if __name__ == "__main__":
    main()
