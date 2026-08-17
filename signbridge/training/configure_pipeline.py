"""Patch an SSD MobileNet pipeline.config for the five-sign dataset."""

import argparse
from pathlib import Path
from signbridge.config import LABELS, PATHS


def replacements(text: str, checkpoint: Path, model_dir: Path) -> str:
    import re
    text = re.sub(r"num_classes:\s*\d+", f"num_classes: {len(LABELS)}", text, count=1)
    text = re.sub(r'fine_tune_checkpoint:\s*"[^"]*"', f'fine_tune_checkpoint: "{checkpoint.as_posix()}"', text, count=1)
    text = re.sub(r'fine_tune_checkpoint_type:\s*"[^"]*"', 'fine_tune_checkpoint_type: "detection"', text, count=1)
    label_path = PATHS.label_map.as_posix()
    train_record = (PATHS.annotations / "train.record").as_posix()
    test_record = (PATHS.annotations / "test.record").as_posix()
    text = re.sub(r'label_map_path:\s*"[^"]*"', f'label_map_path: "{label_path}"', text)
    paths = iter((train_record, test_record))
    text = re.sub(r'input_path:\s*"[^"]*"', lambda _: f'input_path: "{next(paths)}"', text, count=2)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--model-name", default="signbridge_ssd_mobilenet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    target = PATHS.models / args.model_name / "pipeline.config"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(replacements(args.pipeline.read_text(), args.checkpoint, target.parent), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
