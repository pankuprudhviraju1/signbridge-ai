"""Collect balanced webcam images for every sign, as demonstrated in the reference video."""

import argparse
import time
from datetime import datetime, timezone

from signbridge.config import LABELS, PATHS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect sign images from a webcam")
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--images-per-label", type=int, default=15)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--countdown", type=float, default=3)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.images_per_label < 1:
        raise SystemExit("--images-per-label must be positive")
    import cv2

    output = PATHS.images / args.split
    output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    try:
        for label in [value.strip().lower() for value in args.labels]:
            deadline = time.monotonic() + args.countdown
            while time.monotonic() < deadline:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed")
                remaining = max(1, int(deadline - time.monotonic()) + 1)
                cv2.putText(frame, f"Get ready: {label} ({remaining})", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("SignBridge image collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
            for index in range(args.images_per_label):
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed")
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                target = output / f"{label}.{timestamp}.{index:03}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"Could not write {target}")
                cv2.putText(frame, f"{label}: {index + 1}/{args.images_per_label}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("SignBridge image collection", frame)
                if cv2.waitKey(max(1, int(args.interval * 1000))) & 0xFF == ord("q"):
                    return
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
