"""Run exported TensorFlow detection against a live webcam stream."""

import argparse
from pathlib import Path

from signbridge.config import PATHS
from signbridge.utils.labels import parse_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=PATHS.exported / "saved_model")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import tensorflow as tf

    labels = parse_label_map(PATHS.label_map.read_text())
    detect = tf.saved_model.load(str(args.model / "saved_model")).signatures["serving_default"]
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = detect(tf.convert_to_tensor(rgb[None, ...]))
            boxes = output["detection_boxes"][0].numpy()
            classes = output["detection_classes"][0].numpy().astype(int)
            scores = output["detection_scores"][0].numpy()
            height, width = frame.shape[:2]
            for box, class_id, score in zip(boxes, classes, scores):
                if score < args.threshold:
                    continue
                y1, x1, y2, x2 = box
                p1, p2 = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
                color = (76, 230, 154)
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, f"{labels.get(class_id, class_id)} {score:.0%}", (p1[0], max(25, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("SignBridge real-time detection — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
