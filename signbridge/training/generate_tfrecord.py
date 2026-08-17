"""Convert LabelImg Pascal VOC pairs into TensorFlow Object Detection TFRecords."""

import argparse
import hashlib
from pathlib import Path
from xml.etree import ElementTree

from signbridge.config import LABELS


def class_id(label: str) -> int:
    try:
        return LABELS.index(label.lower()) + 1
    except ValueError as error:
        raise ValueError(f"Unknown class {label!r}; expected one of {LABELS}") from error


def create_example(xml_path: Path):
    import tensorflow as tf
    from object_detection.utils import dataset_util

    root = ElementTree.parse(xml_path).getroot()
    image_path = xml_path.with_name(root.findtext("filename"))
    encoded = image_path.read_bytes()
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))
    xmins, xmaxs, ymins, ymaxs, names, ids = [], [], [], [], [], []
    for obj in root.findall("object"):
        label = obj.findtext("name").strip().lower(); box = obj.find("bndbox")
        xmins.append(float(box.findtext("xmin")) / width); xmaxs.append(float(box.findtext("xmax")) / width)
        ymins.append(float(box.findtext("ymin")) / height); ymaxs.append(float(box.findtext("ymax")) / height)
        names.append(label.encode()); ids.append(class_id(label))
    features = {
        "image/height": dataset_util.int64_feature(height), "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(image_path.name.encode()), "image/source_id": dataset_util.bytes_feature(image_path.name.encode()),
        "image/key/sha256": dataset_util.bytes_feature(hashlib.sha256(encoded).hexdigest().encode()), "image/encoded": dataset_util.bytes_feature(encoded),
        "image/format": dataset_util.bytes_feature(image_path.suffix.lstrip(".").encode()), "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs), "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs), "image/object/class/text": dataset_util.bytes_list_feature(names),
        "image/object/class/label": dataset_util.int64_list_feature(ids),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--images", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    import tensorflow as tf
    xml_files = sorted(args.images.glob("*.xml"))
    if not xml_files: raise SystemExit(f"No XML files found in {args.images}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(args.output)) as writer:
        for xml in xml_files: writer.write(create_example(xml).SerializeToString())
    print(f"Wrote {len(xml_files)} examples to {args.output}")


if __name__ == "__main__": main()
