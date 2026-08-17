from pathlib import Path
import pytest
from signbridge.training.generate_tfrecord import class_id
from signbridge.training.configure_pipeline import replacements
from signbridge.utils.annotations import read_voc
from signbridge.utils.labels import parse_label_map, render_label_map


def test_five_video_labels_are_stable():
    text = render_label_map(["hello", "thanks", "yes", "no", "iloveyou"])
    assert parse_label_map(text) == {1: "hello", 2: "thanks", 3: "yes", 4: "no", 5: "iloveyou"}


def test_duplicate_labels_are_rejected():
    with pytest.raises(ValueError): render_label_map(["hello", "Hello"])


def test_pascal_voc_annotation_is_valid():
    dimensions, boxes = read_voc(Path("fixtures/annotations/hello.xml"), {"hello"})
    assert dimensions == (640, 480) and boxes[0].label == "hello"


def test_class_ids_match_label_map():
    assert class_id("hello") == 1 and class_id("iloveyou") == 5
    with pytest.raises(ValueError): class_id("maybe")


def test_pipeline_configuration_rewrites_training_inputs(tmp_path):
    original = '''num_classes: 90\nfine_tune_checkpoint: "old"\nfine_tune_checkpoint_type: "classification"\nlabel_map_path: "old"\ninput_path: "train"\nlabel_map_path: "old"\ninput_path: "test"'''
    configured = replacements(original, tmp_path / "ckpt-0", tmp_path / "model")
    assert "num_classes: 5" in configured
    assert 'fine_tune_checkpoint_type: "detection"' in configured
    assert "train.record" in configured and "test.record" in configured
