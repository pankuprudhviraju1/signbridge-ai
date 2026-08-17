PYTHON ?= python3
MODEL_NAME ?= signbridge_ssd_mobilenet
PIPELINE := Tensorflow/workspace/models/$(MODEL_NAME)/pipeline.config

.PHONY: test collect prepare records train export detect
test:
	pytest -q

collect:
	$(PYTHON) -m signbridge.collection.capture_images --images-per-label 15

prepare:
	$(PYTHON) -m signbridge.training.prepare_workspace --validate

records:
	$(PYTHON) -m signbridge.training.generate_tfrecord --images Tensorflow/workspace/images/train --output Tensorflow/workspace/annotations/train.record
	$(PYTHON) -m signbridge.training.generate_tfrecord --images Tensorflow/workspace/images/test --output Tensorflow/workspace/annotations/test.record

train:
	$(PYTHON) Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/$(MODEL_NAME) --pipeline_config_path=$(PIPELINE) --num_train_steps=10000

export:
	$(PYTHON) Tensorflow/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path=$(PIPELINE) --trained_checkpoint_dir=Tensorflow/workspace/models/$(MODEL_NAME) --output_directory=Tensorflow/workspace/exported-models/$(MODEL_NAME)

detect:
	$(PYTHON) -m signbridge.inference.webcam --model Tensorflow/workspace/exported-models/$(MODEL_NAME)
