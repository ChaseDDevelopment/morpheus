# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
import numpy as np

from models.dataclasses import MorpheusBoundingBox
from utils.general import match_images_to_label_files

LABEL_AS_DICTIONARY = {
    "name": "car",
    "box": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400},
}

IMAGE_LABELS_LIST_AS_DICT = {
    0: {
        "name": "car",
        "box": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400},
    },
    1: {
        "name": "car",
        "box": {"xmin": 400, "ymin": 200, "xmax": 600, "ymax": 400},
    },
}

ROTATED_ONCE_IMAGE = np.full((400, 600, 3), 42, dtype=np.uint8)

EXPECTED_BOXES = [
    MorpheusBoundingBox(xmin=100, ymin=200, xmax=300, ymax=400),
    MorpheusBoundingBox(xmin=400, ymin=200, xmax=600, ymax=400),
]
HORIZONTAL_FLIPPED_BOXES = [
    MorpheusBoundingBox(xmin=100, ymin=200, xmax=300, ymax=400),
    MorpheusBoundingBox(xmin=0, ymin=200, xmax=0, ymax=400),
]
VERTICAL_FLIPPED_BOXES = [
    MorpheusBoundingBox(xmin=16, ymin=34, xmax=50, ymax=67),
    MorpheusBoundingBox(xmin=66, ymin=34, xmax=100, ymax=67),
]

VERTICAL_AND_HORIZONTAL_FLIPPED_BOXES = [
    MorpheusBoundingBox(xmin=17, ymin=34, xmax=51, ymax=67),
    MorpheusBoundingBox(xmin=0, ymin=34, xmax=1, ymax=67),
]

DATA_YAML_FILENAME = "data.yaml"
DATA_VAL_YAML_CONTENT = "val: ../valid/images"
DATA_TRAIN_YAML_CONTENT = "train: ../train/images"
DATA_TEST_YAML_CONTENT = "test: ../test/images"

CLASS_NAMES = ["car", "train"]
REMAPPED_CLASS_NAME = ["car"]

XML_CONTENT = """
    <annotation>
    <filename>image{}.jpg</filename>
    <size>
    <width>720</width>
    <height>540</height>
    <depth>3</depth>
    </size>
    <object>
    <name>car</name>
    <bndbox>
    <xmin>252</xmin>
    <ymin>143</ymin>
    <xmax>291</xmax>
    <ymax>255</ymax>
    </bndbox>
    </object>
    </annotation>
    """
IMAGE_NAME = "image1.jpg"


def get_mocked_morpheus_images(tmp_path):
    for i in range(1, 4):
        (tmp_path / f"image{i}.jpg").write_text("some content")
        (tmp_path / f"image{i}.xml").write_text(XML_CONTENT.format(i))
    return match_images_to_label_files(tmp_path)
