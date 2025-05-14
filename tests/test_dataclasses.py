# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""Tests for the dataclasses module."""
import copy
from typing import Generator
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from models.dataclasses import (
    MorpheusBoundingBox,
    MorpheusLabel,
    MorpheusImage,
)
from tests import constants


@pytest.fixture
def box() -> MorpheusBoundingBox:
    """Fixture that creates a bounding box for testing."""
    return copy.deepcopy(constants.EXPECTED_BOXES[0])


def test_width(box):
    """Test the width() method of MorpheusBoundingBox."""
    assert box.width() == 200


def test_height(box):
    """Test the height() method of MorpheusBoundingBox."""
    assert box.height() == 200


def test_area(box):
    """Test the area() method of MorpheusBoundingBox."""
    assert box.area() == 40000


def test_center(box):
    """Test the center() method of MorpheusBoundingBox."""
    assert box.center() == (200, 300)


def test_box_as_dict(box):
    """Test the as_dict() method of the MorpheusBoundingBox class."""
    assert box.as_dict() == constants.EXPECTED_BOXES[0].as_dict()


@pytest.fixture
def label() -> MorpheusLabel:
    """Fixture that creates a label for testing."""
    return MorpheusLabel(name="car", box=constants.EXPECTED_BOXES[0])


def test_name(label):
    """Test the name attribute of Label."""
    assert label.name == "car"


def test_boxes(label):
    """Test the boxes attribute of Label."""
    assert label.box == constants.EXPECTED_BOXES[0]


def test_label_as_dict(label):
    """Test the as_dict() method of Label."""
    assert label.as_dict() == constants.LABEL_AS_DICTIONARY


@pytest.fixture
def mock_path() -> Generator[Mock, None, None]:
    """Fixture that creates a mocked Path Object for testing"""
    mock_path = patch(
        "pathlib.Path",
        autospec=True,
        name=constants.IMAGE_NAME,
        exists=Mock(return_value=True),
        open=Mock(return_value="some content"),
    )
    yield mock_path


@pytest.fixture
def image(mock_path) -> MorpheusImage:
    """Fixture that creates an image for testing"""
    return copy.deepcopy(
        MorpheusImage(
            name="image.jpg",
            path=mock_path,
            image=np.full((600, 400, 3), 42, dtype=np.uint8),
            labels=[
                MorpheusLabel(
                    name="car", box=copy.deepcopy(constants.EXPECTED_BOXES[0])
                ),
                MorpheusLabel(
                    name="car", box=copy.deepcopy(constants.EXPECTED_BOXES[1])
                ),
            ],
            image_size=(600, 400),
            image_depth=3,
        )
    )


def test_get_labels(image):
    """Test the get_labels() method of MorpheusImage."""
    assert image.get_labels() == constants.IMAGE_LABELS_LIST_AS_DICT


def test_flip_horizontally(image):
    """Test the flip_horizontally() method of MorpheusImage."""
    original_image = image.image.copy()
    original_labels = [label.box.as_dict() for label in image.labels]
    image.flip_horizontally()
    np.testing.assert_array_equal(image.image, np.flip(original_image, axis=1))
    for i, label in enumerate(image.labels):
        assert label.box.xmin == max(
            0, original_image.shape[1] - original_labels[i]["xmax"]
        )
        assert label.box.xmax == min(
            original_image.shape[1],
            original_image.shape[1] - original_labels[i]["xmin"],
        )


def test_flip_vertically(image):
    """Test the flip_vertically() method of MorpheusImage."""
    original_image = image.image.copy()
    original_labels = [label.box.as_dict() for label in image.labels]
    image.flip_vertically()
    np.testing.assert_array_equal(image.image, np.flip(original_image, axis=0))
    for i, label in enumerate(image.labels):
        assert label.box.ymin == max(
            0, original_image.shape[0] - original_labels[i]["ymax"]
        )
        assert label.box.ymax == min(
            original_image.shape[0],
            original_image.shape[0] - original_labels[i]["ymin"],
        )


def test_load_image_to_memory(image):
    """Test the load_image_to_memory() method of MorpheusImage."""
    cv2.imread = Mock()
    image.load_image_to_memory()
    assert cv2.imread.called
    assert image.image is not None


def test_write_image_to_disk(image):
    """Test the write_image_to_disk() method of MorpheusImage."""
    cv2.imwrite = Mock()
    image.write_image_to_disk()
    assert cv2.imwrite.called
    assert image.image is None


def test_resize_image(image):
    """Test the resize_image() method of MorpheusImage."""
    cv2.imread = Mock(return_value=image.image)
    cv2.imwrite = Mock()
    original_image = image.image.copy()
    original_labels = [label.box.as_dict() for label in image.labels]
    image.load_image_to_memory()
    image.resize_image(100, 100)
    image.write_image_to_disk()
    assert cv2.imread.called
    scale = min(100 / original_image.shape[1], 100 / original_image.shape[0])
    scale = min(scale, 1.0)
    assert image.image is None
    assert image.image_size == (67, 100)
    for i, label in enumerate(image.labels):
        assert label.box.xmin == int(original_labels[i]["xmin"] * scale)
        assert label.box.xmax == int(original_labels[i]["xmax"] * scale)
        assert label.box.ymin == int(original_labels[i]["ymin"] * scale)
        assert label.box.ymax == int(original_labels[i]["ymax"] * scale)
    assert cv2.imwrite.called
