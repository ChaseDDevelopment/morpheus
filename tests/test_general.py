import copy
from typing import Generator, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from models.dataclasses import MorpheusImage, MorpheusLabel
from tests import constants
from tests.constants import get_mocked_morpheus_images
from utils.general import get_class_names


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
def images(mock_path) -> List[MorpheusImage]:
    """Fixture that creates a list of MorpheusImages for testing"""
    label1 = MorpheusLabel(name="car", box=copy.deepcopy(constants.EXPECTED_BOXES[0]))
    label2 = MorpheusLabel(name="train", box=copy.deepcopy(constants.EXPECTED_BOXES[1]))
    return copy.deepcopy(
        [
            MorpheusImage(
                name="image.jpg",
                path=mock_path,
                image=np.full((600, 400, 3), 42, dtype=np.uint8),
                labels=[label1, label2],
                image_size=(600, 400),
                image_depth=3,
            ),
            MorpheusImage(
                name=constants.IMAGE_NAME,
                path=mock_path,
                image=np.full((600, 400, 3), 42, dtype=np.uint8),
                labels=[label2, label1],
                image_size=(600, 400),
                image_depth=3,
            ),
        ]
    )


def test_get_class_names(images):
    """Test the get_class_names function"""
    assert get_class_names(images) == constants.CLASS_NAMES


def test_get_class_names_empty():
    """Test the get_class_names function with empty list"""
    assert get_class_names([]) == []


def test_match_images_to_label_files(tmp_path):
    """Test the match_images_to_label_files function"""
    matched_images = get_mocked_morpheus_images(tmp_path)
    matched_images.sort(key=lambda x: x.name)
    # Make sure there are three images
    assert len(matched_images) == 3
    # Make sure the image names are correct
    assert matched_images[0].name == constants.IMAGE_NAME
    # Make sure the image paths are correct
    assert matched_images[0].path == tmp_path / constants.IMAGE_NAME
    # Make sure the image sizes are correct
    assert matched_images[0].image_size == (720, 540)
    # Make sure the image depths are correct
    assert matched_images[0].image_depth == 3
