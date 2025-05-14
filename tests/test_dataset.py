# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""Tests for the dataset module."""
import argparse
import copy
import shutil
from typing import List, Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest

import morpheus.dataset
from models.dataclasses import MorpheusImage, MorpheusLabel
from morpheus.dataset import (
    generate_dataset,
    parse_args,
    multiply_files,
)
from tests import constants
from tests.constants import get_mocked_morpheus_images


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


@pytest.fixture
def remapped_images(mock_path) -> List[MorpheusImage]:
    """Fixture that creates a list of MorpheusImages for testing"""
    label1 = MorpheusLabel(name="car", box=copy.deepcopy(constants.EXPECTED_BOXES[0]))
    return copy.deepcopy(
        [
            MorpheusImage(
                name="image.jpg",
                path=mock_path,
                image=np.full((600, 400, 3), 42, dtype=np.uint8),
                labels=[label1, label1],
                image_size=(600, 400),
                image_depth=3,
            ),
            MorpheusImage(
                name=constants.IMAGE_NAME,
                path=mock_path,
                image=np.full((600, 400, 3), 42, dtype=np.uint8),
                labels=[label1, label1],
                image_size=(600, 400),
                image_depth=3,
            ),
        ]
    )


@pytest.fixture
def arguments(tmp_path):
    """Return the arguments"""
    return argparse.Namespace(
        input=tmp_path / "input",
        output=tmp_path / "output",
        resize=640,
        multiply=1,
        augment=False,
    )


@patch("models.dataclasses.MorpheusImage.resize_image", new=Mock())
@patch("models.dataclasses.MorpheusImage.write_image_to_disk", new=Mock())
def test_generate_dataset(tmp_path, arguments):
    """Test the generate_dataset function"""
    matched_images = get_mocked_morpheus_images(tmp_path)
    matched_images.sort(key=lambda x: x.name)

    shutil.copy2 = Mock()

    output_directory = tmp_path / "output"
    generate_dataset(output_directory, constants.CLASS_NAMES, matched_images, arguments)
    # Ensure that the correct number of files were copied
    assert shutil.copy2.call_count == 3
    # Ensure that the correct directories were made
    for class_name in ["train", "valid", "test"]:
        assert (output_directory / class_name / "images").exists()
        assert (output_directory / class_name / "labels").exists()
    # Ensure that the correct data.yaml file was created
    with open(output_directory / constants.DATA_YAML_FILENAME, "r") as file:
        content = file.read()
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert constants.DATA_VAL_YAML_CONTENT in content
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert f"nc: {len(constants.CLASS_NAMES)}" in content
    assert f"names: {constants.CLASS_NAMES}" in content


@patch("models.dataclasses.MorpheusImage.resize_image", new=Mock())
@patch("models.dataclasses.MorpheusImage.write_image_to_disk", new=Mock())
def test_generate_dataset_with_multiply(tmp_path, arguments):
    arguments.multiply = 3
    """Test the generate_dataset function with multiply parameter"""
    matched_images = get_mocked_morpheus_images(tmp_path)
    matched_images.sort(key=lambda x: x.name)
    shutil.copy2 = Mock()
    output_directory = tmp_path / "output"
    # Multiply by 3
    generate_dataset(output_directory, constants.CLASS_NAMES, matched_images, arguments)
    # Ensure that the correct number of files were copied
    assert shutil.copy2.call_count == 9
    # Ensure that the correct directories were made
    for class_name in ["train", "valid", "test"]:
        assert (output_directory / class_name / "images").exists()
        assert (output_directory / class_name / "labels").exists()
    # Ensure that the correct data.yaml file was created
    with open(output_directory / constants.DATA_YAML_FILENAME, "r") as file:
        content = file.read()
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert constants.DATA_VAL_YAML_CONTENT in content
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert f"nc: {len(constants.CLASS_NAMES)}" in content
    assert f"names: {constants.CLASS_NAMES}" in content


@patch("models.dataclasses.MorpheusImage.resize_image", new=Mock())
@patch("models.dataclasses.MorpheusImage.write_image_to_disk", new=Mock())
@patch("morpheus.dataset.augment", new=Mock())
def test_generate_dataset_with_augment(tmp_path, arguments, images):
    """Test the generate_dataset function with multiply parameter"""
    morpheus.dataset.augment.return_value = images[0]
    arguments.multiply = 3
    arguments.augment = True
    arguments.flip_h = True
    arguments.flip_v = True
    matched_images = get_mocked_morpheus_images(tmp_path)
    matched_images.sort(key=lambda x: x.name)
    shutil.copy2 = Mock()
    output_directory = tmp_path / "output"
    # Multiply by 3
    generate_dataset(output_directory, constants.CLASS_NAMES, matched_images, arguments)
    # Ensure that the correct number of files were copied
    assert shutil.copy2.call_count == 9
    assert morpheus.dataset.augment.call_count == 9
    # Ensure that the correct directories were made
    for class_name in ["train", "valid", "test"]:
        assert (output_directory / class_name / "images").exists()
        assert (output_directory / class_name / "labels").exists()
    # Ensure that the correct data.yaml file was created
    with open(output_directory / constants.DATA_YAML_FILENAME, "r") as file:
        content = file.read()
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert constants.DATA_VAL_YAML_CONTENT in content
    assert constants.DATA_TRAIN_YAML_CONTENT in content
    assert f"nc: {len(constants.CLASS_NAMES)}" in content
    assert f"names: {constants.CLASS_NAMES}" in content


def test_multiply_files(tmp_path):
    """Test the multiply files function"""
    # Multiply by 1
    matched_images = get_mocked_morpheus_images(tmp_path)
    matched_images.sort(key=lambda x: x.name)
    assert len(matched_images) == 3
    assert matched_images[0].name == "image1.jpg"
    assert matched_images[1].name == "image2.jpg"
    assert matched_images[2].name == "image3.jpg"
    # Multiply by 3
    matched_images = multiply_files(matched_images, 3)
    matched_images.sort(key=lambda x: x.name)
    assert len(matched_images) == 9
    assert matched_images[2].name == "image1_2.jpg"
    assert matched_images[5].name == "image2_2.jpg"
    assert matched_images[8].name == "image3_2.jpg"


@patch("random.randint", new=Mock(side_effect=[1, 1, 2]))
@patch(
    "random.sample",
    new=Mock(side_effect=[["flip_h"], ["flip_v"], ["flip_h", "flip_v"]]),
)
def test_augment(images, arguments):
    """Test the augment function"""
    # Test with flip-h
    arguments.augment = True
    arguments.multiply = 3
    arguments.flip_h = True
    arguments.flip_v = True
    # Test with index 0 - horizontal flip augmentation
    original_image = copy.deepcopy(images[0])
    image = morpheus.dataset.augment(original_image, 0, arguments)
    for i, label in enumerate(image.labels):
        assert label.box == constants.HORIZONTAL_FLIPPED_BOXES[i]
    # Test with index 0 - vertical flip augmentation
    original_image = copy.deepcopy(images[0])
    # resize image to have vertical flip actually make changes
    original_image.resize_image(640, 100)
    image = morpheus.dataset.augment(original_image, 0, arguments)
    for i, label in enumerate(image.labels):
        assert label.box == constants.VERTICAL_FLIPPED_BOXES[i]
    # Test with index 0 - vertical and horizontal flip augmentation
    original_image = copy.deepcopy(images[0])
    # resize image to have vertical flip actually make changes
    original_image.resize_image(640, 100)
    image = morpheus.dataset.augment(original_image, 0, arguments)
    for i, label in enumerate(image.labels):
        assert label.box == constants.VERTICAL_AND_HORIZONTAL_FLIPPED_BOXES[i]
    # Test with index 2 - no augmentation
    original_image = copy.deepcopy(images[0])
    image = morpheus.dataset.augment(original_image, 2, arguments)
    for i, label in enumerate(image.labels):
        assert label.box == constants.EXPECTED_BOXES[i]


@patch(
    "morpheus.dataset.remap_classes",
    new=Mock(),
)
@patch(
    "morpheus.dataset.get_class_names",
    new=Mock(return_value=constants.CLASS_NAMES),
)
def test_main(arguments, images):
    """Test the main function"""
    morpheus.dataset.remap_classes.return_value = (
        constants.CLASS_NAMES,
        images,
    )
    # Create the input and output directories
    # input_path = tmp_path / "input"
    # output_path = tmp_path / "output"
    # mock the functions
    morpheus.dataset.match_images_to_label_files = Mock()
    morpheus.dataset.generate_dataset = Mock()
    # Run the main function
    morpheus.dataset.main(arguments.input, arguments.output, arguments)
    # Ensure that the functions were called
    morpheus.dataset.match_images_to_label_files.assert_called()
    morpheus.dataset.get_class_names.assert_called()
    morpheus.dataset.generate_dataset.assert_called()
    morpheus.dataset.remap_classes.assert_called()


def test_parse_args(tmp_path):
    """Test the parse_args function"""
    # Test with minimum required arguments
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    out_dir.mkdir()
    multiple_for_image_duplication = 5
    main_args = [str(in_dir), str(out_dir)]
    args = main_args
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.multiply == 1
    assert parsed_args.augment is False
    # Test with multiply argument
    args += ["--multiply", str(multiple_for_image_duplication)]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.multiply == multiple_for_image_duplication
    assert parsed_args.augment is False
    # Test with multiply argument as -m
    args = main_args
    args += ["-m", str(multiple_for_image_duplication)]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.multiply == multiple_for_image_duplication
    assert parsed_args.augment is False
    # Test with resize argument
    args = main_args
    args += ["--resize", "670"]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.resize == 670
    assert parsed_args.augment is False
    # test with resize argument as -r
    args = main_args
    args += ["-r", "670"]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.resize == 670
    assert parsed_args.augment is False
    # test with flip-h argument
    args = main_args
    args += ["--flip-h"]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.flip_h is True
    assert parsed_args.augment is True
    # test with flip-v argument
    args = main_args
    args += ["--flip-v"]
    parsed_args = parse_args(args)
    assert parsed_args.input_directory == in_dir
    assert parsed_args.output_directory == out_dir
    assert parsed_args.flip_h is True
    assert parsed_args.augment is True


# @patch("morpheus.dataset.remap_class", new=Mock(return_value="car"))
@patch(
    "morpheus.dataset.map_class_choice",
    new=Mock(),
)
@patch(
    "builtins.input",
    new=Mock(
        side_effect=[
            "n",  # prompt to remap classes - no
            "test_bad_input",  # prompt to remap classes - bad input
            "y",  # prompt to remap classes - yes
            "test_bad_input",  # prompt to remap more classes - bad input
            "y",  # prompt to remap more classes - yes
            "n",  # prompt to remap more classes - no
        ]
    ),
)
def test_remap_classes(images, remapped_images):
    """Test the remap_classes function"""
    morpheus.dataset.map_class_choice.return_value = [
        constants.REMAPPED_CLASS_NAME,
        remapped_images,
    ]
    # Test with no remapping
    _, _ = morpheus.dataset.remap_classes(constants.CLASS_NAMES.copy(), images)
    # Test with remapping
    class_names, images_remapped = morpheus.dataset.remap_classes(
        constants.CLASS_NAMES.copy(), images
    )
    assert class_names == ["car"]
    for image in images_remapped:
        for label in image.labels:
            assert label.name == "car"


@patch("morpheus.dataset.remap_class", new=Mock())
@patch(
    "builtins.input",
    new=Mock(
        side_effect=[
            "q",  # prompt to choose a class - quit
            "test_bad_input",  # prompt to remap a class - bad input
            "1",  # prompt to remap a class - class 1
        ]
    ),
)
def test_map_class_choice(images, remapped_images):
    """Test the map_class_choice function"""
    morpheus.dataset.remap_class.return_value = [
        constants.REMAPPED_CLASS_NAME,
        remapped_images,
    ]
    # Test with no remapping - user selects q
    class_names, images_remapped = morpheus.dataset.map_class_choice(
        constants.CLASS_NAMES, images
    )
    assert class_names == constants.CLASS_NAMES
    assert images_remapped == images
    # Test with remapping - user enters bad input, then makes choice
    class_names, images_remapped = morpheus.dataset.map_class_choice(
        constants.CLASS_NAMES, images
    )
    assert class_names == constants.REMAPPED_CLASS_NAME
    assert images_remapped == remapped_images


@patch(
    "builtins.input",
    new=Mock(
        side_effect=[
            "q",  # prompt to enter class name - quit
            "carrrrr",  # prompt to enter class name - new class name
            "car",  # prompt to enter class name - car class
            "y",  # prompt class exists, remap? - yes
            "car",  # prompt to enter class name - car class
            "car_test",  # prompt to enter class name - car class
        ]
    ),
)
def test_remap_class(images):
    """Test the remap_class function"""
    # Test with no remapping - user selects q
    remapped_class_names, mapped_images = morpheus.dataset.remap_class(
        constants.CLASS_NAMES.copy(), 2, images
    )
    assert remapped_class_names == constants.CLASS_NAMES
    assert mapped_images == images
    # Test with remapping - new class name

    remapped_class_names, mapped_images = morpheus.dataset.remap_class(
        remapped_class_names, 2, images
    )
    assert remapped_class_names == ["car", "carrrrr"]
    for image in mapped_images:
        for label in image.labels:
            assert label.name != "train"

    # Test with remapping - existing class name
    remapped_class_names, mapped_images = morpheus.dataset.remap_class(
        remapped_class_names, 2, images
    )
    assert remapped_class_names == constants.REMAPPED_CLASS_NAME
    assert images == mapped_images
    # Test with remapping - name is the same as old class name
    remapped_class_names, mapped_images = morpheus.dataset.remap_class(
        remapped_class_names, 2, images
    )
    assert remapped_class_names == ["car_test"]
    for image in mapped_images:
        for label in image.labels:
            assert label.name == "car_test"
