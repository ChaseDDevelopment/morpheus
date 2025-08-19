# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""General utility functions for Morpheus."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from tqdm import tqdm

from models.dataclasses import (
    MorpheusImage,
    MorpheusLabel,
    MorpheusBoundingBox,
)


def get_class_names(images: List[MorpheusImage]) -> List[str]:
    """Get the class names from a list of images.

    Args:
        images (MorpheusImage): A list of MorpheusImage objects to use for the dataset.

    Returns:
        List[str]: A list of class names.

    """
    class_names = []
    for image in images:
        for label in image.labels:
            if label.name not in class_names:
                class_names.append(label.name)
    class_names.sort()
    return class_names


def match_images_to_label_files(directory: Path) -> List[MorpheusImage]:
    """Map each image to its corresponding xml file.

    Args:
        directory (Path): The directory containing the images and xml files.

    Returns:
        List[MorpheusImages]: A list of MorpheusImage objects.

    """
    morpheus_images = []
    files = directory.rglob("*")
    total_files = sum(1 for _ in files)
    files = directory.rglob("*")
    progress_bar = tqdm(
        total=total_files,
        unit="file(s)",
        desc="Processing and matching images to xml files",
        dynamic_ncols=True,
    )
    for file in files:
        if file.suffix in [".bmp", ".jpg", ".png"]:
            xml_file = file.with_suffix(".xml")
            if xml_file.is_file():
                # print(f"Match found: {file.name} and {xml_file.name}")
                tree = ET.parse(xml_file)
                root = tree.getroot()

                image_size = root.find("size")
                width = int(image_size.find("width").text)
                height = int(image_size.find("height").text)
                image_size = (width, height)

                labels = []
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    # Get the object bounding box
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    label = MorpheusLabel(
                        name, MorpheusBoundingBox(xmin, ymin, xmax, ymax)
                    )
                    labels.append(label)

                # Calculate relative path from the input directory
                relative_path = file.relative_to(directory)

                image = MorpheusImage(
                    file.name, file, labels, image_size, 3, relative_path=relative_path
                )
                morpheus_images.append(image)
                progress_bar.update(1)
    progress_bar.close()
    return morpheus_images
