# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""General utility functions for Morpheus."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import cv2
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


def _indent_xml(elem, level=0):
    """Add indentation to XML elements for pretty printing (Python 3.8 compatible).

    Args:
        elem: The XML element to indent.
        level: Current indentation level.
    """
    indent = "\n" + "\t" * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if not level:
        elem.tail = "\n"


def _write_empty_xml(xml_path: Path, image_path: Path, width: int, height: int, depth: int):
    """Write a LabelIMG-compatible XML annotation file with no objects.

    Args:
        xml_path: Path to write the XML file.
        image_path: Path to the source image.
        width: Image width in pixels.
        height: Image height in pixels.
        depth: Image color depth (channels).
    """
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    _indent_xml(annotation)
    tree = ET.ElementTree(annotation)
    tree.write(str(xml_path), encoding="unicode", xml_declaration=False)


def match_images_to_label_files(
    directory: Path, include_negatives: bool = False
) -> List[MorpheusImage]:
    """Map each image to its corresponding xml file.

    Args:
        directory (Path): The directory containing the images and xml files.
        include_negatives (bool): If True, include images without XML annotations
            as negative samples (empty labels). Generates empty XML files for
            consistency. Defaults to False (skip images without XML).

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
    negative_count = 0
    for file in files:
        if file.suffix.lower() in [".bmp", ".jpg", ".jpeg", ".png"]:
            xml_file = file.with_suffix(".xml")
            relative_path = file.relative_to(directory)

            if xml_file.is_file():
                tree = ET.parse(xml_file)
                root = tree.getroot()

                image_size = root.find("size")
                width = int(image_size.find("width").text)
                height = int(image_size.find("height").text)
                image_size = (width, height)

                labels = []
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    label = MorpheusLabel(
                        name, MorpheusBoundingBox(xmin, ymin, xmax, ymax)
                    )
                    labels.append(label)
            elif include_negatives:
                # No XML = negative sample (empty belt, no objects)
                # Read image to get dimensions and generate XML for consistency
                img = cv2.imread(str(file))
                if img is None:
                    progress_bar.update(1)
                    continue
                height, width = img.shape[:2]
                depth = img.shape[2] if len(img.shape) == 3 else 1
                image_size = (width, height)
                labels = []
                del img

                # Write XML with same structure as LabelIMG but no objects
                _write_empty_xml(xml_file, file, width, height, depth)
                negative_count += 1
            else:
                progress_bar.update(1)
                continue

            image = MorpheusImage(
                file.name, file, labels, image_size, 3, relative_path=relative_path
            )
            morpheus_images.append(image)
            progress_bar.update(1)
    progress_bar.close()
    print(
        f"Matched {len(morpheus_images)} images: "
        f"{len(morpheus_images) - negative_count} labeled, "
        f"{negative_count} negative (no XML)"
    )
    return morpheus_images
