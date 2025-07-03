# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""This module contains dataclasses needed for morpheus"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np


@dataclass
class MorpheusBoundingBox:
    """
    A dataclass representing a bounding box in an image.

    Attributes:
        xmin (int): The x-coordinate of the left edge of the bounding box.
        ymin (int): The y-coordinate of the top edge of the bounding box.
        xmax (int): The x-coordinate of the right edge of the bounding box.
        ymax (int): The y-coordinate of the bottom edge of the bounding box.

    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def width(self) -> int:
        """
        Computes the width of the bounding box.

        Returns:
            int: The width of the bounding box.

        """
        return self.xmax - self.xmin

    def height(self) -> int:
        """
        Computes the height of the bounding box.

        Returns:
            int: The height of the bounding box.

        """
        return self.ymax - self.ymin

    def area(self) -> int:
        """
        Computes the area of the bounding box.

        Returns:
            int: The area of the bounding box.

        """
        return self.width() * self.height()

    def center(self) -> Tuple[int, int]:
        """
        Computes the center point of the bounding box.

        Returns:
            Tuple[int, int]: A tuple containing the x- and y-coordinates
                             of the center point.

        """
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2

    def as_dict(self) -> Dict[str, float]:
        """Return a dictionary representation of the bounding box.

        Returns:
            A dictionary with keys 'xmin', 'ymin', 'xmax', and 'ymax', corresponding to
            the minimum and maximum x and y values of the bounding box.

        """
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }


@dataclass
class MorpheusLabel:
    """
    A dataclass representing a label, and its boxed object.

    Attributes:
        name (str): The name of the object.
        box (MorpheusBoundingBox): A bounding box that encloses its respective object.

    """

    name: str
    box: MorpheusBoundingBox

    def as_dict(self) -> dict:
        """Return a dictionary representation of the Label"""
        return {
            "name": self.name,
            "box": self.box.as_dict(),
        }


@dataclass
class MorpheusImage:
    """
    A dataclass representing a labeled image and all respective objects in an image.

    Attributes:
        name (str): The name of the image.
        path (Path): The path to the image.
        labels (List[MorpheusLabel]): A list of Labels associated with the image
        image_size (Tuple[int, int]): The size of the object in pixels (width, height).
        image_depth (int): The depth of the object in bits per pixel.
        image (Optional: np.ndarray): The image stored as a numpy array.

    """

    name: str
    path: Path
    labels: List[MorpheusLabel]
    image_size: Tuple[int, int]
    image_depth: int
    image: Optional[np.ndarray] = None

    def get_labels(self) -> dict:
        """Return a list of labels associated with the image"""
        return {key: label.as_dict() for key, label in enumerate(self.labels)}

    def flip_horizontally(self):
        """Flip the Image Horizontally and all labels/bounding boxes with it"""
        self.image = np.flip(self.image, axis=1)
        image_width = self.image.shape[1]
        for label in self.labels:
            old_xmin = label.box.xmin
            old_xmax = label.box.xmax
            label.box.xmin = max(0, image_width - old_xmax)
            label.box.xmax = min(image_width, image_width - old_xmin)

    def flip_vertically(self):
        """Flip the Image Vertically and all labels/bounding boxes with it"""
        self.image = np.flip(self.image, axis=0)
        image_height = self.image.shape[0]
        for label in self.labels:
            old_ymin = label.box.ymin
            old_ymax = label.box.ymax
            label.box.ymin = max(0, image_height - old_ymax)
            label.box.ymax = min(image_height, image_height - old_ymin)

    def load_image_to_memory(self):
        """Load the image into memory as a numpy array from disk"""
        self.image = cv2.imread(str(self.path))

    def is_loaded_in_memory(self) -> bool:
        """Check if image is currently loaded in memory"""
        return self.image is not None

    def write_image_to_disk(self, keep_in_memory: bool = False):
        """Write the image to disk as an image

        Args:
            keep_in_memory (bool): If True, keep image in memory after writing
        """
        cv2.imwrite(str(self.path), self.image)
        if not keep_in_memory:
            del self.image

    def resize_image(self, width: int, height: int):
        """Resize the image to the desired size"""
        scale = min(width / self.image.shape[1], height / self.image.shape[0])
        scale = min(scale, 1.0)
        new_width = int(round(self.image.shape[1] * scale))
        new_height = int(round(self.image.shape[0] * scale))
        self.image = cv2.resize(
            self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        for label in self.labels:
            label.box.xmin = int(label.box.xmin * scale)
            label.box.xmax = int(label.box.xmax * scale)
            label.box.ymin = int(label.box.ymin * scale)
            label.box.ymax = int(label.box.ymax * scale)
        self.image_size = (new_width, new_height)

    def apply_gaussian_blur(self, kernel_size: int = 5, sigma: float = 0):
        """Apply Gaussian blur to the image

        Args:
            kernel_size (int): Kernel size for Gaussian blur. Must be odd and positive.
            sigma (float): Standard deviation for Gaussian kernel. If 0, calculated automatically.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be odd and positive")
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
