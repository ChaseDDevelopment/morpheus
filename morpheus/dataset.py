# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""Dataset module for Morpheus."""

import argparse
import copy
import psutil
import random
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

import morpheus.constants as const
from models.dataclasses import (
    MorpheusImage,
)
from utils.general import match_images_to_label_files, get_class_names


def estimate_memory_usage(
    images: List[MorpheusImage], args: argparse.Namespace
) -> Tuple[float, float]:
    """Estimate memory usage for in-memory processing.

    Args:
        images (List[MorpheusImage]): List of images to process.
        args (argparse.Namespace): Arguments containing processing options.

    Returns:
        Tuple[float, float]: (estimated_memory_gb, available_memory_gb)
    """
    if not images:
        return 0.0, psutil.virtual_memory().available / (1024**3)

    # Calculate average image size from first few images
    sample_size = min(5, len(images))
    total_pixels = 0

    for img in images[:sample_size]:
        width, height = img.image_size
        total_pixels += width * height

    avg_pixels = total_pixels / sample_size

    # Estimate memory per image after resize
    resize_pixels = args.resize * args.resize
    bytes_per_pixel = 3  # RGB image

    # Memory per image in bytes (original + resized during processing)
    memory_per_image = (avg_pixels + resize_pixels) * bytes_per_pixel

    # Account for multiplication and augmentation
    total_images = len(images) * args.multiply

    # Add overhead for processing (labels, intermediate data, etc.)
    overhead_factor = 1.2

    estimated_memory_bytes = total_images * memory_per_image * overhead_factor
    estimated_memory_gb = estimated_memory_bytes / (1024**3)

    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    return estimated_memory_gb, available_memory_gb


def check_memory_feasibility(
    images: List[MorpheusImage], args: argparse.Namespace
) -> bool:
    """Check if in-memory processing is feasible and warn user if not.

    Args:
        images (List[MorpheusImage]): List of images to process.
        args (argparse.Namespace): Arguments containing processing options.

    Returns:
        bool: True if processing should continue, False if user cancelled.
    """
    estimated_gb, available_gb = estimate_memory_usage(images, args)

    print("Memory estimation:")
    print(f"  Estimated usage: {estimated_gb:.2f} GB")
    print(f"  Available memory: {available_gb:.2f} GB")

    if estimated_gb > available_gb * 0.8:  # Use 80% as safety threshold
        warnings.warn(
            f"Warning: Estimated memory usage ({estimated_gb:.2f} GB) may exceed "
            f"available memory ({available_gb:.2f} GB). Consider processing without "
            f"--in-memory flag or reducing dataset size.",
            UserWarning,
        )

        while True:
            user_input = input("Continue anyway? (y/n): ").lower()
            if user_input in ["yes", "y"]:
                return True
            elif user_input in ["no", "n"]:
                return False
            print("Please enter 'yes' or 'no' (y/n)")

    return True


def augment(img_data: MorpheusImage, i: int, args: argparse.Namespace) -> MorpheusImage:
    """Augment an image.

    Args:
        img_data (MorpheusImage): The image to augment.
        i (int): The index of the image.
        args (argparse.Namespace): The arguments passed to the program,
                                   containing augmentations.

    Returns:
        MorpheusImage: The augmented image.

    """
    if (i + 1) % args.multiply == 0:
        return img_data
    augment_options = []
    if args.flip_h:
        augment_options.append("flip_h")
    if args.flip_v:
        augment_options.append("flip_v")
    if args.gaussian_blur:
        augment_options.append("gaussian_blur")

    num_augmentations = random.randint(1, len(augment_options))
    chosen_augmentations = random.sample(augment_options, num_augmentations)
    for augmentation in chosen_augmentations:
        if augmentation == "flip_h":
            img_data.flip_horizontally()
        if augmentation == "flip_v":
            img_data.flip_vertically()
        if augmentation == "gaussian_blur":
            kernel_size = random.choice([3, 5, 7])
            img_data.apply_gaussian_blur(kernel_size=kernel_size)
    return img_data


def generate_dataset(
    output_directory: Path,
    class_names: list,
    images: List[MorpheusImage],
    args: argparse.Namespace,
    split_ratios: Tuple = (0.7, 0.2, 0.1),
):
    """Generate a yolov8 dataset from a directory of images and xml files.

    Args:
        output_directory (Path): The directory to output the dataset to.
        class_names (list): A list of class names to use for the dataset.
        images (MorpheusImage): A list of MorpheusImage objects to use for the dataset.
        args (argparse.Namespace): The arguments passed to the program.
        split_ratios (tuple, optional): A tuple of floats representing the train,
                                        validation, and test split ratios.
                                        Defaults to (0.7, 0.2, 0.1).

    Returns:
        None

    """
    # Ensure directory exists
    output_directory.mkdir(parents=True, exist_ok=True)
    resize = args.resize
    multiply = args.multiply if args.multiply != 1 else 1

    train_ratio, val_ratio, test_ratio = split_ratios
    assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, (
        "Split Ratios should sum up to 1"
    )
    if multiply > 1:
        images = multiply_files(images, multiply)
    total_images = len(images)

    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    random.shuffle(images)

    train_data = images[:train_end]
    val_data = images[train_end:val_end]
    test_data = images[val_end:]

    splits = [("train", train_data), ("valid", val_data), ("test", test_data)]

    # Load all images in memory if requested
    if args.in_memory:
        print("Loading all images into memory for faster processing...")
        for img_data in tqdm(images, desc="Loading images", unit="image(s)"):
            if not img_data.is_loaded_in_memory():
                img_data.load_image_to_memory()

    for split_name, split_data in splits:
        labels_dir = output_directory / split_name / "labels"
        images_dir = output_directory / split_name / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        split_images = []
        for i, img_data in tqdm(
            enumerate(split_data),
            desc=f"Writing {split_name} data",
            total=len(split_data),
            unit="image(s)",
        ):
            # Write image with unique filename to avoid collisions
            unique_filename = img_data.get_unique_filename()
            new_img_path = images_dir / unique_filename
            shutil.copy2(img_data.path, new_img_path)
            img_data.path = new_img_path

            # Load image if not already in memory
            if not img_data.is_loaded_in_memory():
                img_data.load_image_to_memory()

            if img_data.image_size != (resize, resize):
                img_data.resize_image(resize, resize)
            if args.augment:
                img_data = augment(img_data, i, args)

            # Keep in memory if in-memory mode, otherwise free memory
            img_data.write_image_to_disk(keep_in_memory=args.in_memory)
            split_images.append(str(img_data.path))

            # Write labels using the unique filename stem
            label_path = labels_dir / f"{Path(unique_filename).stem}.txt"
            with label_path.open("w") as f:
                for label in img_data.labels:
                    bbox = label.box
                    x_center = (bbox.xmin + bbox.xmax) / 2 / img_data.image_size[0]
                    y_center = (bbox.ymin + bbox.ymax) / 2 / img_data.image_size[1]
                    width = (bbox.xmax - bbox.xmin) / img_data.image_size[0]
                    height = (bbox.ymax - bbox.ymin) / img_data.image_size[1]

                    # Get class index
                    class_index = class_names.index(label.name)

                    # Write to file
                    f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

    with open(output_directory / "data.yaml", "w") as outfile:
        outfile.write("train: ../train/images\n")
        outfile.write("val: ../valid/images\n")
        outfile.write("test: ../test/images\n")
        outfile.write(f"nc: {len(class_names)}\n")
        outfile.write(
            "names: [" + ", ".join(f"'{name}'" for name in class_names) + "]\n"
        )


def multiply_files(images: List[MorpheusImage], multiple: int):
    """Multiply the files in the output directory.

    Args:
        images (List[MorpheusImage]): A list of MorpheusImage
                                      objects to use for the dataset.
        multiple (int): The number of times to multiply the files.

    """
    multiplied_images = []
    for image in images:
        multiplied_images.append(image)
        for i in range(1, multiple):
            base = image.path.stem
            new_name = f"{base}_{i}{image.path.suffix}"
            new_image = MorpheusImage(
                name=new_name,
                path=copy.deepcopy(image.path),
                labels=copy.deepcopy(image.labels),
                image_size=copy.deepcopy(image.image_size),
                image_depth=copy.deepcopy(image.image_depth),
                relative_path=copy.deepcopy(image.relative_path),
            )
            multiplied_images.append(new_image)
    return multiplied_images


def count_class_instances(images: List[MorpheusImage]) -> None:
    """Count instances per class, print unlabeled images, and list images with 'jug'."""
    class_counts = Counter()
    unlabeled_count = 0

    for image in images:
        if not image.labels:  # No labels (e.g., no XML)
            unlabeled_count += 1
        else:
            for label in image.labels:
                class_counts[label.name] += 1

    print("Dataset Statistics:")
    print(f"Total unlabeled images: {unlabeled_count}")
    if class_counts:
        print("Total class instances in dataset:")
        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name}: {count} instances")
    else:
        print("No labeled instances found in dataset.")



def main(input_directory: Path, output_directory: Path, args=None):
    """Main function

    Args:
        input_directory (Path): The path to the directory containing the files.
        output_directory (Path): The path to the directory where the generated
                                 dataset will be
        args (list): List of arguments passed in.

    Returns:
        None

    """
    morpheus_images = match_images_to_label_files(
        input_directory, include_negatives=args.include_negatives
    )
    class_names = get_class_names(morpheus_images)
    class_names, _ = remap_classes(class_names, morpheus_images)

    count_class_instances(morpheus_images)

    # Check memory feasibility if in-memory processing is requested
    if args.in_memory:
        if not check_memory_feasibility(morpheus_images, args):
            print("Aborting processing due to memory constraints.")
            return

    generate_dataset(output_directory, class_names, morpheus_images, args)


def remap_classes(
    class_names: List[str], images: List[MorpheusImage]
) -> (List[str], List[MorpheusImage]):
    """Prompt the user to remap any of the class names.

    Args:
        class_names (List[str]): A list of class names.
        images (List[MorpheusImage]): A list of MorpheusImage objects.

    Returns:
        (List[str], List[MorpheusImage]): A tuple containing a
                                        list of class names and
                                        a list of MorpheusImage objects.

    """
    print("The following class names were found within the dataset:")
    print(class_names)

    if len(class_names) > 1:
        while True:
            remap_all_input = input(
                "Would you like to remap all classes to a single name? (y/n): "
            ).lower()
            if remap_all_input in ["yes", "y", "no", "n"]:
                break
            print(const.INVALID_INPUT)
        if remap_all_input in ["yes", "y"]:
            new_name = input("Enter the class name: ").strip()
            if new_name:
                old_count = len(class_names)
                for image in images:
                    for label in image.labels:
                        label.name = new_name
                class_names = [new_name]
                print(f"Remapped {old_count} classes -> '{new_name}'")
                return class_names, images

    print("Would you like to re-map any of them?")
    while True:
        user_input = input("Enter 'yes' or 'no' (y/n): ").lower()
        if user_input in ["yes", "no", "y", "n"]:
            break
        print(const.INVALID_INPUT)
    if user_input == "no" or user_input == "n":
        return class_names, images
    continue_remapping = True
    retry = False
    while continue_remapping:
        if user_input == "yes" or user_input == "y":
            if not retry:
                class_names, images = map_class_choice(class_names, images)
                class_names = sorted(class_names)
                print(class_names)
                continue_choice = input(
                    "Would you like to re-map another class? (y/n): "
                )
            if continue_choice == "y" or continue_choice == "yes":
                retry = False
                continue_remapping = True
                continue
            elif continue_choice == "n" or continue_choice == "no":
                retry = False
                continue_remapping = False
                break
            retry = True
            continue_choice = input(const.INVALID_INPUT)

    return class_names, images


def map_class_choice(
    class_names: List[str], images: List[MorpheusImage]
) -> (List[str], List[MorpheusImage]):
    """

    Args:
        class_names (List[str]): A list of class names.
        images (List[MorpheusImage]): A list of MorpheusImage objects
                                      to use for the dataset.

    Returns:
        (List[str], List[MorpheusImage]): A tuple containing a
                                        list of class names and
                                        a list of MorpheusImage objects.

    """
    print("Which class would you like to re-map?")
    while True:
        for i, option in enumerate(class_names, 1):
            print(f"{i}) {option}")
        class_choice = input("Enter the number of the class or (q) to cancel: ")
        if class_choice.isdigit() and int(class_choice) <= len(class_names):
            break
        elif class_choice == "q":
            break
        else:
            print()
            print("Invalid input. Please enter a class number.")
    if class_choice == "q":
        print("User cancelled.")
        return class_names, images
    else:
        class_choice = int(class_choice)
        return remap_class(class_names, class_choice, images)


def remap_class(
    class_names: List[str], class_choice: int, images: List[MorpheusImage]
) -> (List[str], List[MorpheusImage]):
    """Remap a class name.

    Args:
        class_names (List[str]): A list of class names.
        class_choice (int): The index of the class to remap.
        images (List[MorpheusImage]): A list of MorpheusImage objects.

    Returns:
        (List[str], List[MorpheusImage]): A tuple containing a
                                        list of class names and
                                        a list of MorpheusImage objects.

    """
    class_choice_index = class_choice - 1 if len(class_names) != 1 else 0
    print(f"Remapping class '{class_names[class_choice_index]}'")
    while True:
        new_class_name = input("Enter the new class name or (q) to cancel: ")
        if new_class_name == class_names[class_choice_index]:
            print("New class name is the same as the old class name.")
            continue
        if new_class_name == "q":
            break
        elif new_class_name in class_names:
            class_exists = input(
                f"Class name already exists, would you like to remove"
                f" the current class and remap it to {new_class_name}? (y/n)"
            )
            if class_exists == "y":
                for image in images:
                    for label in image.labels:
                        if label.name == class_names[class_choice_index]:
                            label.name = new_class_name
                class_names.remove(class_names[class_choice_index])
                class_names = sorted(class_names)
                break
        else:
            for image in images:
                for label in image.labels:
                    if label.name == class_names[class_choice_index]:
                        label.name = new_class_name
            class_names[class_choice_index] = new_class_name
            break
    if new_class_name == "q":
        print("User cancelled.")
        return class_names, images
    else:
        return class_names, images


def parse_args(arguments):
    """
    Parses the command line arguments.

    Args:
        arguments (list): List of arguments to parse.

    Returns:
        Namespace: The parsed arguments.

    """
    parser = argparse.ArgumentParser(description="Dataset Tools")
    parser.add_argument(
        "input_directory",
        type=Path,
        help="The path to the directory containing the files.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="The path to the directory where the generated dataset will be",
    )
    parser.add_argument(
        "--multiply",
        "-m",
        nargs="?",
        default=1,
        type=int,
        help="The number of times to multiply the dataset",
    )
    parser.add_argument(
        "--resize",
        "-r",
        nargs="?",
        type=int,
        default=640,
        help="Size to resize images to. Defaults to: 640",
    )
    parser.add_argument(
        "--flip-h", action="store_true", help="Flip images horizontally"
    )
    parser.add_argument("--flip-v", action="store_true", help="Flip images vertically")
    parser.add_argument(
        "--gaussian-blur", action="store_true", help="Apply Gaussian blur augmentation"
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Process all images in memory (faster but uses more RAM)",
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Include images without XML annotations as negative samples (empty labels)",
    )
    arguments = parser.parse_args(arguments)
    if arguments.flip_h or arguments.flip_v or arguments.gaussian_blur:
        setattr(arguments, "augment", True)
    else:
        setattr(arguments, "augment", False)
    return arguments


def cli_main():
    """Entry point for the CLI script."""
    parsed_arguments = parse_args(sys.argv[1:])
    if (
        parsed_arguments.flip_h
        or parsed_arguments.flip_v
        or parsed_arguments.gaussian_blur
    ) and parsed_arguments.multiply == 1:
        print("You must set a multiple greater than 1 to use augmentations")
        quit()

    main(
        Path(parsed_arguments.input_directory),
        Path(parsed_arguments.output_directory),
        parsed_arguments,
    )


if __name__ == "__main__":  # pragma: no cover
    cli_main()
