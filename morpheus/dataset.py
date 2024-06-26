"""Dataset module for Morpheus."""
import argparse
import copy
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

import morpheus.constants as const
from models.dataclasses import (
    MorpheusImage,
)
from utils.general import match_images_to_label_files, get_class_names


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

    num_augmentations = random.randint(1, len(augment_options))
    chosen_augmentations = random.sample(augment_options, num_augmentations)
    for augmentation in chosen_augmentations:
        if augmentation == "flip_h":
            img_data.flip_horizontally()
        if augmentation == "flip_v":
            img_data.flip_vertically()
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
    assert (
        round(train_ratio + val_ratio + test_ratio, 5) == 1.0
    ), "Split Ratios should sum up to 1"
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
            # Write image
            new_img_path = images_dir / img_data.name
            shutil.copy2(img_data.path, new_img_path)
            img_data.path = new_img_path
            img_data.load_image_to_memory()
            if img_data.image_size != (resize, resize):
                img_data.resize_image(resize, resize)
            if args.augment:
                img_data = augment(img_data, i, args)
            img_data.write_image_to_disk()
            split_images.append(str(img_data.path))

            # Write labels
            label_path = labels_dir / f"{Path(img_data.name).stem}.txt"
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
            )
            multiplied_images.append(new_image)
    return multiplied_images


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
    morpheus_images = match_images_to_label_files(input_directory)
    class_names = get_class_names(morpheus_images)
    class_names, _ = remap_classes(class_names, morpheus_images)
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
    print(
        "The following class names were found within the dataset,"
        "would you like to re-map any of them?"
    )
    print(class_names)
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
    arguments = parser.parse_args(arguments)
    if arguments.flip_h or arguments.flip_v:
        setattr(arguments, "augment", True)
    else:
        setattr(arguments, "augment", False)
    return arguments


if __name__ == "__main__":  # pragma: no cover
    parsed_arguments = parse_args(sys.argv[1:])
    if (
        parsed_arguments.flip_h or parsed_arguments.flip_v
    ) and parsed_arguments.multiply == 1:
        print("You must set a multiple greater than 1 to use augmentations")
        quit()

    main(
        Path(parsed_arguments.input_directory),
        Path(parsed_arguments.output_directory),
        parsed_arguments,
    )
