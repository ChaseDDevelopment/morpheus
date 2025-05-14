# Morpheus Dataset Tools

[![Ruff](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/ruff.yml/badge.svg)](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/ruff.yml)
[![Pytest](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/pytest.yml/badge.svg)](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/pytest.yml)

This module provides tools to convert image datasets and their annotation files into a YOLOv8-compatible format, with options for augmentation and class remapping.

## Requirements

- Python 3.8+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

The main script is located at `morpheus/dataset.py`. It processes a directory of images and XML annotation files, and outputs a YOLOv8 dataset.

### Command Line

```sh
python -m morpheus.dataset <input_directory> <output_directory> [options]
```

#### Arguments

- `<input_directory>`: Path to the directory containing images and XML files.
- `<output_directory>`: Path where the generated dataset will be saved.

#### Options

- `--multiply, -m <int>`: Number of times to duplicate the dataset (default: 1).
- `--resize, -r <int>`: Resize images to this size (default: 640).
- `--flip-h`: Apply horizontal flip augmentation (requires `--multiply > 1`).
- `--flip-v`: Apply vertical flip augmentation (requires `--multiply > 1`).

#### Example

```sh
python -m morpheus.dataset ./data/input ./data/output --multiply 3 --resize 416 --flip-h --flip-v
```

This will:
- Read images and XML files from `./data/input`
- Generate a YOLOv8 dataset in `./data/output`
- Duplicate the dataset 3 times
- Resize images to 416x416
- Apply horizontal and vertical flip augmentations

### Interactive Class Remapping

After scanning your dataset, the script will prompt you to optionally remap class names. Follow the prompts to rename or merge classes as needed.

### Output Structure

The output directory will contain:
- `train/`, `valid/`, `test/` folders with `images/` and `labels/` subfolders
- A `data.yaml` file describing the dataset for YOLOv8

---

For more details, see the docstrings in [`morpheus/dataset.py`](morpheus/dataset.py).