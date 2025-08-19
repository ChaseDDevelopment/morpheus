# Morpheus Dataset Tools

[![Ruff](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/ruff.yml/badge.svg)](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/ruff.yml)
[![Pytest](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/pytest.yml/badge.svg)](https://github.com/ChaseDDevelopment/morpheus/actions/workflows/pytest.yml)

A modern Python tool for converting image datasets with XML annotations into YOLOv8-compatible format. Features automatic duplicate filename handling, dataset augmentation, and interactive class remapping.

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/ChaseDDevelopment/morpheus.git
cd morpheus

# Install dependencies
uv sync

# Verify installation
uv run morpheus --help
```

## Usage

### Basic Command
```bash
uv run morpheus <input_directory> <output_directory> [options]
```

### Arguments
- `<input_directory>`: Directory containing images and XML annotation files
- `<output_directory>`: Where to save the YOLOv8 dataset

### Options
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--multiply` | `-m` | Duplicate dataset N times for augmentation | 1 |
| `--resize` | `-r` | Resize images to specified size | 640 |
| `--flip-h` | | Horizontal flip augmentation* | False |
| `--flip-v` | | Vertical flip augmentation* | False |
| `--gaussian-blur` | | Apply Gaussian blur* | False |
| `--in-memory` | | Load all images to RAM (faster) | False |

*Requires `--multiply > 1`

### Examples

#### Basic conversion
```bash
uv run morpheus ./raw_data ./yolo_dataset
```

#### With augmentation
```bash
uv run morpheus ./raw_data ./yolo_dataset \
    --multiply 3 \
    --resize 416 \
    --flip-h \
    --flip-v \
    --gaussian-blur
```

#### Fast processing (high RAM usage)
```bash
uv run morpheus ./raw_data ./yolo_dataset --in-memory
```

### Augmentation Details

When augmentations are enabled (with `--multiply > 1`), the tool randomly applies one or more of the enabled augmentations to each duplicated image:

- **Horizontal Flip** (`--flip-h`): Flips the image horizontally and adjusts bounding boxes accordingly
- **Vertical Flip** (`--flip-v`): Flips the image vertically and adjusts bounding boxes accordingly  
- **Gaussian Blur** (`--gaussian-blur`): Applies Gaussian blur with a randomly selected kernel size (3, 5, or 7) to add slight blur, which can help improve model robustness to image quality variations

### Duplicate Filename Handling

Morpheus automatically handles datasets with duplicate filenames across different subdirectories. This is particularly useful for datasets organized in nested structures where multiple directories might contain files with the same name (e.g., `frame0001.png`).

**How it works:**
- The tool preserves directory structure information when generating output filenames
- Combines parent directory names with the original filename to create unique identifiers
- Example: `category/timestamp_frames/frame0001.png` → `category_timestamp_frame0001.png`
- No configuration needed - this happens automatically when duplicate names are detected

**Supported formats:**
- Images: `.jpg`, `.png`, `.bmp`
- Annotations: `.xml` (Pascal VOC format)

### Interactive Class Remapping

After scanning your dataset, the script will prompt you to optionally remap class names. Follow the prompts to rename or merge classes as needed.

### Output Structure

The output directory will contain:
- `train/`, `valid/`, `test/` folders with `images/` and `labels/` subfolders
- A `data.yaml` file describing the dataset for YOLOv8
- All image files will have unique names to prevent overwrites

## Development

### Running Tests
```bash
# Install development dependencies
uv sync --all-extras

# Run tests with coverage
uv run pytest --cov=. --cov-fail-under=80

# Run linting
uv run ruff check .

# Format code
uv run ruff format .
```

### Project Structure
```
morpheus/
├── morpheus/         # Main package
│   └── dataset.py    # Core functionality
├── models/           # Data models
├── utils/            # Utility functions  
├── tests/            # Test suite
└── pyproject.toml    # Project configuration
```

## License

MIT License - see LICENSE file for details.