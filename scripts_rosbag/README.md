# ROS Bag to Zarr Converter

This directory contains tools for converting ROS bag data to the zarr format used by the Universal Manipulation Interface (UMI).

## `generate_replay_buffer.py`

This script converts extracted ROS bag data into a zarr dataset compatible with the UMI system. It processes trajectory data, timestamps, and images to create a standardized dataset that can be used with the UMI tools.

### Prerequisites

- Python 3.7+
- Required packages:
  - numpy
  - opencv-python (cv2)
  - zarr
  - tqdm
  - scipy
  - click
  - The UMI environment with `diffusion_policy` module

### Input Data Structure

The script expects a directory containing:
- `SLAM_traj.txt`: Robot pose data as 3x4 transformation matrices
- `times.txt`: Timestamps for each frame
- Image files matching a pattern (default: `color_*.png`)

### Usage

```bash
# Basic usage
python generate_replay_buffer.py \
  --input /mnt/ldata/data/temp/spi_postproc/z1_rs_calib_lab_2025-01-22-08-08-49 \
  --output /path/to/output/dataset.zarr.zip

# With all options
python generate_replay_buffer.py \
  --input /mnt/ldata/data/temp/spi_postproc/z1_rs_calib_lab_2025-01-22-08-08-49 \
  --output /path/to/output/dataset.zarr.zip \
  --img-pattern "color_*.png" \
  --out-res "224,224" \
  --compression-level 99
```

### Command Line Options

- `--input`, `-i`: Path to the directory containing extracted ROS bag data (required)
- `--output`, `-o`: Path where the zarr dataset will be saved (required)
- `--img-pattern`: Glob pattern to match image files (default: `color_*.png`)
- `--out-res`: Output image resolution as "width,height" (default: `224,224`)
- `--compression-level`, `-cl`: Image compression level (default: 99)

### Output Dataset Structure

The script creates a zarr dataset with the following structure:

- `robot0_eef_pos`: End-effector positions from SLAM trajectory (shape: [n, 3])
- `robot0_eef_rot_axis_angle`: End-effector rotations as axis-angle (shape: [n, 3])
- `robot0_gripper_width`: Gripper widths (set to zeros by default) (shape: [n, 1])
- `camera0_rgb`: RGB images resized to the specified resolution (shape: [n, height, width, 3])
- `timestamp`: Timestamps from the times.txt file (shape: [n])

Where `n` is the number of frames in the dataset.

### Loading the Dataset

You can load the generated dataset using the UMI tools, similar to how it's done in `load_cup_dataset_example.py`:

```python
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register codecs for zarr compression
register_codecs()

# Define shape metadata
shape_meta = {
    'obs': {
        'camera0_rgb': {
            'shape': [3, 224, 224],
            'type': 'rgb',
            'horizon': 2,
            'latency_steps': 0,
            'down_sample_steps': 3
        },
        'robot0_eef_pos': {
            'shape': [3],
            'type': 'low_dim',
            'horizon': 2,
            'latency_steps': 0,
            'down_sample_steps': 3
        },
        # ... other observations
    },
    'action': {
        'shape': [10],  # [position(3), rotation_6d(6), gripper(1)]
        'horizon': 16,
        'latency_steps': 0,
        'down_sample_steps': 3,
        'rotation_rep': 'rotation_6d'
    }
}

# Load the dataset
dataset = UmiDataset(
    shape_meta=shape_meta,
    dataset_path="/path/to/output/dataset.zarr.zip",
    val_ratio=0.0,
)
```

## Notes

- The script assumes that the SLAM trajectory data is provided as 3x4 transformation matrices and converts them to position vectors and axis-angle rotations.
- If there's a mismatch in the number of poses, timestamps, and images, the script will truncate all data to the minimum length.
- The gripper width is set to zero for all frames, as specified in the requirements.
