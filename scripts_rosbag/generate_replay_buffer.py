#!/usr/bin/env python
"""
Script to convert ROS bag data to a zarr dataset for use with the Universal Manipulation Interface.
This script extracts data from a ROS bag directory containing:
- color_*.png: RGB images
- SLAM_traj.txt: Robot pose data (3x4 transformation matrices)
- times.txt: Timestamps for the data

The script creates a zarr dataset with the same structure as the ones created 
by the UMI data collection pipeline.
"""

import os
import sys
import glob
import numpy as np
import cv2
import zarr
from tqdm import tqdm
import pathlib
import click

# Add the root directory to the path so we can import the necessary modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from umi.common.cv_util import get_image_transform

# Register codecs for zarr compression
register_codecs()

def load_slam_trajectory(traj_path):
    """
    Load SLAM trajectory from a text file and convert to position and rotation.
    
    Args:
        traj_path: Path to the SLAM_traj.txt file
        
    Returns:
        positions: Array of position vectors [n, 3]
        rotations: Array of rotation matrices [n, 3, 3]
    """
    # Load the trajectory data
    traj = np.loadtxt(traj_path, delimiter=" ")
    # Reshape into Nx3x4 transformation matrices
    traj = traj.reshape(-1, 3, 4)
    
    # Extract positions (translation vectors)
    positions = traj[:, :, 3]  # [n, 3]
    
    # Extract rotation matrices
    rotation_matrices = traj[:, :, :3]  # [n, 3, 3]
    
    # Convert rotation matrices to axis-angle representation
    from scipy.spatial.transform import Rotation
    rotations = []
    for rot_mat in rotation_matrices:
        r = Rotation.from_matrix(rot_mat)
        rotations.append(r.as_rotvec())
    
    rotations = np.array(rotations)  # [n, 3]
    
    return positions, rotations

def load_timestamps(time_path):
    """Load timestamps from a text file."""
    return np.loadtxt(time_path, delimiter=" ")

def load_images(img_dir, img_pattern="color_*.png"):
    """
    Load all images matching the pattern from the directory.
    
    Args:
        img_dir: Directory containing images
        img_pattern: Glob pattern to match image files
        
    Returns:
        List of image paths sorted by name
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, img_pattern)))
    return img_paths

@click.command()
@click.option('--input', '-i', required=True, help='Path to the rosbag extract directory')
@click.option('--output', '-o', required=True, help='Zarr output path')
@click.option('--img-pattern', default="color_*.png", help='Image filename pattern')
@click.option('--out-res', type=str, default='224,224', help='Output image resolution "width,height"')
@click.option('--compression-level', '-cl', default=99, type=int, help='Image compression level')
def main(input, output, img_pattern, out_res, compression_level):
    """Process ROS bag data into a zarr dataset."""
    input_dir = pathlib.Path(input)
    output_path = pathlib.Path(output)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    # Parse output resolution
    out_res = tuple(int(x) for x in out_res.split(','))
    print(f"Output image resolution: {out_res}")
    
    # Load SLAM trajectory
    traj_path = input_dir / "SLAM_traj.txt"
    positions, rotations = load_slam_trajectory(traj_path)
    print(f"Loaded trajectory with {len(positions)} poses")
    
    # Load timestamps
    time_path = input_dir / "times.txt"
    timestamps = load_timestamps(time_path)
    print(f"Loaded {len(timestamps)} timestamps")
    
    # Load image paths
    img_paths = load_images(input_dir, img_pattern)
    print(f"Found {len(img_paths)} images")
    
    # Check that we have the same number of poses, timestamps, and images
    if not (len(positions) == len(timestamps) == len(img_paths)):
        print(f"Warning: Mismatch in data sizes: positions={len(positions)}, timestamps={len(timestamps)}, images={len(img_paths)}")
        # Use the minimum length of all three
        min_len = min(len(positions), len(timestamps), len(img_paths))
        positions = positions[:min_len]
        rotations = rotations[:min_len]
        timestamps = timestamps[:min_len]
        img_paths = img_paths[:min_len]
        print(f"Truncating to minimum length: {min_len}")
    
    # Create an empty zarr replay buffer
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # Create a single episode with all data
    # Set all gripper widths to zero as specified
    gripper_widths = np.zeros((len(positions), 1), dtype=np.float32)
    
    # Prepare episode data
    episode_data = {
        'robot0_eef_pos': positions.astype(np.float32),
        'robot0_eef_rot_axis_angle': rotations.astype(np.float32),
        'robot0_gripper_width': gripper_widths.astype(np.float32),
        'timestamp': timestamps.astype(np.float64)
    }
    
    # Add episode to replay buffer
    out_replay_buffer.add_episode(data=episode_data, compressors=None)
    
    # Set up image dataset in the replay buffer
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    name = 'camera0_rgb'
    _ = out_replay_buffer.data.require_dataset(
        name=name,
        shape=(len(positions),) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )
    img_array = out_replay_buffer.data[name]
    
    # Process images
    print("Processing images...")
    # Get an image to determine input resolution
    sample_img = cv2.imread(img_paths[0])
    in_h, in_w = sample_img.shape[:2]
    
    # Create image transformer for resizing
    resize_tf = get_image_transform(
        in_res=(in_w, in_h),
        out_res=out_res
    )
    
    # Process each image
    for i, img_path in enumerate(tqdm(img_paths)):
        # Read image
        img = cv2.imread(img_path)
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = resize_tf(img)
        # Save to zarr
        img_array[i] = img
    
    # Save replay buffer to disk
    print(f"Saving replay buffer to {output_path}")
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    
    print(f"Done! Saved {len(positions)} frames to {output_path}")

if __name__ == "__main__":
    main()