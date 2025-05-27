#!/usr/bin/env python
"""
Script to convert ROS bag data to a zarr dataset for use with the Universal Manipulation Interface.
This script extracts data from ROS bag directories containing:
- color_*.png: RGB images
- SLAM_traj.txt: Robot pose data (3x4 transformation matrices)
- times.txt: Timestamps for the data

The script creates a zarr dataset with the same structure as the ones created 
by the UMI data collection pipeline.

The script now supports multiple input directories using glob patterns to create
a single zarr file with multiple episodes.
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
from scipy.spatial.transform import Rotation

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
    rotations = []
    for rot_mat in rotation_matrices:
        r = Rotation.from_matrix(rot_mat)
        rotations.append(r.as_rotvec())
    
    rotations = np.array(rotations)  # [n, 3]
    
    return positions, rotations

def convert_optical_to_robot_frame(positions, rotations):
    """
    Convert poses from optical frame to robot frame.
    
    Optical frame: Z forward, X leftward, Y downward
    Robot frame: X forward, Z upward, Y leftward
    
    Transformation matrix:
    R = [[0, 0, 1],   # Robot X = Optical Z
         [-1, 0, 0],  # Robot Y = -Optical X  
         [0, -1, 0]]  # Robot Z = -Optical Y
    
    Args:
        positions: Array of position vectors [n, 3] in optical frame
        rotations: Array of rotation vectors [n, 3] in optical frame (axis-angle)
        
    Returns:
        positions_robot: Array of position vectors [n, 3] in robot frame
        rotations_robot: Array of rotation vectors [n, 3] in robot frame (axis-angle)
    """
    # Transformation matrix from optical to robot frame
    T_optical_to_robot = np.array([
        [0, 0, 1],    # Robot X = Optical Z
        [-1, 0, 0],   # Robot Y = -Optical X
        [0, -1, 0]    # Robot Z = -Optical Y
    ])
    
    # Transform positions
    positions_robot = positions @ T_optical_to_robot.T
    
    # Transform rotations
    rotations_robot = []
    for rot_vec in rotations:
        # Convert axis-angle to rotation matrix
        r = Rotation.from_rotvec(rot_vec)
        rot_mat_optical = r.as_matrix()
        
        # Transform rotation matrix: R_robot = T * R_optical * T^-1
        rot_mat_robot = T_optical_to_robot @ rot_mat_optical @ T_optical_to_robot.T
        
        # Convert back to axis-angle
        r_robot = Rotation.from_matrix(rot_mat_robot)
        rotations_robot.append(r_robot.as_rotvec())
    
    rotations_robot = np.array(rotations_robot)
    
    return positions_robot, rotations_robot

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

def process_single_episode(input_dir, img_pattern, out_res, optical_to_robot):
    """
    Process a single episode directory and return the episode data.
    
    Args:
        input_dir: Path to the episode directory
        img_pattern: Pattern for image files
        out_res: Output resolution tuple
        optical_to_robot: Whether to convert from optical to robot frame
        
    Returns:
        episode_data: Dictionary containing episode data
        images: List of processed images
    """
    print(f"Processing episode: {input_dir}")
    
    # Load SLAM trajectory
    traj_path = input_dir / "SLAM_traj.txt"
    if not traj_path.exists():
        raise ValueError(f"SLAM_traj.txt not found in {input_dir}")
    
    positions, rotations = load_slam_trajectory(traj_path)
    print(f"  Loaded trajectory with {len(positions)} poses")
    
    # Apply frame conversion if requested
    if optical_to_robot:
        positions, rotations = convert_optical_to_robot_frame(positions, rotations)
    
    # Load timestamps
    time_path = input_dir / "times.txt"
    if not time_path.exists():
        raise ValueError(f"times.txt not found in {input_dir}")
    
    timestamps = load_timestamps(time_path)
    print(f"  Loaded {len(timestamps)} timestamps")
    
    # Load image paths
    img_paths = load_images(input_dir, img_pattern)
    print(f"  Found {len(img_paths)} images")
    
    # Check that we have the same number of poses, timestamps, and images
    if not (len(positions) == len(timestamps) == len(img_paths)):
        print(f"  Warning: Mismatch in data sizes: positions={len(positions)}, timestamps={len(timestamps)}, images={len(img_paths)}")
        # Use the minimum length of all three
        min_len = min(len(positions), len(timestamps), len(img_paths))
        positions = positions[:min_len]
        rotations = rotations[:min_len]
        timestamps = timestamps[:min_len]
        img_paths = img_paths[:min_len]
        print(f"  Truncating to minimum length: {min_len}")
    
    # Set all gripper widths to zero as specified
    gripper_widths = np.zeros((len(positions), 1), dtype=np.float32)
    
    # Create demo_start_pose and demo_end_pose
    # Use the first pose as the demo_start_pose and last pose as demo_end_pose
    demo_start_pose = np.zeros((len(positions), 6), dtype=np.float32)
    demo_start_pose[:] = np.concatenate([positions[0], rotations[0]])
    
    demo_end_pose = np.zeros((len(positions), 6), dtype=np.float32)
    demo_end_pose[:] = np.concatenate([positions[-1], rotations[-1]])
    
    # Prepare episode data
    episode_data = {
        'robot0_eef_pos': positions.astype(np.float32),
        'robot0_eef_rot_axis_angle': rotations.astype(np.float32),
        'robot0_gripper_width': gripper_widths.astype(np.float32),
        'robot0_demo_start_pose': demo_start_pose.astype(np.float32),
        'robot0_demo_end_pose': demo_end_pose.astype(np.float32),
        'timestamp': timestamps.astype(np.float64)
    }
    
    # Process images
    print("  Processing images...")
    images = []
    
    # Get an image to determine input resolution
    sample_img = cv2.imread(img_paths[0])
    in_h, in_w = sample_img.shape[:2]
    
    # Create image transformer for resizing
    resize_tf = get_image_transform(
        in_res=(in_w, in_h),
        out_res=out_res
    )
    
    # Process each image
    for img_path in tqdm(img_paths, desc="  Images"):
        # Read image
        img = cv2.imread(img_path)
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = resize_tf(img)
        images.append(img)
    
    return episode_data, images

@click.command()
@click.option('--input', '-i', required=True, help='Path to the rosbag extract directory or glob pattern for multiple directories')
@click.option('--output', '-o', required=True, help='Zarr output path')
@click.option('--img-pattern', default="color_*.png", help='Image filename pattern')
@click.option('--out-res', type=str, default='224,224', help='Output image resolution "width,height"')
@click.option('--compression-level', '-cl', default=99, type=int, help='Image compression level')
@click.option('--optical-to-robot', default=True, type=bool, help='Convert optical frame poses (Z forward, X leftward) to robot frame poses (X forward, Z upward)')
def main(input, output, img_pattern, out_res, compression_level, optical_to_robot):
    """Process ROS bag data into a zarr dataset."""
    output_path = pathlib.Path(output)
    
    # Parse output resolution
    out_res = tuple(int(x) for x in out_res.split(','))
    print(f"Output image resolution: {out_res}")
    
    # Find all input directories matching the pattern
    input_dirs = []
    if '*' in input or '?' in input or '[' in input:
        # Use glob to find matching directories
        matched_paths = glob.glob(input)
        for path in matched_paths:
            if os.path.isdir(path):
                input_dirs.append(pathlib.Path(path))
        input_dirs.sort()  # Sort for consistent ordering
    else:
        # Single directory
        input_path = pathlib.Path(input)
        if not input_path.exists():
            raise ValueError(f"Input directory {input_path} does not exist")
        input_dirs = [input_path]
    
    if not input_dirs:
        raise ValueError(f"No directories found matching pattern: {input}")
    
    print(f"Found {len(input_dirs)} directories to process:")
    for dir_path in input_dirs:
        print(f"  {dir_path}")
    
    # Apply frame conversion if requested
    if optical_to_robot:
        print("Converting from optical frame to robot frame...")
        print("Using Robot Frame coordinate system")
    else:
        print("Using Optical Frame coordinate system")
    
    # Create an empty zarr replay buffer
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # Process each episode directory
    all_images = []
    total_frames = 0
    
    for episode_idx, input_dir in enumerate(input_dirs):
        print(f"\nProcessing episode {episode_idx + 1}/{len(input_dirs)}")
        
        try:
            episode_data, images = process_single_episode(
                input_dir, img_pattern, out_res, optical_to_robot)
            
            # Add episode to replay buffer
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # Store images for later processing
            all_images.extend(images)
            total_frames += len(images)
            
            print(f"  Added episode with {len(images)} frames")
            
        except Exception as e:
            print(f"  Error processing {input_dir}: {e}")
            continue
    
    if total_frames == 0:
        raise ValueError("No valid episodes were processed")
    
    # Set up image dataset in the replay buffer
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    name = 'camera0_rgb'
    _ = out_replay_buffer.data.require_dataset(
        name=name,
        shape=(total_frames,) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )
    img_array = out_replay_buffer.data[name]
    
    # Save all images to the dataset
    print(f"\nSaving {total_frames} images to zarr dataset...")
    for i, img in enumerate(tqdm(all_images, desc="Saving images")):
        img_array[i] = img
    
    # Save replay buffer to disk
    print(f"\nSaving replay buffer to {output_path}")
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    
    print(f"Done! Saved {len(input_dirs)} episodes with {total_frames} total frames to {output_path}")

if __name__ == "__main__":
    main()