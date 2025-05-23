#!/usr/bin/env python3
"""
Script to load and visualize an example episode from the cup_in_the_wild dataset
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm
import cv2
import argparse

# Add the root directory to the path so we can import the necessary modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register codecs for zarr compression, suppress warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    register_codecs()

# Default paths (will be overridden by command line arguments if provided)
DEFAULT_DATASET_PATH = "/media/zfei/d/tempdata/umi/cup_in_the_wild/cup_in_the_wild.zarr.zip"
DEFAULT_VIS_DIR = "/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis"

DEFAULT_SAVE_DATA = True

def parse_args():
    parser = argparse.ArgumentParser(description='Load and visualize the cup_in_the_wild dataset')
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_DATASET_PATH,
                        help='Path to the dataset zarr file')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_VIS_DIR,
                        help='Directory to save visualizations')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save data to output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set paths from command line arguments
    dataset_path = args.input
    vis_dir = args.output
    
    # Set save data flag
    SAVE_DATA = not args.no_save
    
    print(f"Loading dataset from {dataset_path}")
    
    # Define shape_meta for the UMI dataset (cup arrangement)
    shape_meta = {
        'obs': {
            'camera0_rgb': {  # Note: corrected key from camera_0 to camera0_rgb to match dataset
                'shape': [3, 224, 224],  # [C, H, W] - Standard size used by pretrained models
                'type': 'rgb',
                'horizon': 2,  # According to config: img_obs_horizon = 2
                'latency_steps': 0,
                'down_sample_steps': 3  # According to config: obs_down_sample_steps = 3
            },
            'robot0_eef_pos': {
                'shape': [3],  # XYZ position
                'type': 'low_dim',
                'horizon': 2,  # According to config: low_dim_obs_horizon = 2
                'latency_steps': 0,
                'down_sample_steps': 3
            },
            'robot0_eef_rot_axis_angle': {
                'shape': [3],  # Axis-angle rotation
                'type': 'low_dim',
                'horizon': 2,  # According to config: low_dim_obs_horizon = 2
                'latency_steps': 0,
                'down_sample_steps': 3
            },
            'robot0_gripper_width': {
                'shape': [1],  # Gripper opening width
                'type': 'low_dim',
                'horizon': 2,  # According to config: low_dim_obs_horizon = 2
                'latency_steps': 0,
                'down_sample_steps': 3
            },
            'robot0_eef_rot_axis_angle_wrt_start': {
                'raw_shape': [3],
                'shape': [6],  # Using 6D rotation representation
                'type': 'low_dim',
                'horizon': 2,  # According to config: low_dim_obs_horizon = 2
                'latency_steps': 0,
                'down_sample_steps': 3
            }
        },
        'action': {
            'shape': [10],  # [position(3), rotation_6d(6), gripper(1)]
            'horizon': 16,  # According to config: action_horizon = 16
            'latency_steps': 0,
            'down_sample_steps': 3,
            'rotation_rep': 'rotation_6d'  # According to config: rotation_6d instead of axis_angle
        }
    }

    # First, let's directly examine the zarr structure to better understand the dataset
    print("Examining zarr structure...")
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        root = zarr.group(store=zip_store)
        
        # Print the structure of the zarr file
        print("\nDataset structure:")
        def print_zarr_structure(group, prefix=""):
            for key, value in group.items():
                if isinstance(value, zarr.Group):
                    print(f"{prefix}{key}/")
                    print_zarr_structure(value, prefix + "  ")
                else:
                    print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
        
        print_zarr_structure(root)
        
        # Get episode boundaries
        episodes = root['meta/episode_ends'][:]
        print(f"\nTotal episodes: {len(episodes)}")
        
        # Load the dataset using the UmiDataset class
        print("\nLoading dataset using UmiDataset...")
        dataset = UmiDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            val_ratio=0.0,  # Don't split validation for this example
        )
        
        # Let's load and visualize a single episode (first episode for simplicity)
        episode_idx = 0
        start_idx = 0 if episode_idx == 0 else episodes[episode_idx-1]
        end_idx = episodes[episode_idx]
        
        print(f"\nVisualization of Episode {episode_idx} (frames {start_idx} to {end_idx-1})")
        
        # Get raw data directly from replay buffer for visualization
        rgb_data = root['data/camera0_rgb'][start_idx:end_idx]
        positions = root['data/robot0_eef_pos'][start_idx:end_idx]
        rotations = root['data/robot0_eef_rot_axis_angle'][start_idx:end_idx]
        gripper = root['data/robot0_gripper_width'][start_idx:end_idx]
        
        # Create a directory to save visualizations
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Visualization directory: {vis_dir}")
        
        if SAVE_DATA:
            # First, save all image frames as JPEG files
            print(f"\nSaving all {end_idx - start_idx} frames as JPEG images...")
            for i in range(end_idx - start_idx):
                img = rgb_data[i]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                # Save the image as JPEG
                cv2.imwrite(os.path.join(vis_dir, f"frame_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Print progress every 100 frames
                if (i + 1) % 100 == 0:
                    print(f"  Saved {i+1}/{end_idx - start_idx} frames")
            
            # Save pose information as text file
            print(f"\nSaving pose information to text file...")
            pose_file = os.path.join(vis_dir, "episode_poses.txt")
            with open(pose_file, 'w') as f:
                f.write("frame_idx,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,gripper_width\n")
                for i in range(end_idx - start_idx):
                    pos = positions[i]
                    rot = rotations[i]
                    grip = gripper[i][0]  # Extract scalar from [1] array
                    f.write(f"{i},{pos[0]},{pos[1]},{pos[2]},{rot[0]},{rot[1]},{rot[2]},{grip}\n")
            
            # Also create a few visualization figures with pose information
            print(f"\nCreating visualization figures with pose information...")
            num_vis_frames = min(10, end_idx - start_idx)  # Show 10 frames or all if fewer
            for i in range(num_vis_frames):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Display the image
                img = rgb_data[i]
                ax1.imshow(img)
                ax1.set_title(f"Frame {start_idx + i}")
                ax1.axis('off')
                
                # Display robot state information
                ax2.axis('off')
                ax2.text(0.1, 0.7, f"Position: {positions[i]}", fontsize=12)
                ax2.text(0.1, 0.6, f"Rotation: {rotations[i]}", fontsize=12)
                ax2.text(0.1, 0.5, f"Gripper: {gripper[i]}", fontsize=12)
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"vis_frame_{i:04d}.png"))
                plt.close()
            
            print(f"\nAll data saved to {vis_dir}/:")
            
        # Sample access via the dataset interface (like during training)
        print("\nAccessing sample through dataset interface...")
        sample = dataset[0]  # Get first sample
        print("Sample keys:", sample.keys())
        
        # Print sample observation shapes
        print("\nSample observation shapes:")
        for k, v in sample['obs'].items():
            print(f"  {k}: {v.shape}")
        print(f"Action shape: {sample['action'].shape}")

if __name__ == "__main__":
    main()
