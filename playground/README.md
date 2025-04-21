# UMI Cup Arrangement Playground

This directory contains utility scripts for exploring and visualizing the Cup in the Wild dataset.

## Available Scripts

### 1. `load_cup_dataset_example.py`

This script loads and visualizes data from the cup_in_the_wild dataset, demonstrating the proper dataset structure required for training a diffusion policy.

**Features:**
- Examines and displays the structure of the zarr dataset
- Extracts raw image frames and robot pose data from an example episode
- Saves frames as JPEG images and pose information as a CSV-format text file
- Shows how to access the dataset through the UMI dataset interface

**Usage:**
```bash
python playground/load_cup_dataset_example.py
```

**Output:**
- Saves all frames and pose data to `/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/`
- Creates visualization figures showing representative frames with corresponding pose information

### 2. `visualize_poses_3d.py`

This script visualizes the robot end-effector trajectory in 3D space, providing insights into the movement patterns in the cup arrangement task.

**Features:**
- Generates a 3D plot of the robot's trajectory with time-based coloring
- Creates an animated visualization showing the trajectory evolution
- Produces 2D time series plots of position and gripper width
- Highlights significant gripper events (opening/closing)

**Usage:**
```bash
python playground/visualize_poses_3d.py
```

**Output:**
- `trajectory_3d.png`: Static 3D visualization of the robot trajectory
- `trajectory_animation.mp4`: Animated video of the trajectory
- `position_gripper_time_series.png`: 2D plots of position and gripper state over time

## Workflow

The typical workflow for exploring the dataset is:

1. Run `load_cup_dataset_example.py` first to extract and save the raw data from an episode
2. Then run `visualize_poses_3d.py` to visualize the robot trajectory in 3D space

## Notes for Training Diffusion Policy

These scripts demonstrate how the cup_in_the_wild dataset is structured for training a diffusion policy:

- **Observations**: The dataset includes RGB images (camera0_rgb) and robot state (position, rotation, gripper width)
- **Horizon Values**: Image and state observations use horizon=2, actions use horizon=16
- **Down-sampling**: All data uses down_sample_steps=3
- **Rotation Representation**: The model uses 6D rotation representation for actions

The dataset structure shown here matches the requirements for training a diffusion policy using the UMI framework's diffusion_unet_timm_umi_workspace configuration.
