#!/usr/bin/env python3
"""
Script to visualize poses in 3D space
Supports both episode poses (CSV format) and SLAM trajectory (space-delimited format)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation

# Default paths (will be overridden by command line arguments if provided)
DEFAULT_POSE_FILE = "/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/episode_poses.txt"
DEFAULT_OUTPUT_DIR = "/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis"

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize poses in 3D space (supports episode poses and SLAM trajectories)')
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_POSE_FILE,
                        help='Path to the pose data file')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save visualizations')
    parser.add_argument('--type', type=str, choices=['episode', 'slam'], default='auto',
                        help='Type of data to visualize (episode or slam). Auto-detect if not specified.')
    parser.add_argument('--optical-to-robot', action='store_true', default=True,
                        help='Convert optical frame poses (Z forward, X leftward) to robot frame poses (X forward, Z upward)')
    return parser.parse_args()

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

def detect_data_type(file_path):
    """
    Auto-detect whether the file contains episode poses or SLAM trajectory data.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        str: 'episode' or 'slam'
    """
    try:
        # Try to read as CSV first (episode poses)
        df = pd.read_csv(file_path, nrows=1)
        expected_columns = ['frame_idx', 'pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper_width']
        if all(col in df.columns for col in expected_columns):
            return 'episode'
    except:
        pass
    
    try:
        # Try to read as space-delimited array (SLAM trajectory)
        data = np.loadtxt(file_path, delimiter=" ")
        # SLAM trajectory should have 12 columns (3x4 transformation matrix flattened)
        if data.shape[1] == 12:
            return 'slam'
    except:
        pass
    
    # Default to episode if uncertain
    return 'episode'

def main():
    args = parse_args()
    
    # Set paths from command line arguments
    pose_file = args.input
    output_dir = args.output
    data_type = args.type
    convert_to_robot_frame = args.optical_to_robot
    
    # Check if the pose file exists
    if not os.path.exists(pose_file):
        print(f"Error: Pose file {pose_file} not found.")
        return

    # Auto-detect data type if not specified
    if data_type == 'auto':
        data_type = detect_data_type(pose_file)
        print(f"Auto-detected data type: {data_type}")
    
    # Load the pose data based on type
    print(f"Loading {data_type} data from {pose_file}...")
    
    if data_type == 'episode':
        # Load episode poses (CSV format)
        df = pd.read_csv(pose_file)
        positions = df[['pos_x', 'pos_y', 'pos_z']].values
        rotations = df[['rot_x', 'rot_y', 'rot_z']].values
        gripper = df['gripper_width'].values
        frame_indices = df['frame_idx'].values
        has_gripper_data = True
        
    elif data_type == 'slam':
        # Load SLAM trajectory
        positions, rotations = load_slam_trajectory(pose_file)
        gripper = np.zeros(len(positions))  # No gripper data for SLAM
        frame_indices = np.arange(len(positions))
        has_gripper_data = False
        
    else:
        print(f"Error: Unknown data type {data_type}")
        return
    
    # Apply frame conversion if requested
    frame_name = "Optical Frame"
    if convert_to_robot_frame:
        print("Converting from optical frame to robot frame...")
        positions, rotations = convert_optical_to_robot_frame(positions, rotations)
        frame_name = "Robot Frame"
    
    print(f"Using {frame_name} coordinate system")
    
    # Basic statistics
    num_frames = len(positions)
    print(f"Number of frames: {num_frames}")
    
    # Position ranges
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    
    print(f"X range: [{x_min:.4f}, {x_max:.4f}], range: {x_max - x_min:.4f}")
    print(f"Y range: [{y_min:.4f}, {y_max:.4f}], range: {y_max - y_min:.4f}")
    print(f"Z range: [{z_min:.4f}, {z_max:.4f}], range: {z_max - z_min:.4f}")
    
    # Create a static 3D plot of the trajectory
    print("Creating static 3D trajectory plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=np.arange(len(positions)), 
        cmap='viridis', 
        s=10,
        alpha=0.8
    )
    
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    
    # Plot every 20th orientation as a line
    step = 20
    for i in range(0, len(positions), step):
        # Create a small line representing the orientation
        # Scale the rotation axis-angle by a small factor to make it visible
        scale = 0.05
        ax.quiver(
            positions[i, 0], positions[i, 1], positions[i, 2],
            rotations[i, 0] * scale, rotations[i, 1] * scale, rotations[i, 2] * scale,
            color='red', alpha=0.6
        )
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, ax=ax, label='Frame Index')
    
    # Set labels and title
    frame_info = "Robot Frame (X: forward, Y: left, Z: up)" if convert_to_robot_frame else "Optical Frame (X: left, Y: down, Z: forward)"
    ax.set_xlabel(f'X Position\n{frame_info}')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    title = f'{"Robot End-Effector" if data_type == "episode" else "SLAM"} Trajectory ({frame_name})'
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure
    frame_suffix = "robot" if convert_to_robot_frame else "optical"
    static_plot_filename = f'trajectory_3d_{data_type}_{frame_suffix}.png'
    static_plot_path = os.path.join(output_dir, static_plot_filename)
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    print(f"Static plot saved to {static_plot_path}")
    
    # Create an animated version that shows the trajectory over time
    print("Creating animated 3D trajectory...")
    
    # Create a new figure for the animation
    fig_anim = plt.figure(figsize=(12, 10))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    # Set fixed axis limits based on the full trajectory
    margin = 0.1  # Add 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax_anim.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax_anim.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax_anim.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    # Set labels
    ax_anim.set_xlabel(f'X Position\n{frame_info}')
    ax_anim.set_ylabel('Y Position')
    ax_anim.set_zlabel('Z Position')
    ax_anim.set_title(title)
    
    # Initialize empty plots
    trajectory_line, = ax_anim.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    current_point = ax_anim.scatter([], [], [], color='red', s=100)
    
    # Number of frames to show in the animation (fewer for efficiency)
    num_animation_frames = min(500, num_frames)
    frame_skip = max(1, num_frames // num_animation_frames)
    
    # Animation update function
    def update(frame):
        idx = frame * frame_skip
        if idx >= len(positions):
            idx = len(positions) - 1
            
        trajectory_line.set_data(positions[:idx+1, 0], positions[:idx+1, 1])
        trajectory_line.set_3d_properties(positions[:idx+1, 2])
        
        current_point._offsets3d = ([positions[idx, 0]], [positions[idx, 1]], [positions[idx, 2]])
        
        # Update title with frame information and gripper if available
        if has_gripper_data:
            ax_anim.set_title(f'{title} - Frame: {idx}, Gripper Width: {gripper[idx]:.3f}')
        else:
            ax_anim.set_title(f'{title} - Frame: {idx}')
        
        return trajectory_line, current_point
    
    # Create the animation
    anim = FuncAnimation(
        fig_anim, update, frames=num_animation_frames, 
        interval=50, blit=False
    )
    
    # Save the animation
    animation_filename = f'trajectory_animation_{data_type}_{frame_suffix}.mp4'
    animation_path = os.path.join(output_dir, animation_filename)
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(animation_path, writer=writer)
    print(f"Animation saved to {animation_path}")
    
    # Also create a 2D plot showing position and gripper width over time
    if has_gripper_data:
        print("Creating 2D plots of position and gripper width...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        print("Creating 2D plots of position...")
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot positions
    ax1.plot(frame_indices, positions[:, 0], 'r-', label='X Position')
    ax1.plot(frame_indices, positions[:, 1], 'g-', label='Y Position')
    ax1.plot(frame_indices, positions[:, 2], 'b-', label='Z Position')
    ax1.set_ylabel('Position (m)')
    position_title = f'{"End-Effector" if data_type == "episode" else "SLAM"} Position over Time ({frame_name})'
    ax1.set_title(position_title)
    ax1.legend()
    ax1.grid(True)
    
    if has_gripper_data:
        # Plot gripper width
        ax2.plot(frame_indices, gripper, 'k-')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Gripper Width')
        ax2.set_title('Gripper Width over Time')
        ax2.grid(True)
        
        # Highlight where gripper closes (gripper width decreases significantly)
        threshold = 0.02  # Threshold to detect gripper closing
        gripper_diff = np.abs(np.diff(gripper))
        significant_changes = np.where(gripper_diff > threshold)[0]
        if len(significant_changes) > 0:
            for idx in significant_changes:
                ax2.axvline(x=frame_indices[idx], color='red', linestyle='--', alpha=0.5)
                ax1.axvline(x=frame_indices[idx], color='red', linestyle='--', alpha=0.5)
    else:
        ax1.set_xlabel('Frame Index')
    
    plt.tight_layout()
    time_series_filename = f'position_time_series_{data_type}_{frame_suffix}.png' if not has_gripper_data else f'position_gripper_time_series_{data_type}_{frame_suffix}.png'
    time_series_path = os.path.join(output_dir, time_series_filename)
    plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
    print(f"Time series plots saved to {time_series_path}")
    
    print("Visualization complete!")
    
    # Show the plots (will block execution until closed)
    plt.show()

if __name__ == "__main__":
    main()
