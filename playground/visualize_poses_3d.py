#!/usr/bin/env python3
"""
Script to visualize episode poses in 3D space
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

# Default paths (will be overridden by command line arguments if provided)
DEFAULT_POSE_FILE = "/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/episode_poses.txt"
DEFAULT_OUTPUT_DIR = "/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis"

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize episode poses in 3D space')
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_POSE_FILE,
                        help='Path to the pose data file')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set paths from command line arguments
    pose_file = args.input
    output_dir = args.output
    
    # Check if the pose file exists
    if not os.path.exists(pose_file):
        print(f"Error: Pose file {pose_file} not found. Run load_cup_dataset_example.py first.")
        return

    # Load the pose data
    print(f"Loading pose data from {pose_file}...")
    df = pd.read_csv(pose_file)
    
    # Basic statistics
    num_frames = len(df)
    print(f"Number of frames: {num_frames}")
    
    # Position ranges
    x_min, x_max = df['pos_x'].min(), df['pos_x'].max()
    y_min, y_max = df['pos_y'].min(), df['pos_y'].max()
    z_min, z_max = df['pos_z'].min(), df['pos_z'].max()
    
    print(f"X range: [{x_min:.4f}, {x_max:.4f}], range: {x_max - x_min:.4f}")
    print(f"Y range: [{y_min:.4f}, {y_max:.4f}], range: {y_max - y_min:.4f}")
    print(f"Z range: [{z_min:.4f}, {z_max:.4f}], range: {z_max - z_min:.4f}")
    
    # Extract the data
    positions = df[['pos_x', 'pos_y', 'pos_z']].values
    rotations = df[['rot_x', 'rot_y', 'rot_z']].values
    gripper = df['gripper_width'].values
    
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
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Robot End-Effector Trajectory')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure
    static_plot_path = os.path.join(output_dir, 'trajectory_3d.png')
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
    ax_anim.set_xlabel('X Position')
    ax_anim.set_ylabel('Y Position')
    ax_anim.set_zlabel('Z Position')
    ax_anim.set_title('Robot End-Effector Trajectory')
    
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
        
        # Update title with gripper information
        ax_anim.set_title(f'Robot End-Effector Trajectory - Frame: {idx}, Gripper Width: {gripper[idx]:.3f}')
        
        return trajectory_line, current_point
    
    # Create the animation
    anim = FuncAnimation(
        fig_anim, update, frames=num_animation_frames, 
        interval=50, blit=False
    )
    
    # Save the animation
    animation_path = os.path.join(output_dir, 'trajectory_animation.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(animation_path, writer=writer)
    print(f"Animation saved to {animation_path}")
    
    # Also create a 2D plot showing position and gripper width over time
    print("Creating 2D plots of position and gripper width...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot positions
    ax1.plot(df['frame_idx'], df['pos_x'], 'r-', label='X Position')
    ax1.plot(df['frame_idx'], df['pos_y'], 'g-', label='Y Position')
    ax1.plot(df['frame_idx'], df['pos_z'], 'b-', label='Z Position')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('End-Effector Position over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot gripper width
    ax2.plot(df['frame_idx'], df['gripper_width'], 'k-')
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
            ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    time_series_path = os.path.join(output_dir, 'position_gripper_time_series.png')
    plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
    print(f"Time series plots saved to {time_series_path}")
    
    print("Visualization complete!")
    
    # Show the plots (will block execution until closed)
    plt.show()

if __name__ == "__main__":
    main()
