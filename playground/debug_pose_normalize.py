#!/usr/bin/env python3
"""
Debug script for visualizing pose normalization.
Loads poses from episode_poses.txt and visualizes the original and normalized trajectories in 3D.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy.spatial.transform import Rotation
import sys

# Add UMI path to system path
umi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(umi_path)


def load_poses_from_file(file_path):
    """
    Load pose data from the episode_poses.txt file
    
    Returns:
    --------
    poses: numpy.ndarray
        Array of poses [x, y, z, rx, ry, rz]
    timestamps: numpy.ndarray
        Array of timestamps
    gripper_widths: numpy.ndarray
        Array of gripper widths
    """
    df = pd.read_csv(file_path)
    
    # Extract position, rotation, and gripper data
    positions = df[['pos_x', 'pos_y', 'pos_z']].values
    rotations = df[['rot_x', 'rot_y', 'rot_z']].values
    gripper_widths = df['gripper_width'].values
    
    # Combine position and rotation into pose [x, y, z, rx, ry, rz]
    poses = np.concatenate([positions, rotations], axis=1)
    
    # Calculate timestamps (assuming 10Hz frequency)
    frame_indices = df['frame_idx'].values
    timestamps = frame_indices / 10.0  # 10Hz
    
    return poses, timestamps, gripper_widths


def normalize_poses_to_reference(poses, reference_pose):
    """
    Normalize the loaded poses to align with a reference pose
    using homogeneous transformation matrices for accuracy
    
    Parameters:
    -----------
    poses: numpy.ndarray
        Array of poses [x, y, z, rx, ry, rz]
    reference_pose: numpy.ndarray
        Reference pose [x, y, z, rx, ry, rz]
        
    Returns:
    --------
    normalized_poses: numpy.ndarray
        Array of poses normalized relative to reference pose
    """
    # Extract the first pose from the loaded poses
    first_pose = poses[0]
    
    # Create homogeneous transformation matrix for the first pose
    first_rot_mat = Rotation.from_euler('xyz', first_pose[3:]).as_matrix()
    first_trans = np.eye(4)
    first_trans[:3, :3] = first_rot_mat
    first_trans[:3, 3] = first_pose[:3]
    
    # Create homogeneous transformation matrix for the reference pose
    ref_rot_mat = Rotation.from_euler('xyz', reference_pose[3:]).as_matrix()
    ref_trans = np.eye(4)
    ref_trans[:3, :3] = ref_rot_mat
    ref_trans[:3, 3] = reference_pose[:3]
    
    # Calculate the transformation from first pose to reference pose
    # T_rel = T_ref * inv(T_first)
    rel_trans = ref_trans @ np.linalg.inv(first_trans)
    
    # Apply the transformation to all poses
    normalized_poses = np.zeros_like(poses)
    
    for i in range(len(poses)):
        # Create transformation matrix for this pose
        pose_rot_mat = Rotation.from_euler('xyz', poses[i, 3:]).as_matrix()
        pose_trans = np.eye(4)
        pose_trans[:3, :3] = pose_rot_mat
        pose_trans[:3, 3] = poses[i, :3]
        
        # Apply the relative transformation
        new_trans = rel_trans @ pose_trans
        
        # Extract position
        normalized_poses[i, :3] = new_trans[:3, 3]
        
        # Extract rotation (convert back to Euler angles)
        new_rot_mat = new_trans[:3, :3]
        normalized_poses[i, 3:] = Rotation.from_matrix(new_rot_mat).as_euler('xyz')
    
    return normalized_poses


def plot_pose_arrows(ax, poses, color, scale=0.05, stride=5, label=None):
    """
    Plot poses as arrows in 3D space. The arrows represent the z-axis of each pose.
    
    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        3D axes to plot on
    poses: numpy.ndarray
        Array of poses [x, y, z, rx, ry, rz]
    color: str
        Color of the arrows
    scale: float
        Scale factor for the arrows
    stride: int
        Plot every nth pose to avoid clutter
    label: str
        Label for the legend
    """
    # Plot positions
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'o-', color=color, markersize=2, alpha=0.5, label=label)
    
    # Plot orientation arrows (z-axis direction)
    for i in range(0, len(poses), stride):
        pose = poses[i]
        pos = pose[:3]
        rot = Rotation.from_euler('xyz', pose[3:])
        
        # Get the z-axis direction (typically pointing forward for end effectors)
        z_axis = rot.as_matrix()[:, 2]  # Third column is the z-axis
        
        # Draw arrow for z-axis
        ax.quiver(pos[0], pos[1], pos[2], 
                  scale * z_axis[0], scale * z_axis[1], scale * z_axis[2],
                  color=color, alpha=0.8)


def visualize_poses(original_poses, normalized_poses, reference_pose, save_path=None):
    """
    Visualize original and normalized poses in 3D
    
    Parameters:
    -----------
    original_poses: numpy.ndarray
        Array of original poses [x, y, z, rx, ry, rz]
    normalized_poses: numpy.ndarray
        Array of normalized poses [x, y, z, rx, ry, rz]
    reference_pose: numpy.ndarray
        Reference pose used for normalization [x, y, z, rx, ry, rz]
    save_path: str
        Path to save the plot (if None, plot will be shown instead)
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create 3D plot for trajectories
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original trajectory
    plot_pose_arrows(ax, original_poses, 'blue', scale=0.05, stride=5, label='Original')
    
    # Plot normalized trajectory
    plot_pose_arrows(ax, normalized_poses, 'red', scale=0.05, stride=5, label='Normalized')
    
    # Plot reference pose
    ax.scatter(reference_pose[0], reference_pose[1], reference_pose[2], 
               color='green', s=100, marker='*', label='Reference Pose')
    
    # Plot first pose of both trajectories
    ax.scatter(original_poses[0, 0], original_poses[0, 1], original_poses[0, 2], 
               color='blue', s=100, marker='o', label='Original Start')
    ax.scatter(normalized_poses[0, 0], normalized_poses[0, 1], normalized_poses[0, 2], 
               color='red', s=100, marker='o', label='Normalized Start')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectory Normalization Visualization')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0

    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Debug and visualize pose normalization')
    parser.add_argument('--poses-file', type=str, 
                        default='/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/episode_poses.txt',
                        help='Path to the file containing poses')
    parser.add_argument('--reference-pose', type=str, default=None,
                        help='Reference pose as comma-separated values [x,y,z,rx,ry,rz], defaults to [0,0,0,0,0,0]')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the visualization (if not specified, plot will be shown)')
    parser.add_argument('--num-poses', type=int, default=100,
                        help='Number of poses to visualize (default: 100)')
    args = parser.parse_args()
    
    # Load poses from file
    if not os.path.exists(args.poses_file):
        print(f"File not found: {args.poses_file}")
        return
    
    print(f"Loading poses from {args.poses_file}...")
    poses, timestamps, gripper_widths = load_poses_from_file(args.poses_file)
    print(f"Loaded {len(poses)} poses")
    
    # Use specified number of poses for visualization
    poses = poses[:args.num_poses]
    print(f"Using {len(poses)} poses for visualization")
    
    # Parse reference pose or use default
    if args.reference_pose:
        reference_pose = np.array([float(x) for x in args.reference_pose.split(',')])
        assert len(reference_pose) == 6, "Reference pose must have 6 elements [x,y,z,rx,ry,rz]"
    else:
        # Default reference pose at origin with identity rotation
        reference_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # reference_pose = poses[0]
    print(f"Reference pose: {reference_pose}")
    
    # Normalize poses
    print("Normalizing poses...")
    normalized_poses = normalize_poses_to_reference(poses, reference_pose)
    print("Normalization complete")
    
    # Print first few original and normalized poses for comparison
    print("\nFirst few original poses:")
    for i in range(min(3, len(poses))):
        print(f"{i}: {poses[i]}")
    
    print("\nFirst few normalized poses:")
    for i in range(min(3, len(normalized_poses))):
        print(f"{i}: {normalized_poses[i]}")
    
    # Calculate statistics
    displacement_original = np.linalg.norm(poses[-1, :3] - poses[0, :3])
    displacement_normalized = np.linalg.norm(normalized_poses[-1, :3] - normalized_poses[0, :3])
    
    print(f"\nOriginal trajectory start-end displacement: {displacement_original:.4f} m")
    print(f"Normalized trajectory start-end displacement: {displacement_normalized:.4f} m")
    
    # Visualize original and normalized poses
    print("\nVisualizing poses...")
    visualize_poses(poses, normalized_poses, reference_pose, args.save_path)
    print("Visualization complete")


if __name__ == "__main__":
    main()