#!/usr/bin/env python3
"""
Test script for the ROS interpolation controller.
Loads poses from episode_poses.txt and executes the trajectory.
"""

import os
import time
import numpy as np
import pandas as pd
import rospy
import argparse
from umi.real_world.ros_interpolation_controller import ROSInterpolationController
from scipy.spatial.transform import Rotation

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

def main(args):
    # Configure ROS node
    rospy.init_node('test_ros_interpolation_controller', anonymous=True, disable_signals=True)
    
    # Load poses from file
    poses_file = args.poses_file
    if not os.path.exists(poses_file):
        print(f"File not found: {poses_file}")
        return
    
    print(f"Loading poses from {poses_file}...")
    poses, timestamps, gripper_widths = load_poses_from_file(poses_file)
    print(f"Loaded {len(poses)} poses")
    
    # Initialize the controller
    joint_names = args.joint_names.split(',')
    print(f"Using joint names: {joint_names}")
    controller = ROSInterpolationController(
        joint_names=joint_names,
        group_name=args.group_name,
        eef_link=args.eef_link,
        traj_action_name=args.traj_action_name,
        frequency=args.frequency,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        verbose=args.verbose
    )
    
    try:
        # Start the controller
        print("Starting controller...")
        controller.start(wait=True)
        print("Controller started")
        
        # Wait until the controller is ready
        while not controller.is_ready:
            rospy.sleep(0.1)
        print("Controller is ready")
        
        # Get current time as base
        base_time = time.time()
        
        # Adjust timestamps to be relative to current time
        execution_timestamps = base_time + timestamps + args.delay
        
        # Execute trajectory
        print(f"Executing {len(poses)} waypoints...")
        for i in range(len(poses)):
            if args.stop_on_shutdown and rospy.is_shutdown():
                print("ROS shutdown detected, stopping execution")
                break
                
            current_pose = poses[i]
            target_time = execution_timestamps[i]
            
            # Skip waypoints that are in the past
            if target_time <= time.time():
                continue
            
            # Schedule waypoint
            controller.schedule_waypoint(current_pose, target_time)
            
            if i % 10 == 0:  # Print progress every 10 waypoints
                print(f"Scheduled waypoint {i}/{len(poses)}: {current_pose}")
            
            # Optional: slow down scheduling for debugging
            if args.schedule_delay > 0:
                time.sleep(args.schedule_delay)
        
        print("All waypoints scheduled. Waiting for trajectory completion...")
        
        # Wait until the trajectory is expected to finish
        last_timestamp = execution_timestamps[-1] + 2.0  # Add buffer time
        wait_time = last_timestamp - time.time()
        if wait_time > 0:
            time.sleep(wait_time)
            
        print("Trajectory execution completed")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping...")
    finally:
        # Stop the controller
        print("Stopping controller...")
        controller.stop(wait=True)
        print("Controller stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ROS Interpolation Controller')
    parser.add_argument('--poses-file', type=str, 
                        default='/home/zfei/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/episode_poses.txt',
                        help='Path to the file containing poses')
    parser.add_argument('--joint-names', type=str, 
                        default='joint1,joint2,joint3,joint4,joint5,joint6',
                        help='Comma-separated list of joint names')
    parser.add_argument('--group-name', type=str, default='manipulator',
                        help='MoveIt group name for IK/FK')
    parser.add_argument('--eef-link', type=str, default='tool0',
                        help='End effector link name')
    parser.add_argument('--traj-action-name', type=str, 
                        default='/z1_joint_traj_controller/follow_joint_trajectory',
                        help='Joint trajectory action server name')
    parser.add_argument('--frequency', type=float, default=10.0,
                        help='Control frequency (Hz)')
    parser.add_argument('--max-pos-speed', type=float, default=0.25,
                        help='Maximum position speed (m/s)')
    parser.add_argument('--max-rot-speed', type=float, default=0.16,
                        help='Maximum rotation speed (rad/s)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay before starting trajectory (seconds)')
    parser.add_argument('--schedule-delay', type=float, default=0.0,
                        help='Delay between scheduling waypoints (seconds)')
    parser.add_argument('--stop-on-shutdown', action='store_true',
                        help='Stop execution when ROS shutdown is detected')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    main(args)