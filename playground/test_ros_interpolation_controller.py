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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point  # Import Point from geometry_msgs.msg
import std_msgs.msg
umi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(umi_path)
from umi.real_world.ros_interpolation_controller import ROSInterpolationController
from scipy.spatial.transform import Rotation
import threading

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

def normalize_poses_to_current_tcp(poses, current_tcp_pose):
    """
    Normalize the loaded poses to align with the robot's current TCP pose
    using homogeneous transformation matrices for accuracy
    
    Parameters:
    -----------
    poses: numpy.ndarray
        Array of poses [x, y, z, rx, ry, rz]
    current_tcp_pose: numpy.ndarray
        Current TCP pose of the robot [x, y, z, rx, ry, rz]
        
    Returns:
    --------
    normalized_poses: numpy.ndarray
        Array of poses normalized relative to current TCP pose
    """
    # Extract the first pose from the loaded poses
    first_pose = poses[0]
    
    # Create homogeneous transformation matrix for the first pose
    first_rot_mat = Rotation.from_euler('xyz', first_pose[3:]).as_matrix()
    first_trans = np.eye(4)
    first_trans[:3, :3] = first_rot_mat
    first_trans[:3, 3] = first_pose[:3]
    
    # Create homogeneous transformation matrix for the current robot pose
    current_rot_mat = Rotation.from_euler('xyz', current_tcp_pose[3:]).as_matrix()
    current_trans = np.eye(4)
    current_trans[:3, :3] = current_rot_mat
    current_trans[:3, 3] = current_tcp_pose[:3]
    
    # Calculate the transformation from first pose to current pose
    # T_rel = T_current * inv(T_first)
    rel_trans = current_trans @ np.linalg.inv(first_trans)
    
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

def publish_trajectory_markers(poses, publisher, frame_id="world", marker_lifetime=10.0):
    """
    Publish trajectory as visualization markers in RViz
    
    Parameters:
    -----------
    poses: numpy.ndarray
        Array of poses [x, y, z, rx, ry, rz]
    publisher: rospy.Publisher
        ROS publisher for MarkerArray
    frame_id: str
        Reference frame for visualization
    marker_lifetime: float
        How long markers should persist (seconds)
    """
    marker_array = MarkerArray()
    
    # Create a marker for the trajectory path (line strip)
    path_marker = Marker()
    path_marker.header.frame_id = frame_id
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = "trajectory_path"
    path_marker.id = 0
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.scale.x = 0.001  # Line width
    path_marker.color.r = 0.0
    path_marker.color.g = 1.0
    path_marker.color.b = 0.0
    path_marker.color.a = 1.0
    path_marker.lifetime = rospy.Duration(marker_lifetime)
    
    # Add points to the path marker
    for pose in poses:
        p = Point()  # Use geometry_msgs.msg.Point
        p.x, p.y, p.z = pose[:3]
        path_marker.points.append(p)
    
    marker_array.markers.append(path_marker)
    
    # Create markers for orientation arrows (every few poses to avoid clutter)
    arrow_stride = max(1, len(poses) // 20)  # Show ~20 arrows along trajectory
    for i in range(0, len(poses), arrow_stride):
        pose = poses[i]
        rot = Rotation.from_euler('xyz', pose[3:])
        rot_matrix = rot.as_matrix()
        
        # Create markers for X, Y, Z axes
        for axis_idx, color in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = rospy.Time.now()
            arrow.ns = f"pose_orientation"
            arrow.id = i * 3 + axis_idx
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.005  # Shaft diameter
            arrow.scale.y = 0.005  # Head diameter
            arrow.scale.z = 0.01  # Head length
            arrow.color.r = color[0]
            arrow.color.g = color[1]
            arrow.color.b = color[2]
            arrow.color.a = 0.7
            arrow.lifetime = rospy.Duration(marker_lifetime)
            
            # Set arrow start point
            arrow.pose.position.x = pose[0]
            arrow.pose.position.y = pose[1]
            arrow.pose.position.z = pose[2]
            
            # Set arrow length and direction (based on rotation matrix)
            axis_direction = rot_matrix[:, axis_idx]
            arrow_length = 0.05  # 5cm arrows
            
            # Set arrow direction (points vector)
            arrow.points.append(Point(x=0, y=0, z=0))  # Use Point directly
            end_point = Point(
                x=axis_direction[0] * arrow_length,
                y=axis_direction[1] * arrow_length,
                z=axis_direction[2] * arrow_length
            )
            arrow.points.append(end_point)
            
            marker_array.markers.append(arrow)
    
    # Numbered waypoints
    for i in range(0, len(poses), arrow_stride):
        pose = poses[i]
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "waypoint_labels"
        text_marker.id = i
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = pose[0]
        text_marker.pose.position.y = pose[1]
        text_marker.pose.position.z = pose[2] + 0.05  # Slightly above the waypoint
        text_marker.scale.z = 0.02  # Text height
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.8
        text_marker.lifetime = rospy.Duration(marker_lifetime)
        text_marker.text = f"{i}"
        
        marker_array.markers.append(text_marker)
    
    # Publish the marker array
    publisher.publish(marker_array)
    print(f"Published trajectory markers with {len(poses)} waypoints")

def save_array_to_txt(array, filename):
    np.savetxt(filename, array, fmt="%.6f", delimiter=",")
    print(f"Saved array to {filename}")

def save_list_to_txt(lst, filename):
    np.savetxt(filename, np.array(lst), fmt="%.6f", delimiter=",")
    print(f"Saved list to {filename}")

def save_array_with_timestamps(array, timestamps, filename):
    array = np.asarray(array)
    timestamps = np.asarray(timestamps).reshape(-1, 1)
    data = np.hstack([timestamps, array])
    np.savetxt(filename, data, fmt="%.6f", delimiter=",")
    print(f"Saved array with timestamps to {filename}")

def periodic_state_sampler(controller, joint_states_list, joint_times_list, tcp_poses_list, tcp_times_list, stop_event, frequency=10.0):
    period = 1.0 / frequency
    while not stop_event.is_set():
        try:
            joint_state = controller.get_current_joint_positions()
            tcp_pose = controller.getActualTCPPose()
            now = time.time()
            joint_states_list.append(joint_state)
            joint_times_list.append(now)
            tcp_poses_list.append(tcp_pose)
            tcp_times_list.append(now)
        except Exception as e:
            print(f"Warning: Failed to get robot state during periodic sampling: {e}")
        time.sleep(period)

def main(args):
    # Configure ROS node
    rospy.init_node('test_ros_interpolation_controller', anonymous=True, disable_signals=True)
    
    # Create RViz visualization publisher
    traj_viz_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=1, latch=True)
    
    # Load poses from file
    poses_file = args.poses_file
    if not os.path.exists(poses_file):
        print(f"File not found: {poses_file}")
        return
    
    print(f"Loading poses from {poses_file}...")
    poses, timestamps, gripper_widths = load_poses_from_file(poses_file)
    print(f"Loaded {len(poses)} poses")

    # Use 10 poses for testing
    # poses = np.asarray([[0.3, 0, 0.5, 0, 0, 0]])
    poses = poses[:100]
    timestamps = timestamps[:100]
    print(f"Using {len(poses)} poses for testing")
    print(f"Poses: {poses}")
    print(f"Timestamps: {timestamps}")


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
    
    # Normalize poses relative to current TCP pose
    # Get current TCP pose to use as reference
    current_tcp_pose = controller.getActualTCPPose()
    print(f"Current TCP pose: {current_tcp_pose}")
    if args.normalize_poses:
        print("Normalizing poses to current TCP pose...")
        poses = normalize_poses_to_current_tcp(poses, current_tcp_pose)
        print(f"Normalized poses: {poses}")
        # Save normalized target poses if eval_track is enabled
        if getattr(args, 'eval_track', False):
            save_array_with_timestamps(poses, timestamps, "./temp/temp_target_poses.txt")
        # Visualize the normalized trajectory in RViz
        print("Publishing trajectory to RViz for visualization...")
        publish_trajectory_markers(poses, traj_viz_pub, frame_id="world", marker_lifetime=50)
        rospy.sleep(0.5)  # Small delay to ensure markers are published

    # Prepare for tracking robot states and poses if eval_track is enabled
    robot_joint_states = []
    robot_joint_times = []
    robot_tcp_poses = []
    robot_tcp_times = []
    sampler_stop_event = threading.Event() if getattr(args, 'eval_track', False) else None
    if getattr(args, 'eval_track', False):
        os.makedirs("./temp", exist_ok=True)
    sampler_thread = None

    try:
        # Start the controller
        print("Starting controller...")
        controller.start(wait=True)
        print("Controller started")
        
        # Wait until the controller is ready
        while not controller.is_ready:
            rospy.sleep(0.1)
        print("Controller is ready")

        # Start periodic sampling thread if eval_track is enabled
        if getattr(args, 'eval_track', False):
            sampler_thread = threading.Thread(
                target=periodic_state_sampler,
                args=(controller, robot_joint_states, robot_joint_times, robot_tcp_poses, robot_tcp_times, sampler_stop_event, args.frequency),
                daemon=True
            )
            sampler_thread.start()

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
            print(f"Scheduled waypoint {i}: {current_pose} at time {target_time}")
            
            if i % 10 == 0:  # Print progress every 10 waypoints
                print(f"Scheduled waypoint {i}/{len(poses)}: {current_pose}")
            
            # Optional: slow down scheduling for debugging
            if args.schedule_delay > 0:
                time.sleep(args.schedule_delay)
        
        print("All waypoints scheduled. Waiting for trajectory completion...")
        
        # Wait until the trajectory is expected to finish
        last_timestamp = execution_timestamps[-1] + 2.0  # Add buffer time
        wait_time = last_timestamp - time.time()
        # wait_time = wait_time + 10
        if wait_time > 0:
            time.sleep(wait_time)
            
        print("Trajectory execution completed")
        
        # Stop periodic sampling thread if running
        if getattr(args, 'eval_track', False) and sampler_thread is not None:
            sampler_stop_event.set()
            sampler_thread.join()
        
        # Save tracked robot states and poses if eval_track is enabled
        if getattr(args, 'eval_track', False):
            save_array_with_timestamps(robot_joint_states, robot_joint_times, "./temp/temp_robot_states.txt")
            save_array_with_timestamps(robot_tcp_poses, robot_tcp_times, "./temp/temp_robot_poses.txt")
        
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
                        default='/codes/unitree_ws/universal_manipulation_interface/visualization/cup_dataset_vis/episode_poses.txt',
                        help='Path to the file containing poses')
    parser.add_argument('--joint-names', type=str, 
                        default='joint1,joint2,joint3,joint4,joint5,joint6',
                        help='Comma-separated list of joint names')
    parser.add_argument('--group-name', type=str, default='manipulator',
                        help='MoveIt group name for IK/FK')
    parser.add_argument('--eef-link', type=str, default='link06',
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
    parser.add_argument('--eval-track', action='store_true',
                        help='Save normalized target poses, robot joint states, and TCP poses during execution')
    parser.add_argument('--no-normalize-poses', dest='normalize_poses', action='store_false',
                        help='Disable pose normalization relative to current TCP pose')
    parser.set_defaults(normalize_poses=True)
    
    args = parser.parse_args()
    main(args)