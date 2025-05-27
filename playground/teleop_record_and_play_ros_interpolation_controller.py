#!/usr/bin/env python3
"""
Teleoperation script using the ROS interpolation controller with record and replay functionality.
Allows recording camera poses and replaying them to control the robot.

Usage:
- Press 'r' to start recording camera poses
- Press 'f' to finish recording
- Press 'p' to play back the recorded poses
"""

import os
import time
import numpy as np
import pandas as pd
import rospy
import argparse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
import std_msgs.msg
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
import datetime
umi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(umi_path)
from umi.real_world.ros_interpolation_controller import ROSInterpolationController
from scipy.spatial.transform import Rotation
import threading
import copy
import tf2_ros

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

def pose_array_to_pose_stamped(pose_array, frame_id="world", timestamp=None):
    """Convert a pose array [x,y,z,rx,ry,rz] to a PoseStamped message"""
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    
    # Handle timestamp conversion - if it's a float, convert to rospy.Time
    if timestamp is not None:
        if isinstance(timestamp, (int, float)):
            pose_stamped.header.stamp = rospy.Time.from_sec(timestamp)
        else:
            pose_stamped.header.stamp = timestamp
    else:
        pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = pose_array[0]
    pose_stamped.pose.position.y = pose_array[1]
    pose_stamped.pose.position.z = pose_array[2]
    
    # Convert Euler angles to quaternion
    quat = Rotation.from_euler('xyz', pose_array[3:]).as_quat()
    pose_stamped.pose.orientation.x = quat[0]
    pose_stamped.pose.orientation.y = quat[1]
    pose_stamped.pose.orientation.z = quat[2]
    pose_stamped.pose.orientation.w = quat[3]
    
    
    return pose_stamped

def trajectory_to_path_msg(trajectory, timestamps, frame_id="world"):
    """Convert a list of pose arrays to a Path message"""
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rospy.Time.now()
    
    for i, pose_array in enumerate(trajectory):
        # Use timestamp if available, otherwise use current time
        timestamp = rospy.Time.from_sec(timestamps[i]) if i < len(timestamps) else rospy.Time.now()
        pose_stamped = pose_array_to_pose_stamped(pose_array, frame_id, timestamp)
        path_msg.poses.append(pose_stamped)
    
    return path_msg

def save_trajectory_to_file(trajectory, timestamps, file_path, trajectory_name="trajectory"):
    """
    Save trajectory data to a text file
    
    Parameters:
    -----------
    trajectory: list of numpy arrays
        List of pose arrays [x, y, z, rx, ry, rz]
    timestamps: list of float
        List of timestamps corresponding to each pose
    file_path: str
        Full path to save the file
    trajectory_name: str
        Name identifier for the trajectory
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(f"# {trajectory_name} trajectory data\n")
            f.write("# Format: timestamp,x,y,z,rx,ry,rz\n")
            
            for i, pose in enumerate(trajectory):
                timestamp = timestamps[i] if i < len(timestamps) else 0.0
                f.write(f"{timestamp:.6f},{pose[0]:.6f},{pose[1]:.6f},{pose[2]:.6f},"
                       f"{pose[3]:.6f},{pose[4]:.6f},{pose[5]:.6f}\n")
        
        rospy.loginfo(f"Saved {trajectory_name} trajectory to {file_path}")
        return True
    except Exception as e:
        rospy.logerr(f"Failed to save {trajectory_name} trajectory: {e}")
        return False

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
        p = Point()
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
            arrow.points.append(Point(x=0, y=0, z=0))
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
    # print(f"Published trajectory markers with {len(poses)} waypoints")

def pose_msg_to_matrix(pose_msg):
    """Convert a geometry_msgs/Pose to a 4x4 transformation matrix"""
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = Rotation.from_quat([
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w
    ]).as_matrix()
    pose_matrix[:3, 3] = [
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z
    ]
    return pose_matrix

def matrix_to_pose_msg(pose_matrix):
    """Convert a 4x4 transformation matrix to a geometry_msgs/Pose"""
    pose_msg = Pose()
    pose_msg.position.x = pose_matrix[0, 3]
    pose_msg.position.y = pose_matrix[1, 3]
    pose_msg.position.z = pose_matrix[2, 3]
    quat = Rotation.from_matrix(pose_matrix[:3, :3]).as_quat()
    pose_msg.orientation.x = quat[0]
    pose_msg.orientation.y = quat[1]
    pose_msg.orientation.z = quat[2]
    pose_msg.orientation.w = quat[3]
    return pose_msg

def matrix_to_pose_array(pose_matrix):
    """Convert a 4x4 transformation matrix to a pose array [x,y,z,rx,ry,rz]"""
    position = pose_matrix[:3, 3]
    rotation = Rotation.from_matrix(pose_matrix[:3, :3]).as_euler('xyz')
    return np.concatenate([position, rotation])

def pose_array_to_matrix(pose_array):
    """Convert a pose array [x,y,z,rx,ry,rz] to a 4x4 transformation matrix"""
    assert pose_array.shape == (6,), f"Pose array must be 6D, got {pose_array.shape}"
    matrix = np.eye(4)
    matrix[:3, 3] = pose_array[:3]
    matrix[:3, :3] = Rotation.from_euler('xyz', pose_array[3:]).as_matrix()
    return matrix

def calculate_target_pose(origin_pose_matrix, camera_pose_offset_matrix, camera_pose_matrix):
    """
    Calculate target pose for the robot based on camera movement
    
    Parameters:
    -----------
    origin_pose_matrix: numpy.ndarray (4x4)
        Original robot pose matrix
    camera_pose_offset_matrix: numpy.ndarray (4x4)
        Offset matrix between original camera pose and robot pose
    camera_pose_matrix: numpy.ndarray (4x4)
        Current camera pose matrix
        
    Returns:
    --------
    target_pose_array: numpy.ndarray
        Target pose as [x, y, z, rx, ry, rz]
    """
    # Calculate target pose: T_target = T_origin * T_camera_offset * T_camera
    target_pose_matrix = origin_pose_matrix @ camera_pose_offset_matrix @ camera_pose_matrix
    return matrix_to_pose_array(target_pose_matrix)

def create_target_pose_marker_array(target_pose, frame_id="world"):
    """
    Create a MarkerArray for visualizing the target pose as an arrow in RViz.
    
    Parameters:
    -----------
    target_pose: array-like
        Target pose as [x, y, z, rx, ry, rz]
    frame_id: str
        Reference frame for visualization
    
    Returns:
    --------
    marker_array: MarkerArray
        MarkerArray containing a single arrow marker for the target pose
    """
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "target_pose"
    marker.id = 0
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Shaft diameter
    marker.scale.y = 0.01  # Head diameter
    marker.scale.z = 0.01  # Head length
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    # Set position and orientation from target pose
    marker.pose.position.x = target_pose[0]
    marker.pose.position.y = target_pose[1]
    marker.pose.position.z = target_pose[2]
    # Convert euler angles to quaternion for the marker
    quat = Rotation.from_euler('xyz', target_pose[3:]).as_quat()
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker_array.markers.append(marker)
    return marker_array

class VOTeleopController:
    """Visual Odometry Teleoperation Controller with Record and Replay"""
    
    def __init__(self, args):
        """Initialize the VO teleoperation controller"""
        # Store arguments
        self.args = args
        
        # Initialize ROS node
        rospy.loginfo("Initializing VO teleoperation controller with record and replay...")
        
        # Initialize state variables
        self.camera_pose_msg = None
        self.joy_msg = None
        self.origin_pose_matrix = np.eye(4)
        self.camera_pose_offset_matrix = np.eye(4)
        
        # Recording state variables
        self.is_recording = False
        self.is_playing = False
        self.recorded_poses = []
        self.recorded_timestamps = []
        self.playback_start_time = None
        
        # Thread-safe TCP pose tracking
        self.cached_tcp_pose = None
        self.tcp_pose_lock = threading.Lock()
        self.tcp_pose_thread = None
        self.tcp_pose_thread_running = False
        
        # Initialize TF listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Define transformation matrices between camera frames
        self.camera_T_optical_mat = np.eye(4)
        self.camera_T_optical_mat[:3, :3] = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
        self.optical_T_camera_mat = np.linalg.inv(self.camera_T_optical_mat)
        
        # Trajectory tracking for playback analysis
        self.target_trajectory = []
        self.actual_trajectory = []
        self.trajectory_timestamps = []
        
        # Initialize publishers for visualization
        self.traj_viz_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=1, latch=True)
        self.target_pose_pub = rospy.Publisher('/rviz/target_pose', MarkerArray, queue_size=1, latch=True)
        
        # Initialize publishers for trajectory tracking (real-time pose comparison)
        self.target_trajectory_pub = rospy.Publisher('/playback/target_trajectory', PoseStamped, queue_size=1)
        self.actual_trajectory_pub = rospy.Publisher('/playback/actual_trajectory', PoseStamped, queue_size=1)
        
        # Initialize subscribers
        cam_pose_topic = args.camera_pose_topic if args.camera_pose_topic else "/orbslam3/camera_pose"
        rospy.Subscriber(cam_pose_topic, PoseStamped, self.camera_pose_callback)
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        # Initialize ROS services
        self.start_recording_service = rospy.Service(
            '/vo_teleop_controller/start_recording', 
            Trigger, 
            self.start_recording_service_callback
        )
        self.stop_recording_service = rospy.Service(
            '/vo_teleop_controller/stop_recording', 
            Trigger, 
            self.stop_recording_service_callback
        )
        self.start_playback_service = rospy.Service(
            '/vo_teleop_controller/start_playback', 
            Trigger, 
            self.start_playback_service_callback
        )
        self.reset_home_service = rospy.Service(
            '/vo_teleop_controller/reset_to_home', 
            Trigger, 
            self.reset_to_home_service
        )
        rospy.loginfo("ROS services initialized")
        
        # Initialize the controller
        self.init_controller()
        
        # Buffer for visualization
        self.target_pose_buffer = []
        self.target_pose_buffer_size = 100
    
    # Add service callback handlers
    def toggle_camera_following_service(self, req):
        """ROS service callback to toggle camera following"""
        try:
            # Call the existing toggle_camera_following method
            self.toggle_camera_following()
            
            # Create response
            response = TriggerResponse()
            response.success = True
            response.message = f"Camera following {'enabled' if self.follow_camera else 'disabled'}"
            return response
        except Exception as e:
            # Return error response
            response = TriggerResponse()
            response.success = False
            response.message = f"Error toggling camera following: {str(e)}"
            return response
    
    def reset_to_home_service(self, req):
        """ROS service callback to reset robot to home position"""
        try:
            # Call the existing reset_to_home method
            self.reset_to_home()
            
            # Create response
            response = TriggerResponse()
            response.success = True
            response.message = "Robot reset to home position"
            return response
        except Exception as e:
            # Return error response
            response = TriggerResponse()
            response.success = False
            response.message = f"Error resetting to home: {str(e)}"
            return response
    
    def start_recording_service_callback(self, req):
        """ROS service callback to start recording camera poses"""
        try:
            self.start_recording()
            response = TriggerResponse()
            response.success = True
            response.message = "Started recording camera poses"
            return response
        except Exception as e:
            response = TriggerResponse()
            response.success = False
            response.message = f"Error starting recording: {str(e)}"
            return response
    
    def stop_recording_service_callback(self, req):
        """ROS service callback to stop recording camera poses"""
        try:
            self.stop_recording()
            response = TriggerResponse()
            response.success = True
            response.message = "Stopped recording camera poses"
            return response
        except Exception as e:
            response = TriggerResponse()
            response.success = False
            response.message = f"Error stopping recording: {str(e)}"
            return response
    
    def start_playback_service_callback(self, req):
        """ROS service callback to start playback of recorded poses"""
        try:
            self.start_playback()
            response = TriggerResponse()
            response.success = True
            response.message = "Started playback of recorded poses"
            return response
        except Exception as e:
            response = TriggerResponse()
            response.success = False
            response.message = f"Error starting playback: {str(e)}"
            return response
    
    def init_controller(self):
        """Initialize the ROS interpolation controller"""
        rospy.loginfo("Initializing ROS interpolation controller...")
        
        # Parse joint names from args
        joint_names = self.args.joint_names.split(',')
        rospy.loginfo(f"Using joint names: {joint_names}")
        
        # Create controller instance
        self.controller = ROSInterpolationController(
            joint_names=joint_names,
            group_name=self.args.group_name,
            eef_link=self.args.eef_link,
            traj_action_name=self.args.traj_action_name,
            frequency=self.args.frequency,
            max_pos_speed=self.args.max_pos_speed,
            max_rot_speed=self.args.max_rot_speed,
            verbose=self.args.verbose,
            debug=self.args.debug,
            pose_interp_timeout=2.0
        )
        
        # Start the controller
        rospy.loginfo("Starting controller...")
        self.controller.start(wait=True)
        rospy.loginfo("Controller started")
        
        # Wait until the controller is ready
        while not self.controller.is_ready:
            rospy.sleep(0.1)
        rospy.loginfo("Controller is ready")
        
        # Get initial TCP pose
        self.current_tcp_pose = self.controller.getActualTCPPose()
        self.origin_pose_matrix = pose_array_to_matrix(self.current_tcp_pose)
        rospy.loginfo(f"Initial TCP pose: {self.current_tcp_pose}")
        
        # Start TCP pose reading thread for performance optimization
        self.start_tcp_pose_thread()
    
    def camera_pose_callback(self, optical_pose_stamped_msg):
        """Callback for camera pose messages"""
        # Extract pose from the message
        optical_pose = optical_pose_stamped_msg.pose
        
        # Convert from optical frame to camera frame
        optical_pose_mat = pose_msg_to_matrix(optical_pose)
        camera_pose_mat = np.dot(np.dot(self.camera_T_optical_mat, optical_pose_mat), self.optical_T_camera_mat)
        camera_pose = matrix_to_pose_msg(camera_pose_mat)
        
        # Store the camera pose
        self.camera_pose_msg = camera_pose
        
        # If we're recording, store the pose
        if self.is_recording:
            # Calculate target pose based on camera movement
            target_pose = calculate_target_pose(
                self.origin_pose_matrix,
                self.camera_pose_offset_matrix,
                camera_pose_mat
            )
            
            # Add to recorded poses
            self.recorded_poses.append(target_pose.copy())
            self.recorded_timestamps.append(time.time())
            
            if self.args.debug:
                rospy.loginfo(f"Recorded pose {len(self.recorded_poses)} at time {time.time():.3f}")
    
    def joy_callback(self, joy_msg):
        """Callback for joystick messages"""
        self.joy_msg = joy_msg
        
        # Check for new button presses by comparing with previous state
        if hasattr(self, "prev_joy_msg") and self.prev_joy_msg is not None:
            # Button mappings (adapt these to your joystick)
            # Assuming a standard controller with buttons:
            # 0: A, 1: B, 2: X, 3: Y, 6: LB, 7: RB
            
            # Button 0 (A) to start recording
            if joy_msg.buttons[0] == 1 and self.prev_joy_msg.buttons[0] == 0:
                self.start_recording()
            
            # Button 1 (B) to stop recording
            if joy_msg.buttons[1] == 1 and self.prev_joy_msg.buttons[1] == 0:
                self.stop_recording()
            
            # Button 2 (X) to start playback
            if joy_msg.buttons[2] == 1 and self.prev_joy_msg.buttons[2] == 0:
                self.start_playback()
            
            # Button 7 (RB) resets to home position
            if joy_msg.buttons[7] == 1 and self.prev_joy_msg.buttons[7] == 0:
                self.reset_to_home()
        
        # Store current message for next comparison
        self.prev_joy_msg = copy.deepcopy(joy_msg)
    
    def start_recording(self):
        """Start recording camera poses"""
        # Only start if not already recording or playing
        if not self.is_recording and not self.is_playing:
            rospy.loginfo("Starting to record camera poses")
            
            # Reset recorded poses
            self.recorded_poses = []
            self.recorded_timestamps = []
            
            # Save current pose as origin
            self.current_tcp_pose = self.controller.getActualTCPPose()
            self.origin_pose_matrix = pose_array_to_matrix(self.current_tcp_pose)
            
            # Calculate offset between camera and robot
            if self.camera_pose_msg is not None:
                camera_pose_matrix = pose_msg_to_matrix(self.camera_pose_msg)
                self.camera_pose_offset_matrix = np.linalg.inv(camera_pose_matrix)
                rospy.loginfo("Camera pose offset calculated")
            else:
                rospy.logwarn("No camera pose available, using identity for offset")
                self.camera_pose_offset_matrix = np.eye(4)
            
            # Start recording
            self.is_recording = True
        else:
            rospy.logwarn("Cannot start recording - already recording or playing")
    
    def stop_recording(self):
        """Stop recording camera poses"""
        if self.is_recording:
            self.is_recording = False
            
            num_poses = len(self.recorded_poses)
            rospy.loginfo(f"Stopped recording camera poses. Recorded {num_poses} poses")
            
            # Visualize the recorded trajectory
            if num_poses > 0:
                publish_trajectory_markers(
                    np.array(self.recorded_poses),
                    self.traj_viz_pub,
                    frame_id="world",
                    marker_lifetime=30.0  # Show for 30 seconds
                )
                rospy.loginfo("Published recorded trajectory markers")
            else:
                rospy.logwarn("No poses were recorded")
        else:
            rospy.logwarn("Not currently recording")
    
    def start_playback(self):
        """Start playing back recorded poses"""
        if not self.is_recording and not self.is_playing and len(self.recorded_poses) > 0:
            rospy.loginfo(f"Starting playback of {len(self.recorded_poses)} recorded poses")
            self.is_playing = True
            self.playback_start_time = time.time()
            
            # Initialize trajectory tracking for this playback session
            self.target_trajectory = []
            self.actual_trajectory = []
            self.trajectory_timestamps = []
            
            # Publish trajectory visualization
            publish_trajectory_markers(
                np.array(self.recorded_poses),
                self.traj_viz_pub,
                frame_id="world",
                marker_lifetime=30.0
            )
        elif self.is_recording:
            rospy.logwarn("Cannot start playback while recording")
        elif len(self.recorded_poses) == 0:
            rospy.logwarn("No recorded poses to play back")
        else:
            rospy.logwarn("Already playing back poses")
    
    def reset_to_home(self):
        """Reset the robot to home joint state"""
        # Stop recording/playback if active
        self.is_recording = False
        self.is_playing = False
        
        rospy.loginfo("Reset arm pose and returning to home (joint state)")
        # Example home joint state (update as needed)
        home_joint_states = np.array([
            -1.426289054506924e-05, 1.5749942064285278, -0.7059323787689209,
            -0.8982672095298767, -3.4126722312066704e-05, 0.11976243555545807
        ])
        self.controller.move_to_joint_positions(home_joint_states, duration=5.0)
        rospy.sleep(self.args.delay + 1.0)
        self.current_tcp_pose = self.controller.getActualTCPPose()
        self.origin_pose_matrix = pose_array_to_matrix(self.current_tcp_pose)
        if self.camera_pose_msg is not None:
            camera_pose_matrix = pose_msg_to_matrix(self.camera_pose_msg)
            self.camera_pose_offset_matrix = np.linalg.inv(camera_pose_matrix)
        rospy.loginfo("Reset complete, arm's origin pose reset")
        self.controller.reset_pose_interpolator()
    
    def save_playback_trajectories(self):
        """Save target and actual trajectories from playback to files"""
        if len(self.target_trajectory) == 0:
            rospy.logwarn("No trajectory data to save")
            return
        
        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for saving trajectories
        save_dir = os.path.expanduser(self.args.trajectory_save_folder)
        
        # Define file paths
        target_file = os.path.join(save_dir, f"target_trajectory_{timestamp}.txt")
        actual_file = os.path.join(save_dir, f"actual_trajectory_{timestamp}.txt")
        
        # Save target trajectory
        success_target = save_trajectory_to_file(
            self.target_trajectory,
            self.trajectory_timestamps,
            target_file,
            "target"
        )
        
        # Save actual trajectory
        success_actual = save_trajectory_to_file(
            self.actual_trajectory,
            self.trajectory_timestamps,
            actual_file,
            "actual"
        )
        
        if success_target and success_actual:
            rospy.loginfo(f"Successfully saved trajectories to {save_dir}")
        else:
            rospy.logerr("Failed to save one or more trajectory files")
        
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting teleoperation control loop...")
        rate = rospy.Rate(self.args.frequency)
        
        last_target_pose = None
        
        try:
            while not rospy.is_shutdown():
                # Handle playback mode
                if self.is_playing and len(self.recorded_poses) > 0:
                    # Calculate playback progress
                    elapsed_time = time.time() - self.playback_start_time
                    
                    # Check if we've reached the end of recorded poses
                    if len(self.recorded_timestamps) > 0:
                        total_duration = self.recorded_timestamps[-1] - self.recorded_timestamps[0]
                        if elapsed_time > total_duration:
                            rospy.loginfo("Playback complete")
                            self.is_playing = False
                            
                            # Save trajectories if enabled
                            if self.args.save_trajectories and len(self.target_trajectory) > 0:
                                self.save_playback_trajectories()
                            
                            continue
                    
                    # Find the appropriate pose to play based on elapsed time
                    playback_time = self.recorded_timestamps[0] + elapsed_time
                    
                    # Find the closest poses before and after the current time
                    next_idx = 0
                    for i, ts in enumerate(self.recorded_timestamps):
                        if ts > playback_time:
                            next_idx = i
                            break
                    
                    if next_idx == 0:
                        # We're at the beginning, just use the first pose
                        target_pose = self.recorded_poses[0].copy()
                    else:
                        # Interpolate between the two closest poses
                        prev_idx = next_idx - 1
                        prev_time = self.recorded_timestamps[prev_idx]
                        next_time = self.recorded_timestamps[next_idx]
                        prev_pose = self.recorded_poses[prev_idx]
                        next_pose = self.recorded_poses[next_idx]
                        
                        # Linear interpolation factor
                        alpha = (playback_time - prev_time) / (next_time - prev_time)
                        target_pose = prev_pose * (1 - alpha) + next_pose * alpha
                    
                    # Apply smoothing if needed
                    if last_target_pose is not None and self.args.smooth_factor > 0:
                        smooth = self.args.smooth_factor
                        target_pose = smooth * last_target_pose + (1 - smooth) * target_pose
                    
                    # Send the pose to the robot
                    if self.args.debug:
                        print(f"DEBUG: Playback at {elapsed_time:.3f}/{total_duration:.3f}, " +
                              f"pose {next_idx}/{len(self.recorded_poses)}")
                    
                    self.controller.schedule_waypoint(target_pose, time.time() + self.args.delay)
                    
                    # Get actual robot pose for trajectory tracking (using cached pose from thread)
                    current_actual_pose = self.get_cached_tcp_pose()
                    current_time = time.time()
                    
                    # Store trajectory data
                    self.target_trajectory.append(target_pose.copy())
                    self.actual_trajectory.append(current_actual_pose.copy())
                    self.trajectory_timestamps.append(current_time)
                    
                    # Publish current poses for real-time comparison in rqt
                    target_pose_stamped = pose_array_to_pose_stamped(target_pose, frame_id="world", timestamp=current_time)
                    actual_pose_stamped = pose_array_to_pose_stamped(current_actual_pose, frame_id="world", timestamp=current_time)
                    self.target_trajectory_pub.publish(target_pose_stamped)
                    self.actual_trajectory_pub.publish(actual_pose_stamped)
                    
                    # Publish target pose for visualization
                    marker_array = create_target_pose_marker_array(target_pose, frame_id="world")
                    self.target_pose_pub.publish(marker_array)
                    
                    # Store for next iteration
                    last_target_pose = target_pose.copy()
                
                # Handle visualization during recording
                if self.is_recording and len(self.recorded_poses) > 0:
                    # Visualize the current trajectory being recorded
                    publish_trajectory_markers(
                        np.array(self.recorded_poses),
                        self.traj_viz_pub,
                        frame_id="world",
                        marker_lifetime=1.0
                    )
                
                # Maintain control loop frequency
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt detected. Stopping...")
        except Exception as e:
            rospy.logerr(f"Error in control loop: {e}")
        finally:
            # Stop the TCP pose reading thread
            rospy.loginfo("Stopping TCP pose reading thread...")
            self.stop_tcp_pose_thread()
            
            # Stop the controller
            rospy.loginfo("Stopping controller...")
            self.controller.stop(wait=True)
            rospy.loginfo("Controller stopped")
    
    def start_tcp_pose_thread(self):
        """Start the TCP pose reading thread"""
        if not self.tcp_pose_thread_running:
            self.tcp_pose_thread_running = True
            self.tcp_pose_thread = threading.Thread(target=self._tcp_pose_reader_thread)
            self.tcp_pose_thread.daemon = True
            self.tcp_pose_thread.start()
            rospy.loginfo("TCP pose reading thread started")
    
    def stop_tcp_pose_thread(self):
        """Stop the TCP pose reading thread"""
        if self.tcp_pose_thread_running:
            self.tcp_pose_thread_running = False
            if self.tcp_pose_thread and self.tcp_pose_thread.is_alive():
                self.tcp_pose_thread.join(timeout=1.0)
            rospy.loginfo("TCP pose reading thread stopped")
    
    def _tcp_pose_reader_thread(self):
        """Thread function that continuously reads TCP pose"""
        rate = rospy.Rate(self.args.frequency)  # Same frequency as main loop
        
        while self.tcp_pose_thread_running and not rospy.is_shutdown():
            try:
                # Read current TCP pose
                current_pose = self.controller.getActualTCPPose()
                
                # Update cached pose in thread-safe manner
                with self.tcp_pose_lock:
                    self.cached_tcp_pose = current_pose.copy()
                    
            except Exception as e:
                rospy.logwarn(f"Error reading TCP pose in thread: {e}")
                rospy.sleep(0.1)  # Brief sleep on error
                continue
            
            rate.sleep()
    
    def get_cached_tcp_pose(self):
        """Get the cached TCP pose in a thread-safe way"""
        with self.tcp_pose_lock:
            if self.cached_tcp_pose is not None:
                return self.cached_tcp_pose.copy()
            else:
                # Fallback to direct call if no cached pose available
                rospy.logwarn("No cached TCP pose available, falling back to direct call")
                return self.controller.getActualTCPPose()

def main(args):
    # Configure ROS node
    rospy.init_node('vo_teleop_controller', anonymous=False, disable_signals=True)
    
    # Create controller and run
    controller = VOTeleopController(args)
    controller.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record and Replay Teleoperation Controller')
    parser.add_argument('--camera-pose-topic', type=str, 
                        default='/orbslam3/camera_pose',
                        help='ROS topic for camera pose')
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
    parser.add_argument('--frequency', type=float, default=30.0,
                        help='Control frequency (Hz)')
    parser.add_argument('--max-pos-speed', type=float, default=0.3,
                        help='Maximum position speed (m/s)')
    parser.add_argument('--max-rot-speed', type=float, default=0.7,
                        help='Maximum rotation speed (rad/s)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay before starting trajectory (seconds)')
    parser.add_argument('--smooth-factor', type=float, default=0,
                        help='Smoothing factor for pose transitions (0-1, higher is smoother)')
    parser.add_argument('--save-trajectories', action='store_true',
                        help='Save target and actual trajectories to files during playback')
    parser.add_argument('--trajectory-save-folder', type=str, default='~/trajectory_data',
                        help='Folder to save trajectory files (default: ~/trajectory_data)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("""
    Record and Replay Teleoperation Controller
    -----------------------------------------
    Use the following joystick buttons:
    - A (button 0): Start recording poses
    - B (button 1): Finish recording
    - X (button 2): Play recorded poses
    - RB (button 7): Reset to home position
    
    If you're using keyboard input instead of joystick, make sure to map these buttons in your joy node.
    """)
    
    main(args)