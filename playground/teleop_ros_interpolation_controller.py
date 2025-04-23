#!/usr/bin/env python3
"""
Teleoperation script using the ROS interpolation controller.
Subscribes to VO estimated camera poses and controls robot through ROSInterpolationController.
"""

import os
import time
import numpy as np
import pandas as pd
import rospy
import argparse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped
import std_msgs.msg
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger, TriggerResponse
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
    print(f"Published trajectory markers with {len(poses)} waypoints")

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

class VOTeleopController:
    """Visual Odometry Teleoperation Controller"""
    
    def __init__(self, args):
        """Initialize the VO teleoperation controller"""
        # Store arguments
        self.args = args
        
        # Initialize ROS node
        rospy.loginfo("Initializing VO teleoperation controller...")
        
        # Initialize state variables
        self.follow_camera = False
        self.camera_pose_msg = None
        self.joy_msg = None
        self.origin_pose_matrix = np.eye(4)
        self.camera_pose_offset_matrix = np.eye(4)
        
        # Initialize TF listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Define transformation matrices between camera frames
        self.camera_T_optical_mat = np.eye(4)
        self.camera_T_optical_mat[:3, :3] = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
        self.optical_T_camera_mat = np.linalg.inv(self.camera_T_optical_mat)
        
        # Initialize publishers for visualization
        self.traj_viz_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=1, latch=True)
        self.target_pose_pub = rospy.Publisher('/rviz/target_pose', MarkerArray, queue_size=1, latch=True)
        
        # Initialize subscribers
        cam_pose_topic = args.camera_pose_topic if args.camera_pose_topic else "/orbslam3/camera_pose"
        rospy.Subscriber(cam_pose_topic, PoseStamped, self.camera_pose_callback)
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        # Initialize ROS services
        self.toggle_camera_service = rospy.Service(
            '/vo_teleop_controller/toggle_camera_following', 
            Trigger, 
            self.toggle_camera_following_service
        )
        self.reset_home_service = rospy.Service(
            '/vo_teleop_controller/reset_to_home', 
            Trigger, 
            self.reset_to_home_service
        )
        rospy.loginfo("ROS services initialized")
        
        # Initialize the controller
        self.init_controller()
        
        # Buffer for last 100 target poses
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
            verbose=self.args.verbose
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
    
    def joy_callback(self, joy_msg):
        """Callback for joystick messages"""
        self.joy_msg = joy_msg
        
        # Button LB (index 6) toggles camera following
        if joy_msg.buttons[6] == 1:
            self.toggle_camera_following()
        
        # Button RB (index 7) resets to home position
        if joy_msg.buttons[7] == 1:
            self.reset_to_home()
    
    def toggle_camera_following(self):
        """Toggle camera following on/off"""
        self.follow_camera = not self.follow_camera
        
        if self.follow_camera:
            rospy.loginfo("Starting camera following")
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
        else:
            rospy.loginfo("Stopping camera following")
    
    def reset_to_home(self):
        """Reset the robot to home position"""
        # Stop following
        self.follow_camera = False
        rospy.loginfo("Reset arm pose and returning to home")
        
        # Move to home position - use a simple predefined pose
        home_pose = np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])  # Example home position
        
        # Schedule the waypoint with a delay
        self.controller.schedule_waypoint(home_pose, time.time() + self.args.delay)
        
        # Wait for movement to complete
        rospy.sleep(self.args.delay + 1.0)
        
        # Reset origin and offset
        self.current_tcp_pose = self.controller.getActualTCPPose()
        self.origin_pose_matrix = pose_array_to_matrix(self.current_tcp_pose)
        
        # Reset camera offset if camera pose is available
        if self.camera_pose_msg is not None:
            camera_pose_matrix = pose_msg_to_matrix(self.camera_pose_msg)
            self.camera_pose_offset_matrix = np.linalg.inv(camera_pose_matrix)
            
        rospy.loginfo("Reset complete, arm's origin pose reset")
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting teleoperation control loop...")
        rate = rospy.Rate(self.args.frequency)
        
        last_target_pose = None
        
        try:
            while not rospy.is_shutdown():
                # If following camera and we have a camera pose
                if self.follow_camera and self.camera_pose_msg is not None:
                    # Calculate target pose based on camera movement
                    camera_pose_matrix = pose_msg_to_matrix(self.camera_pose_msg)
                    target_pose = calculate_target_pose(
                        self.origin_pose_matrix,
                        self.camera_pose_offset_matrix,
                        camera_pose_matrix
                    )
                    
                    # Apply smoothing if needed
                    if last_target_pose is not None and self.args.smooth_factor > 0:
                        smooth = self.args.smooth_factor
                        target_pose = smooth * last_target_pose + (1 - smooth) * target_pose
                    
                    # Schedule waypoint with the controller
                    # Add a small delay for stability
                    delay = 0.5
                    self.controller.schedule_waypoint(target_pose, time.time() + delay)
                    
                    # Publish target pose for visualization
                    marker_array = MarkerArray()
                    marker = Marker()
                    marker.header.frame_id = "world"
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
                    marker.color.b = 0.0
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
                    self.target_pose_pub.publish(marker_array)
                    
                    # --- Moving window trajectory visualization ---
                    self.target_pose_buffer.append(target_pose.copy())
                    if len(self.target_pose_buffer) > self.target_pose_buffer_size:
                        self.target_pose_buffer.pop(0)
                    if len(self.target_pose_buffer) > 1:
                        publish_trajectory_markers(
                            np.array(self.target_pose_buffer),
                            self.traj_viz_pub,
                            frame_id="world",
                            marker_lifetime=0.2
                        )
                    
                    # Store last target pose for smoothing
                    last_target_pose = target_pose.copy()
                
                # Maintain control loop frequency
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt detected. Stopping...")
        except Exception as e:
            rospy.logerr(f"Error in control loop: {e}")
        finally:
            # Stop the controller
            rospy.loginfo("Stopping controller...")
            self.controller.stop(wait=True)
            rospy.loginfo("Controller stopped")

def main(args):
    # Configure ROS node
    rospy.init_node('vo_teleop_controller', anonymous=False, disable_signals=True)
    
    # Create controller and run
    controller = VOTeleopController(args)
    controller.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VO Teleoperation Controller')
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
    parser.add_argument('--max-pos-speed', type=float, default=0.25,
                        help='Maximum position speed (m/s)')
    parser.add_argument('--max-rot-speed', type=float, default=0.16,
                        help='Maximum rotation speed (rad/s)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay before starting trajectory (seconds)')
    parser.add_argument('--smooth-factor', type=float, default=0,
                        help='Smoothing factor for pose transitions (0-1, higher is smoother)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    main(args)