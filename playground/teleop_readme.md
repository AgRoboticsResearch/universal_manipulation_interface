# Visual Odometry Teleoperation Controller

This package provides a teleoperation system that uses Visual Odometry (VO) estimated poses to control a robot arm. It enables intuitive control by mapping camera movements to robot end-effector movements.

## Overview

The VO Teleoperation Controller subscribes to camera pose data (typically from a SLAM or Visual Odometry system) and uses these pose changes to control a robot arm. This creates an intuitive mapping between camera movements and robot movements, making it easy to precisely position the robot's end effector.

The system leverages the `ROSInterpolationController` for smooth trajectory execution and provides both joystick controls and ROS services for operation.

## Features

- **Camera-based control**: Control the robot by moving a camera
- **Smooth motion**: Interpolated trajectories with configurable speed limits
- **Frame transformations**: Properly handles optical-to-camera transformations
- **Joystick integration**: Toggle following and reset position with joystick buttons
- **ROS service API**: Programmatic control through standard ROS services
- **Motion smoothing**: Configurable smoothing factor for stable movement
- **Visualization**: Publishes markers for RViz visualization

## Requirements

- ROS (tested with ROS Noetic)
- Python 3
- A robot with a follow_joint_trajectory action server
- A camera system publishing PoseStamped messages (e.g., ORBSLAM3)
- MoveIt! configuration for your robot (for IK/FK services)
- (Optional) Joystick/gamepad for manual control

## Installation

The controller is part of the Universal Manipulation Interface (UMI) framework. No additional installation is required if you have UMI installed.

## Usage

### Basic Startup

To launch the teleoperation controller with default settings:

```bash
python teleop_ros_interpolation_controller.py
```

### Command-line Arguments

The controller supports various command-line arguments for customization:

```
--camera-pose-topic TOPIC    ROS topic for camera pose (default: /orbslam3/camera_pose)
--joint-names JOINT_NAMES    Comma-separated list of joint names (default: joint1,joint2,joint3,joint4,joint5,joint6)
--group-name GROUP_NAME      MoveIt group name for IK/FK (default: manipulator)
--eef-link EEF_LINK          End effector link name (default: link06)
--traj-action-name ACTION    Joint trajectory action server name (default: /z1_joint_traj_controller/follow_joint_trajectory)
--frequency HZ               Control frequency in Hz (default: 30.0)
--max-pos-speed SPEED        Maximum position speed in m/s (default: 0.25)
--max-rot-speed SPEED        Maximum rotation speed in rad/s (default: 0.16)
--delay SECONDS              Delay before starting trajectory (default: 1.0)
--smooth-factor FACTOR       Smoothing factor for transitions (0-1, higher is smoother, default: 0)
--verbose                    Enable verbose output
```

### Example Usage

#### Basic Usage with Default Settings

```bash
python teleop_ros_interpolation_controller.py
```

#### Advanced Configuration

```bash
python teleop_ros_interpolation_controller.py --camera-pose-topic /my_camera/pose \
  --joint-names joint_1,joint_2,joint_3,joint_4,joint_5,joint_6 \
  --frequency 60 \
  --max-pos-speed 0.15 \
  --max-rot-speed 0.1 \
  --smooth-factor 0.7
```

## Controls

### Joystick Controls

The controller supports standard gamepad/joystick controls:

- **LB Button (Button 6)**: Toggle camera following on/off
- **RB Button (Button 7)**: Reset to home position

### ROS Services

Two ROS services are available for programmatic control:

#### Toggle Camera Following

```bash
# Enable/disable camera following
rosservice call /vo_teleop_controller/toggle_camera_following "{}"
```

#### Reset to Home

```bash
# Reset robot to home position
rosservice call /vo_teleop_controller/reset_to_home "{}"
```

### Service API in Python

```python
import rospy
from std_srvs.srv import Trigger

# Initialize ROS node
rospy.init_node('teleop_client', anonymous=True)

# Create service proxies
toggle_following = rospy.ServiceProxy('/vo_teleop_controller/toggle_camera_following', Trigger)
reset_home = rospy.ServiceProxy('/vo_teleop_controller/reset_to_home', Trigger)

# Call the services
response = toggle_following()
print(f"Toggle camera following: {response.success}, {response.message}")

response = reset_home()
print(f"Reset to home: {response.success}, {response.message}")
```

## Frame Transformations

The controller handles transformations between different coordinate frames:

1. **Optical Frame**: The frame in which pose estimates are received (typically camera optical frame)
2. **Camera Frame**: Standard camera frame following ROS conventions
3. **Robot Frame**: The reference frame for robot control

The controller automatically converts between optical and camera frames using a predefined transformation. This transformation is configured for standard ROS camera conventions.

## How It Works

### Camera Following Process

1. When camera following is enabled:
   - The current robot pose and camera pose are recorded
   - An offset matrix between camera and robot frames is calculated
   
2. During operation:
   - New camera poses are transformed using the offset matrix
   - Target robot poses are calculated based on camera movement
   - Smooth interpolated trajectories are generated for the robot

3. When disabled:
   - The robot maintains its current position
   - Camera movements are ignored

## Visualization

The controller publishes visualization markers to help monitor the system:

- **Target Pose**: Red arrow showing the current target pose
- **Trajectory Visualization**: Path markers showing planned trajectories

These can be viewed in RViz by adding MarkerArray displays for the topics:
- `/rviz/target_pose`
- `/trajectory_visualization`

## Troubleshooting

### Robot Not Moving

- Ensure MoveIt! services are running
- Check that the camera pose topic is being published
- Verify that camera following is enabled
- Check for error messages in the controller output

### Erratic Movement

- Decrease control frequency (--frequency option)
- Increase smoothing factor (--smooth-factor)
- Reduce maximum speeds (--max-pos-speed and --max-rot-speed)

### Transformation Issues

- Ensure the camera's optical frame has the expected orientation
- Check that the camera pose is being published correctly
- Verify that IK solutions can be found for the target poses

## Advanced Configuration

### Custom Home Position

To modify the home position, edit the `reset_to_home` method in the `VOTeleopController` class:

```python
# Example home position [x, y, z, rx, ry, rz]
home_pose = np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
```

### Custom Camera Transformations

If your camera setup differs from standard ROS conventions, you may need to modify the transformation matrices in the `__init__` method:

```python
# Define transformation matrices between camera frames
self.camera_T_optical_mat = np.eye(4)
self.camera_T_optical_mat[:3, :3] = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
```
