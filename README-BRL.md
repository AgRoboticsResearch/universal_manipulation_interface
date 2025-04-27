# Universal Manipulation Interface - ROS Integration

This document provides instructions for using the ROS-based components integrated into the Universal Manipulation Interface (UMI) framework.

## Overview

The ROS integration provides:

1. A ROS-compatible environment interface (`RosEnv`) that mimics the API of the original `UmiEnv`
2. A controller for ROS-based robots (`ROSInterpolationController`) 
3. Evaluation and demo scripts to help you test and use these components

## Prerequisites

- ROS (tested with ROS Noetic) with properly configured workspace
- A robot with a ROS interface using `/joint_states` topic and a joint trajectory action server
- Python 3 with the UMI dependencies
- A camera accessible through a ROS topic

## Components

### 1. ROS Interpolation Controller

The `ROSInterpolationController` provides a ROS-compatible controller with an API similar to the `RTDEInterpolationController` used in UMI. It:

- Connects to the robot's trajectory action server
- Provides forward and inverse kinematics using MoveIt services
- Handles trajectory interpolation and scheduling
- Buffers robot state information

Key parameters:

```python
controller = ROSInterpolationController(
    joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
    traj_action_name='/robot_controller/follow_joint_trajectory',
    group_name="manipulator",  # MoveIt group name
    eef_link="tool0",          # End effector link
    frequency=125,             # Control frequency (Hz)
    max_pos_speed=0.25,        # Max position speed (m/s)
    max_rot_speed=0.6          # Max rotation speed (rad/s)
)
```

### 2. ROS Environment

The `RosEnv` class provides an interface compatible with the original `UmiEnv` but using:

- ROS topics for camera image acquisition 
- The `ROSInterpolationController` for robot control
- Video recording capabilities (using NVENC for hardware acceleration)

Key parameters:

```python
env = RosEnv(
    output_dir="path/to/output",
    frequency=20,
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    camera_topic="/camera_ee_cam",
    traj_action_name="/robot_controller/follow_joint_trajectory",
    group_name="manipulator",
    eef_link="tool0"
)
```

### 3. Demo Script - `demo_ros_teleop.py`

A simple script to test robot teleoperation with SpaceMouse control, without requiring a trained policy.

Features:
- Real-time camera visualization from ROS topic
- SpaceMouse control of the robot's end effector
- Optional recording of episodes
- Table collision prevention

Usage:

```bash
python demo_ros_teleop.py -o data_local/test_data --camera_topic /camera_ee_cam
```

Controls:
- SpaceMouse movement controls robot position
- SpaceMouse buttons control gripper open/close
- Press 'R' to start/stop recording
- Press 'Q' to exit

### 4. Evaluation Script - `eval_ros_real.py`

An evaluation script that can:
- Use trained policies from the UMI framework
- Switch between human teleoperation and policy control
- Record episodes with synchronized camera and robot data
- Load reference demonstrations for comparison

Usage:

```bash
# With a trained model
python eval_ros_real.py -i path/to/checkpoint.ckpt -o data_local/test_data --camera_topic /camera_ee_cam

# Without a model (for testing camera and robot setup)
python eval_ros_real.py -i dummy -o data_local/test_data --camera_topic /camera_ee_cam
```

Controls:
- SpaceMouse movement controls robot position in human control mode
- Press 'C' to switch from human control to policy control
- Press 'S' during policy control to switch back to human control
- Press 'Q' to exit

## Configuration

### ROS Services

The system requires two MoveIt services:
- `/compute_ik`: Inverse kinematics service
- `/compute_fk`: Forward kinematics service

These are typically provided by the MoveIt setup for your robot.

### Camera Setup

Configure your camera topic in the environment initialization:

```python
env = RosEnv(
    # ... other parameters ...
    camera_topic="/your_camera_topic",
    # ... other parameters ...
)
```

The camera feed will be resized to the observation resolution and processed to match the format expected by the UMI policies.

## Recording

Episodes are recorded in the specified output directory with:
- Video files in `{output_dir}/videos/{episode_id}/0.mp4`
- Robot and action data in `{output_dir}/replay_buffer.zarr`

## Troubleshooting

1. If the robot controller doesn't move the robot:
   - Check that the joint trajectory action server name is correct
   - Ensure the MoveIt services are available
   - Verify that the joint names match your robot's configuration

2. If the camera feed isn't visible:
   - Check that the camera topic is publishing
   - Verify ROS connections with `rostopic echo /camera_ee_cam`

3. If video recording isn't working:
   - Ensure you have NVENC support on your GPU
   - Check permissions for writing to the output directory

## Example Workflow

1. Test basic connectivity with `demo_ros_teleop.py`
2. Record demonstrations using the demo script
3. Train a policy using the UMI framework with your recorded data
4. Evaluate the policy with `eval_ros_real.py`

## API Compatibility

The `RosEnv` class implements the same API as `UmiEnv`, making it compatible with existing UMI code. Key methods:

- `get_obs()`: Get current observations including camera images
- `exec_actions(actions, timestamps)`: Send actions to the robot
- `start_episode()`, `end_episode()`: Control recording of episodes
