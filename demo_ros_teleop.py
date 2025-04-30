#!/usr/bin/env python3
"""
Demo script to test ROS environment with SpaceMouse teleoperation.
This script doesn't require any trained policy, just lets you control the robot manually.

Usage:
(umi): python demo_ros_teleop.py -o data_local/test_data --camera_topic /camera_ee_cam

Control:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.
Press button 0 (front) to close gripper.
Press button 1 (back) to open gripper.

Press 'Q' to exit the program.
Press 'R' to start/stop recording.
Press Backspace to drop the current episode.
"""

import os
import sys
import time
import pathlib
import numpy as np
import cv2
import click
import scipy.spatial.transform as st

from multiprocessing.managers import SharedMemoryManager
from umi.common.precise_sleep import precise_wait
from umi.real_world.ros_env import RosEnv
from umi.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode
from umi.real_world.spacemouse_shared_memory import Spacemouse


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    """
    Prevent the robot from colliding with the table.
    Adjusts the height of the end effector if needed.
    """
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta


@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--camera_topic', default='/camera_ee_cam', help='ROS camera topic')
@click.option('--joint_names', default='joint1,joint2,joint3,joint4,joint5,joint6', help='Comma-separated joint names')
@click.option('--group_name', default='manipulator', help='MoveIt group name')
@click.option('--eef_link', default='link06', help='End effector link name')
@click.option('--traj_action_name', default='/z1_joint_traj_controller/follow_joint_trajectory', help='Joint trajectory action name')
@click.option('--vis_camera_idx', default=0, type=int, help='Camera index to visualize')
@click.option('--init_joints', '-j', is_flag=True, default=False, help='Whether to initialize robot joint configuration in the beginning')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
@click.option('--command_latency', '-cl', default=0.01, type=float, help='Latency between receiving SpaceMouse command to executing on Robot in Sec')
@click.option('--no_mirror', is_flag=True, default=False, help='Disable mirror in camera view')
@click.option('--no_camera', is_flag=True, default=False, help='Run without camera visualization (for headless or no camera)')
@click.option('--delay', default=0.1, type=float, help='Delay before executing scheduled waypoint (seconds)')
def main(output, camera_topic, joint_names, group_name, eef_link, traj_action_name, 
         vis_camera_idx, init_joints, frequency, command_latency, no_mirror, no_camera, delay):
    
    max_gripper_width = 0.09  # Maximum gripper width in meters
    gripper_speed = 0.2       # Gripper speed for manual control
    height_threshold = 0.02   # Minimum height above the table
    
    # Parse joint names from command line
    joint_names_list = joint_names.split(',')
    
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
             KeystrokeCounter() as key_counter, \
             RosEnv(
                output_dir=output,
                frequency=frequency,
                joint_names=joint_names_list,
                group_name=group_name,
                eef_link=eef_link,
                traj_action_name=traj_action_name,
                camera_topic=camera_topic,
                obs_image_resolution=(224, 224),
                obs_float32=True,
                init_joints=init_joints,
                # Latency parameters
                camera_obs_latency=0.17,
                robot_obs_latency=0.0001,
                robot_action_latency=0.1,
                # Processing options
                no_mirror=no_mirror,
                # Action speed limits
                max_pos_speed=0.5,
                max_rot_speed=2.0,
                # Video recording
                camera_fps=30,
                video_bit_rate=6000*1000,
                shm_manager=shm_manager,
                require_camera=not no_camera
             ) as env:
            
            cv2.setNumThreads(2)
            if not no_camera:
                print("Waiting for camera...")
                time.sleep(1.0)
            
            recording = False
            print("System ready! Human in control.")
            state = env.get_robot_state()
            target_pose = state['ActualTCPPose']
            gripper_target_pos = max_gripper_width  # Start with open gripper
            
            t_start = time.monotonic()
            iter_idx = 0
            dt = 1/frequency
            
            while True:
                # Calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt + delay  # Add delay to command target time

                # Get observations
                obs = env.get_obs()
                
                # Visualization (skip if no_camera or no image)
                episode_id = env.replay_buffer.n_episodes
                vis_img = None
                if not no_camera:
                    vis_img = obs.get('camera0_rgb', None)
                    if vis_img is not None:
                        vis_img = vis_img[-1]
                        # Add status text with recording indicator
                        status = f'Episode: {episode_id}' + (' RECORDING' if recording else '')
                        cv2.putText(
                            vis_img,
                            status,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            lineType=cv2.LINE_AA,
                            thickness=3,
                            color=(0, 0, 0)
                        )
                        cv2.putText(
                            vis_img,
                            status,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255) if not recording else (0, 255, 0)
                        )
                        
                        # Show robot pose
                        pose_text = f'Pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]'
                        cv2.putText(
                            vis_img,
                            pose_text,
                            (10, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255)
                        )
                        
                        # Show gripper width
                        gripper_text = f'Gripper: {gripper_target_pos:.3f} m'
                        cv2.putText(
                            vis_img,
                            gripper_text,
                            (10, 80),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255)
                        )
                        
                        # Show the visualization
                        cv2.imshow('ROS Teleop', vis_img[...,::-1])  # RGB to BGR for cv2
                        _ = cv2.pollKey()
                
                # Handle keyboard input
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        print("Exiting...")
                        if recording:
                            env.end_episode()
                        return
                    elif key_stroke == KeyCode(char='r'):
                        # Toggle recording
                        if recording:
                            print("Stopping recording.")
                            env.end_episode()
                            recording = False
                        else:
                            print("Starting recording...")
                            env.start_episode()
                            recording = True
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure you want to drop this episode?'):
                            env.drop_episode()
                            recording = False
                            key_counter.clear()

                # --- Reset to home when SpaceMouse button 1 pressed ---
                # Home joint state (update as needed)
                home_joint_state = np.array([
                    -1.426289054506924e-05, 1.5749942064285278, -0.7059323787689209,
                    -0.8982672095298767, -3.4126722312066704e-05, 0.11976243555545807
                ])
                if sm.is_button_pressed(1):
                    print("Resetting arm to home position...")
                    env.robot.move_to_joint_positions(home_joint_state, duration=5.0)
                    time.sleep(delay + 1.0)
                    # Update target_pose to new TCP pose
                    state = env.get_robot_state()
                    target_pose = state['ActualTCPPose'].copy()
                    print("Reset complete.")
                    continue  # Skip rest of loop this cycle

                # Wait until the right time to sample the SpaceMouse state
                precise_wait(t_sample)
                
                # Get teleop command from SpaceMouse
                sm_state = sm.get_motion_state_transformed()
                # print(f"DEBUG: SpaceMouse state: {sm_state}")
                dpos = sm_state[:3] * (0.5 / frequency)
                drot_xyz = sm_state[3:] * (1.5 / frequency)
                
                # Apply the command to the target pose
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
                # print(f"DEBUG: Target pose: {target_pose}")
                
                # Avoid collision with the table
                solve_table_collision(
                    ee_pose=target_pose,
                    gripper_width=gripper_target_pos,
                    height_threshold=height_threshold
                )
                # print(f"DEBUG: Target pose after collision avoid: {target_pose}")

                # Publish target pose for RViz visualization
                env.publish_target_pose(target_pose)

                # Handle gripper control
                dpos = 0
                if sm.is_button_pressed(0):
                    # Close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    # Open gripper
                    dpos = gripper_speed / frequency
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)
                
                # Combine pose and gripper into a single action
                action = np.zeros((7,))
                action[:6] = target_pose
                action[6] = gripper_target_pos
                
                # Execute the teleop command
                env.exec_actions(
                    actions=[action], 
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    compensate_latency=False
                )
                
                # Wait until the end of this control cycle
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()