"""
Usage:
(umi): python eval_ros_real.py -i data/outputs/model_checkpoint.ckpt -o data_local/test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.ros_env import RosEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action
)
from umi.real_world.spacemouse_shared_memory import Spacemouse

OmegaConf.register_new_resolver("eval", eval, replace=True)

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
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--camera_topic', default='/camera_ee_cam', help='ROS camera topic')
@click.option('--joint_names', default='joint1,joint2,joint3,joint4,joint5,joint6', help='Comma-separated joint names')
@click.option('--group_name', default='manipulator', help='MoveIt group name')
@click.option('--eef_link', default='link06', help='End effector link name')
@click.option('--traj_action_name', default='/z1_joint_traj_controller/follow_joint_trajectory', help='Joint trajectory action name')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, type=int, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
@click.option('--no_mirror', is_flag=True, default=False)
@click.option('--sim_fov', type=float, default=None)
@click.option('--camera_intrinsics', type=str, default=None)
def main(
    input, output, camera_topic, joint_names, group_name, eef_link, traj_action_name,
    match_dataset, match_episode, match_camera, vis_camera_idx, init_joints, 
    steps_per_inference, max_duration, frequency, command_latency,
    no_mirror, sim_fov, camera_intrinsics):
    
    max_gripper_width = 0.09  # Maximum gripper width in meters
    gripper_speed = 0.2       # Gripper speed for manual control
    height_threshold = 0.02   # Minimum height above the table

    # Load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("Model name:", cfg.policy.obs_encoder.model_name)
    print("Dataset path:", cfg.task.dataset.dataset_path)

    # Setup experiment timing
    dt = 1/frequency

    # Get observation resolution from model config
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    
    # Load fisheye converter if specified
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print(f"Steps per inference: {steps_per_inference}")
    
    # Parse joint names
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
                obs_image_resolution=obs_res,
                obs_float32=True,
                init_joints=init_joints,
                # Latency parameters
                camera_obs_latency=0.17,
                robot_obs_latency=0.0001,
                robot_action_latency=0.1,
                # Observation horizons from model config
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                # Processing options
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                # Action speed limits
                max_pos_speed=0.5,
                max_rot_speed=2.0,
                # Video recording
                camera_fps=30,
                video_bit_rate=6000*1000,
                shm_manager=shm_manager
            ) as env:
            
            cv2.setNumThreads(2)
            print("Waiting for camera...")
            time.sleep(1.0)

            # Load match dataset for overlaying reference frames
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frames for {len(episode_first_frame_map)} episodes")

            # Create and initialize policy model
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
                policy = workspace.ema_model
            policy.num_inference_steps = 16  # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('Observation pose representation:', obs_pose_rep)
            print('Action pose representation:', action_pose_repr)

            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference...")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10  # 6 for position + rotation, 4 for auxillary outputs
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7  # 6 for position + rotation, 1 for gripper
                del result

            print('System ready!')
            
            while True:
                # ========= Human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                gripper_target_pos = max_gripper_width  # Start with open gripper
                
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # Get observations
                    obs = env.get_obs()

                    # Visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera0_rgb'][-1]
                    
                    # Overlay reference frame if requested
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    
                    # Add text overlay with episode information
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0, 0, 0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255)
                    )
                    
                    # Show the visualization
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    
                    # Handle keyboard input
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Previous episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # Move to the position from the reference dataset
                            if match_replay_buffer is not None and match_episode_id < match_replay_buffer.n_episodes:
                                duration = 3.0
                                ep = match_replay_buffer.get_episode(match_episode_id)
                                pos = ep['robot0_eef_pos'][0]
                                rot = ep['robot0_eef_rot_axis_angle'][0]
                                grip = ep['robot0_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robot.servoL(pose, duration=duration)
                                time.sleep(duration)
                                target_pose = pose
                                gripper_target_pos = grip
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure you want to drop this episode?'):
                                env.drop_episode()
                                key_counter.clear()
                    
                    if start_policy:
                        break

                    # Wait until the right time to sample the SpaceMouse state
                    precise_wait(t_sample)
                    
                    # Get teleop command from SpaceMouse
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    # Apply the command to the target pose
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
                    
                    # Avoid collision with the table
                    solve_table_collision(
                        ee_pose=target_pose,
                        gripper_width=gripper_target_pos,
                        height_threshold=height_threshold
                    )
                    
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
                
                # ========== Policy control loop ==============
                try:
                    # Start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # Wait for a small period to get the closest frame
                    # This reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy control started!")
                    iter_idx = 0
                    
                    while True:
                        # Calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # Get observations
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Observation latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        # Run inference with the policy
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print(f'Inference latency: {time.time() - s:.3f}s')
                        
                        # Process the policy's actions
                        this_target_poses = action
                        
                        # Prevent collision with the table for each pose
                        for target_pose in this_target_poses:
                            solve_table_collision(
                                ee_pose=target_pose[:6],
                                gripper_width=target_pose[6],
                                height_threshold=height_threshold
                            )

                        # Handle timing for action execution
                        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        
                        if np.sum(is_new) == 0:
                            # Exceeded time budget, still do something with the last action
                            this_target_poses = this_target_poses[[-1]]
                            # Schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print(f'Over time budget: {action_timestamp - curr_time:.3f}s')
                            action_timestamps = np.array([action_timestamp])
                        else:
                            # Use only future actions
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # Execute the actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # Visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs['camera0_rgb'][-1]
                        
                        # Add status text
                        text = f'Episode: {episode_id}, Time: {time.monotonic() - t_start:.1f}s'
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        # Check for keyboard input
                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode and hand control back to human
                                print('Stopping policy control.')
                                stop_episode = True

                        # Check if we've reached the maximum duration
                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Maximum duration reached.")
                            stop_episode = True
                            
                        if stop_episode:
                            env.end_episode()
                            break

                        # Wait until the next cycle
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # Stop robot and end episode
                    env.end_episode()
                
                print("Policy control stopped.")

if __name__ == '__main__':
    main()