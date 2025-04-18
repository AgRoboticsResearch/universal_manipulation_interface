#!/usr/bin/env python3
import sys
import os
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st

# Add UMI to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from umi.real_world.ros2_interpolation_controller import ROS2InterpolationController
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait

@click.command()
@click.option('--ns', default='', help='ROS2 namespace for the robot')
@click.option('--controller', default='joint_trajectory_controller', help='Name of the ROS2 controller')
@click.option('-f', '--frequency', type=float, default=30, help='Control frequency in Hz')
@click.option('--joints', help='Comma-separated list of joint names')
def main(ns, controller, frequency, joints):
    """
    Control a ROS2 robot using the UMI architecture with a spacemouse.
    """
    # Configure control parameters
    max_pos_speed = 0.25  # m/s
    max_rot_speed = 0.6   # rad/s
    dt = 1 / frequency
    command_latency = dt / 2
    
    # Parse joint names if provided
    if joints:
        joint_names = [j.strip() for j in joints.split(',')]
    else:
        # Default to a 6-DOF robot arm
        joint_names = None  # Controller will use defaults

    print("Starting ROS2 robot control with UMI architecture")
    print(f"Controller: {controller}")
    print(f"Frequency: {frequency} Hz")
    print(f"Joint names: {joint_names if joints else 'default'}")
    
    with SharedMemoryManager() as shm_manager:
        # Initialize controller and spacemouse
        with ROS2InterpolationController(
            shm_manager=shm_manager,
            node_name='umi_controller' if not ns else f'{ns}_umi_controller',
            controller_name=controller,
            joint_names=joint_names,
            frequency=125,
            lookahead_time=0.1,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            verbose=True
        ) as controller, \
        Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print('Controller ready!')
            print('Use spacemouse to control the robot')
            print('Press Ctrl+C to exit')
            
            # Get initial robot state
            state = controller.get_state()
            target_pose = state['ActualTCPPose'].copy()
            
            print(f"Initial pose: {target_pose}")
            
            # Set up timing
            t_start = time.monotonic()
            iter_idx = 0
            
            try:
                # Main control loop
                while True:
                    # Calculate timing for this cycle
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt
                    
                    # Wait until sample time
                    precise_wait(t_sample)
                    
                    # Read spacemouse state
                    sm_state = sm.get_motion_state_transformed()
                    
                    # Calculate position and rotation deltas
                    dpos = sm_state[:3] * (max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                    
                    # Apply deltas to target pose
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    
                    # Apply rotation change (using rotation vector representation)
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    
                    # Schedule waypoint with precise timing
                    controller.schedule_waypoint(
                        target_pose, 
                        t_command_target - time.monotonic() + time.time()
                    )
                    
                    # Wait until end of cycle
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    
            except KeyboardInterrupt:
                print("\nExiting...")

if __name__ == '__main__':
    main()
