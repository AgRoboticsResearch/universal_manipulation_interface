#!/usr/bin/env python3
import os
import sys
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

# Path to Z1 SDK library
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('umi_robots')
sdk_path = os.path.join(package_path, "libs/z1_sdk/lib")
sys.path.append(sdk_path)
import z1_arm_interface

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    SERVO_L = 1  # Linear motion (Cartesian)
    SERVO_J = 2  # Joint motion
    SCHEDULE_WAYPOINT = 3

45
class Z1InterpolationController(mp.Process):
    """
    Z1 Robot Arm controller with interpolation capabilities.
    Provides similar functionality to the RTDE interpolation controller for UR robots.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager, 
            frequency=125,  # Z1 usually runs at 125Hz
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25,  # m/s
            max_rot_speed=0.16,  # rad/s
            max_joint_speed=0.5,  # rad/s
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            has_gripper=False,
            soft_real_time=False,
            verbose=False,
            get_max_k=128,
            ):
        """
        Initialize Z1 Interpolation Controller.
        
        Args:
            shm_manager: SharedMemoryManager for interprocess communication
            frequency: Control loop frequency in Hz
            lookahead_time: Path smoothing parameter (seconds)
            gain: Position following gain
            max_pos_speed: Maximum position speed (m/s)
            max_rot_speed: Maximum rotation speed (rad/s)
            max_joint_speed: Maximum joint speed (rad/s)
            launch_timeout: Timeout for controller startup (seconds)
            tcp_offset_pose: 6D pose offset for tool center point
            payload_mass: Tool payload mass (kg)
            payload_cog: Center of gravity for the payload [x, y, z]
            joints_init: Initial joint positions
            has_gripper: Whether the robot has a gripper
            soft_real_time: Use real-time scheduling if True
            verbose: Enable verbose logging
            get_max_k: Maximum number of states to retrieve
        """
        # Parameter verification
        assert 0 < frequency <= 500
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        assert 0 < max_joint_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 < payload_mass
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="Z1InterpolationController")
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.max_joint_speed = max_joint_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.has_gripper = has_gripper
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # Build input queue for commands
        example = {
            'cmd': Command.SERVO_L.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        
        # Build state ring buffer
        # Initialize with placeholder data
        example = {
            'actual_pose': np.zeros((6,), dtype=np.float64),  # current Cartesian pose
            'actual_joints': np.zeros((6,), dtype=np.float64),  # current joint positions
            'target_pose': np.zeros((6,), dtype=np.float64),  # target Cartesian pose
            'target_joints': np.zeros((6,), dtype=np.float64),  # target joint positions
            'actual_velocity': np.zeros((6,), dtype=np.float64),  # current Cartesian velocity
            'actual_joint_velocity': np.zeros((6,), dtype=np.float64),  # current joint velocity
            'robot_timestamp': 0.0  # timestamp
        }
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Z1InterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        Linear motion to a target pose.
        
        Args:
            pose: Target pose [roll, pitch, yaw, x, y, z]
            duration: Desired time to reach the target pose (seconds)
        """
        assert self.is_alive()
        assert(duration >= self.dt)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVO_L.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def servoJ(self, joints, duration=0.1):
        """
        Joint motion to target joint positions.
        
        Args:
            joints: Target joint positions (radians)
            duration: Desired time to reach the target joint positions (seconds)
        """
        assert self.is_alive()
        assert(duration >= self.dt)
        joints = np.array(joints)
        assert joints.shape == (6,)

        message = {
            'cmd': Command.SERVO_J.value,
            'target_joints': joints,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        """
        Schedule a waypoint to be reached at a specific time in the future.
        
        Args:
            pose: Target pose [roll, pitch, yaw, x, y, z]
            target_time: Absolute time when the target should be reached (seconds)
        """
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        """
        Get the latest robot state or the last k states.
        
        Args:
            k: Number of states to retrieve
            out: Output buffer
            
        Returns:
            Dictionary containing robot state(s)
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        """
        Get all recorded robot states.
        
        Returns:
            Dictionary containing all robot states
        """
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        """Main control loop running in a separate process."""
        
        # Enable soft real-time scheduling if requested
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # Initialize Z1 arm interface
        arm = z1_arm_interface.ArmInterface(hasGripper=self.has_gripper)
        arm_model = arm._ctrlComp.armModel
        
        try:
            if self.verbose:
                print(f"[Z1InterpolationController] Initializing Z1 arm")
            
            # Start the control loop
            arm.loopOn()

            # Get into joint control mode
            self.switch_to_jointctrl(arm)
            
            # Set parameters if provided
            if self.tcp_offset_pose is not None:
                # TODO: Add TCP offset implementation
                pass
            if self.payload_mass is not None:
                arm_model.addLoad(self.payload_mass)
            
            # Go to initial position if specified
            if self.joints_init is not None:
                if self.verbose:
                    print(f"[Z1InterpolationController] Moving to initial joint position")
                arm.MoveJ(self.joints_init, self.max_joint_speed)
            else:
                # Default to "forward" position
                arm.labelRun("forward")

            # Main control loop
            dt = self.dt
            curr_pose = self.get_actual_pose(arm)
            curr_joints = arm.q.copy()
            
            # Use monotonic time to make sure the control loop never goes backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            
            # Initialize pose interpolator
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            iter_idx = 0
            keep_running = True
            control_mode = "cartesian"  # Default to Cartesian control
            
            while keep_running:
                loop_start_time = time.time()
                
                # Get current robot state
                t_now = time.monotonic()
                curr_pose = self.get_actual_pose(arm)
                curr_joints = arm.q.copy()
                curr_vel = np.zeros(6)  # Placeholder for actual velocity
                curr_joint_vel = np.zeros(6)  # Placeholder for actual joint velocity
                
                # Calculate target pose for this timestep via interpolation
                if control_mode == "cartesian":
                    target_pose = pose_interp(t_now)
                    # Execute movement using MoveL or servoL
                    self.execute_cartesian_move(arm, target_pose, curr_pose)
                else:
                    # Joint control mode
                    # TODO: Implement joint interpolation
                    pass
                
                # Update state dict for the ring buffer
                state = {
                    'actual_pose': curr_pose,
                    'actual_joints': curr_joints,
                    'target_pose': target_pose if control_mode == "cartesian" else np.zeros(6),
                    'target_joints': arm.q.copy() if control_mode != "cartesian" else np.zeros(6),
                    'actual_velocity': curr_vel,
                    'actual_joint_velocity': curr_joint_vel,
                    'robot_timestamp': time.time()
                }
                self.ring_buffer.put(state)
                
                # Fetch commands from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVO_L.value:
                        control_mode = "cartesian"
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        
                        # Calculate trajectory to new target using interpolator
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        
                        if self.verbose:
                            print(f"[Z1InterpolationController] New pose target: {target_pose}, duration: {duration}s")
                    elif cmd == Command.SERVO_J.value:
                        control_mode = "joint"
                        target_joints = command['target_joints']
                        duration = float(command['duration'])
                        
                        # Perform joint move
                        self.execute_joint_move(arm, target_joints)
                        
                        if self.verbose:
                            print(f"[Z1InterpolationController] New joint target: {target_joints}, duration: {duration}s")
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        control_mode = "cartesian"
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        
                        # Translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        print(f"[Z1InterpolationController] Unknown command: {cmd}")
                
                # Sleep to maintain control frequency
                elapsed = time.time() - loop_start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif self.verbose and iter_idx % 100 == 0:
                    print(f"[Z1InterpolationController] Control loop running behind: {elapsed - dt:.4f}s")
                
                # First loop completed successfully
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                if self.verbose and iter_idx % 100 == 0:
                    print(f"[Z1InterpolationController] Actual frequency {1/elapsed:.2f}Hz")

        finally:
            # Cleanup
            if self.verbose:
                print(f"[Z1InterpolationController] Shutting down")
            
            # Set robot to PASSIVE mode for safety
            try:
                arm.setFsm(z1_arm_interface.ArmFSMState.PASSIVE)
            except:
                pass
            
            # Stop the control loop
            try:
                arm.loopOff()
            except:
                pass
                
            self.ready_event.set()
            if self.verbose:
                print(f"[Z1InterpolationController] Shutdown complete")
    
    # ========= utility methods ============
    def switch_to_jointctrl(self, arm):
        """Switch to joint control mode"""
        if arm.getCurrentState() != z1_arm_interface.ArmFSMState.JOINTCTRL:
            if self.verbose:
                print("[Z1InterpolationController] Switching to JOINTCTRL mode")
            arm.setFsm(z1_arm_interface.ArmFSMState.PASSIVE)
            time.sleep(0.2)
            arm.startTrack(z1_arm_interface.ArmFSMState.JOINTCTRL)
            time.sleep(0.5)

    def switch_to_cartesian(self, arm):
        """Switch to Cartesian control mode"""
        if arm.getCurrentState() != z1_arm_interface.ArmFSMState.CARTESIAN:
            if self.verbose:
                print("[Z1InterpolationController] Switching to CARTESIAN mode")
            arm.setFsm(z1_arm_interface.ArmFSMState.PASSIVE)
            time.sleep(0.2)
            arm.startTrack(z1_arm_interface.ArmFSMState.CARTESIAN)
            time.sleep(0.5)

    def get_actual_pose(self, arm):
        """Get current end-effector pose from forward kinematics"""
        # For Z1, we need to compute the forward kinematics from joint angles
        # This is just an example - you may need to adjust based on actual API
        current_joints = arm.q
        return self.forward_kinematics_to_pose(arm, current_joints)
    
    def forward_kinematics_to_pose(self, arm, joint_positions):
        """Convert joint positions to end-effector pose using forward kinematics"""
        # Here we would use the arm_model to do forward kinematics
        # For now, returning a dummy pose (you'd implement this with actual model)
        # Since Z1 arm model should provide forward kinematics
        arm_model = arm._ctrlComp.armModel
        # Forward kinematics should return a homogeneous transformation matrix
        # Convert this to [roll, pitch, yaw, x, y, z] format
        # This is placeholder code - implement with actual SDK calls
        homo_mat = arm_model.forwardKinematics(joint_positions)
        pose = np.zeros(6)
        # TODO: Extract proper pose from homo_mat using the SDK
        return pose
    
    def execute_cartesian_move(self, arm, target_pose, current_pose):
        """Execute a Cartesian move to target pose"""
        # Switch to appropriate mode if needed
        self.switch_to_cartesian(arm)
        
        # For small increments, we can use cartesianCtrlCmd
        # For larger moves, MoveL might be more appropriate
        # This is a simple implementation - you may need to adjust based on your needs
        
        # Calculate pose difference - simplified
        diff = target_pose - current_pose
        directions = np.sign(diff)
        
        # Use cartesian control command for smooth movements
        angular_vel = 0.3
        linear_vel = 0.3
        arm.cartesianCtrlCmd(
            np.append(directions, 0),  # Add gripper direction (0 = no change)
            angular_vel, 
            linear_vel
        )
    
    def execute_joint_move(self, arm, target_joints):
        """Execute a joint move to target joint positions"""
        # Switch to appropriate mode if needed
        self.switch_to_jointctrl(arm)
        
        # Set joint positions directly
        zero_array = np.zeros(6)
        arm.setArmCmd(target_joints, zero_array, zero_array)
        arm.sendRecv()
