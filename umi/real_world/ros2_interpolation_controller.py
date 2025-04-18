import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
import threading
import queue

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import RealtimeCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from controller_manager_msgs.srv import SwitchController
from tf2_ros import TransformBroadcaster

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class ROS2InterpolationController(mp.Process):
    """
    ROS2 implementation of the UMI interpolation controller pattern.
    Sends commands to a ROS2 Control interface with predictable latency.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager, 
            node_name='umi_controller',
            controller_name='joint_trajectory_controller',
            joint_names=None,
            base_frame='base_link',
            end_effector_frame='tool0',
            frequency=100, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25,  # m/s
            max_rot_speed=0.16,  # rad/s
            launch_timeout=3,
            tcp_offset_pose=None,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=None,
            receive_latency=0.0
            ):
        """
        Parameters:
        -----------
        shm_manager: SharedMemoryManager
            Shared memory manager for inter-process communication
        node_name: str
            Name of the ROS2 node
        controller_name: str
            Name of the ROS2 controller to send commands to
        joint_names: list of str
            List of joint names to control
        base_frame: str
            Name of the robot's base frame
        end_effector_frame: str
            Name of the robot's end-effector frame
        frequency: float
            Control loop frequency (Hz)
        lookahead_time: float
            Time horizon for trajectory smoothing
        gain: float
            Proportional gain for position control
        max_pos_speed: float
            Maximum positional speed (m/s)
        max_rot_speed: float
            Maximum rotational speed (rad/s)
        launch_timeout: float
            Timeout for controller initialization
        tcp_offset_pose: array-like, shape (6,)
            TCP offset pose [x, y, z, rx, ry, rz]
        soft_real_time: bool
            Enable soft real-time scheduling
        verbose: bool
            Enable verbose logging
        receive_keys: list of str
            List of state keys to receive
        get_max_k: int
            Maximum number of states to store in the ring buffer
        receive_latency: float
            Latency compensation for state measurements
        """
        
        # Validate parameters
        assert frequency > 0, "Frequency must be positive"
        assert max_pos_speed > 0, "Maximum positional speed must be positive"
        assert max_rot_speed > 0, "Maximum rotational speed must be positive"
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,), "TCP offset pose must be 6-dimensional"

        super().__init__(name="ROS2InterpolationController")
        self.node_name = node_name
        self.controller_name = controller_name
        
        if joint_names is None:
            # Default joint names for common robots
            self.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
        else:
            self.joint_names = joint_names
            
        self.base_frame = base_frame
        self.end_effector_frame = end_effector_frame
        self.frequency = frequency
        self.dt = 1.0/frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.receive_latency = receive_latency

        # Initialize synchronization primitives
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ros_initialized = False
        self.current_joint_state = None
        self.current_tcp_pose = None
        
        # Set up shared memory for multi-process communication
        if get_max_k is None:
            get_max_k = int(frequency * 5)  # Store 5 seconds of data
            
        # Initialize examples for shared memory
        example_state = {
            'ActualTCPPose': np.zeros(6, dtype=np.float64),
            'ActualTCPSpeed': np.zeros(6, dtype=np.float64),
            'ActualQ': np.zeros(len(self.joint_names), dtype=np.float64),
            'ActualQd': np.zeros(len(self.joint_names), dtype=np.float64),
            'TargetTCPPose': np.zeros(6, dtype=np.float64),
            'TargetQ': np.zeros(len(self.joint_names), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        
        example_cmd = {
            'cmd': Command.STOP.value, 
            'target_pose': np.zeros(6, dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        
        # Create shared memory structures
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=256
        )
        
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
    
    # ========= ROS2 Utilities =========
    def _init_ros2(self):
        """Initialize ROS2 node and interfaces"""
        if rclpy.ok():
            self.node = rclpy.create_node(
                self.node_name,
                parameter_overrides=[],
                allow_undeclared_parameters=True,
                automatically_declare_parameters_from_overrides=True
            )
        else:
            rclpy.init()
            self.node = rclpy.create_node(
                self.node_name,
                parameter_overrides=[],
                allow_undeclared_parameters=True,
                automatically_declare_parameters_from_overrides=True
            )
            
        # Create callback groups for thread safety
        self.rt_callback_group = RealtimeCallbackGroup()
        self.default_callback_group = MutuallyExclusiveCallbackGroup()
            
        # Create action client for trajectory execution
        self.action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            f'/{self.controller_name}/follow_joint_trajectory',
            callback_group=self.default_callback_group
        )
        
        # Subscribe to joint states
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.rt_callback_group
        )
        
        # Create transform broadcaster for publishing TCP transforms
        self.tf_broadcaster = TransformBroadcaster(self.node)
        
        # Initialize controller interface
        self.switch_controller_client = self.node.create_client(
            SwitchController,
            '/controller_manager/switch_controller',
            callback_group=self.default_callback_group
        )
        
        # Set up the executor for processing callbacks
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        
        # Start spinning in a separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Wait for the action server
        if not self.action_client.wait_for_server(timeout_sec=self.launch_timeout):
            raise RuntimeError(f"Timeout waiting for action server: {self.controller_name}")
            
        self.node.get_logger().info("ROS2 node initialized successfully")
        self.ros_initialized = True
    
    def _spin_ros(self):
        """Spin ROS2 executor in a separate thread"""
        while not self.stop_event.is_set():
            self.executor.spin_once(timeout_sec=0.01)
    
    def _joint_state_callback(self, msg):
        """Callback for processing joint state messages"""
        # Filter out the joints we care about
        indices = []
        for name in self.joint_names:
            if name in msg.name:
                indices.append(msg.name.index(name))
            else:
                if self.verbose:
                    self.node.get_logger().warn(f"Joint {name} not found in joint state message")
                return
        
        # Create a filtered joint state
        filtered_state = JointState()
        filtered_state.header = msg.header
        filtered_state.name = self.joint_names
        filtered_state.position = [msg.position[i] for i in indices]
        
        if len(msg.velocity) > 0:
            filtered_state.velocity = [msg.velocity[i] for i in indices]
        else:
            filtered_state.velocity = [0.0] * len(self.joint_names)
            
        if len(msg.effort) > 0:
            filtered_state.effort = [msg.effort[i] for i in indices]
        
        self.current_joint_state = filtered_state
        
        # Update TCP pose using forward kinematics
        # Note: In a real implementation, this would use a proper FK solver
        # For now we'll leave this as a TODO
        # self.current_tcp_pose = self._forward_kinematics(filtered_state.position)
    
    def _forward_kinematics(self, joint_positions):
        """
        Compute forward kinematics to get end-effector pose from joint positions
        
        In a real implementation, this would use a proper FK solver from a library
        like KDL or MoveIt. For now, this is a placeholder.
        
        Returns: 6-DOF pose [x, y, z, rx, ry, rz]
        """
        # TODO: Implement proper FK
        # This is where you'd either:
        # 1. Use KDL to compute FK
        # 2. Call a MoveIt service
        # 3. Use tf2 to look up the transform
        
        # For now, returning dummy data as placeholder
        return np.zeros(6)
    
    def _inverse_kinematics(self, target_pose):
        """
        Compute inverse kinematics to get joint positions from end-effector pose
        
        In a real implementation, this would use a proper IK solver from a library
        like KDL or MoveIt. For now, this is a placeholder.
        
        Parameters:
        -----------
        target_pose: array-like, shape (6,)
            Target pose [x, y, z, rx, ry, rz]
            
        Returns:
        --------
        joint_positions: array-like
            Target joint positions
        """
        # TODO: Implement proper IK
        # This is where you'd either:
        # 1. Use KDL to compute IK
        # 2. Call a MoveIt service
        
        # For now, returning dummy data as placeholder
        return np.zeros(len(self.joint_names))
        
    def _send_trajectory(self, target_pose, duration):
        """
        Send a trajectory to the controller
        
        Parameters:
        -----------
        target_pose: array-like, shape (6,)
            Target pose [x, y, z, rx, ry, rz]
        duration: float
            Time to reach the target pose
        """
        if not self.ros_initialized:
            self.node.get_logger().error("ROS2 not initialized")
            return False
        
        # Convert pose to joint positions using inverse kinematics
        target_joints = self._inverse_kinematics(target_pose)
        
        # Create the trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Add trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal_msg.trajectory.points.append(point)
        
        # Send the goal
        self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._trajectory_feedback_callback
        )
        
        return True
    
    def _trajectory_feedback_callback(self, feedback_msg):
        """Process trajectory feedback"""
        pass  # We don't need to do anything with the feedback for now

    # ========= Public API =========
    def start(self, wait=True):
        """Start the controller process"""
        super().start()
        if wait:
            self.ready_event.wait(self.launch_timeout)
            if not self.ready_event.is_set():
                raise RuntimeError(f"Controller failed to start within timeout: {self.launch_timeout}s")
        if self.verbose:
            print(f"[ROS2InterpolationController] Controller process started with PID: {self.pid}")

    def stop(self, wait=True):
        """Stop the controller process"""
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        self.stop_event.set()
        if wait:
            self.join()

    def servoL(self, pose, duration=0.1):
        """
        Move to pose directly with specified duration
        
        Parameters:
        -----------
        pose: array-like, shape (6,)
            Target pose [x, y, z, rx, ry, rz]
        duration: float
            Desired time to reach pose
        """
        assert self.is_alive(), "Controller process is not running"
        assert duration >= (1/self.frequency), "Duration must be at least one control cycle"
        
        pose = np.array(pose)
        assert pose.shape == (6,), "Pose must be 6-dimensional"

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        """
        Schedule a waypoint at a specific time
        
        Parameters:
        -----------
        pose: array-like, shape (6,)
            Target pose [x, y, z, rx, ry, rz]
        target_time: float
            Time at which the pose should be reached (in system time)
        """
        pose = np.array(pose)
        assert pose.shape == (6,), "Pose must be 6-dimensional"

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= State access methods =========
    def get_state(self, k=None, out=None):
        """
        Get the current state or the last k states
        
        Parameters:
        -----------
        k: int, optional
            Number of states to retrieve
        out: dict, optional
            Dictionary to store the result in
            
        Returns:
        --------
        state: dict
            Current robot state
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        """Get all states in the buffer"""
        return self.ring_buffer.get_all()

    # ========= Context manager interface =========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    # ========= Main process loop =========
    def run(self):
        """Main control loop in separate process"""
        # Enable soft real-time if requested
        if self.soft_real_time and hasattr(os, 'sched_setscheduler'):
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                print(f"Failed to set real-time scheduler: {e}")

        try:
            # Initialize ROS2
            self._init_ros2()
            
            # Wait for the first joint state message
            wait_start = time.monotonic()
            while self.current_joint_state is None:
                time.sleep(0.01)
                if time.monotonic() - wait_start > self.launch_timeout:
                    raise RuntimeError("Timeout waiting for first joint state message")
            
            # Initialize interpolator with current state
            curr_t = time.monotonic()
            curr_pose = np.zeros(6)  # Placeholder until we implement FK
            
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            # Signal that we're ready
            self.ready_event.set()
            
            # Main control loop
            t_start = time.monotonic()
            iter_idx = 0
            last_waypoint_time = curr_t
            
            while not self.stop_event.is_set():
                t_cycle_start = time.monotonic()
                
                # Process incoming commands
                try:
                    command = self.input_queue.get()
                    cmd = command['cmd']
                    
                    if cmd == Command.STOP.value:
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = command['duration']
                        self._send_trajectory(target_pose, duration)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = command['target_time']
                        
                        # Convert from system time to monotonic for internal use
                        target_time = target_time - time.time() + time.monotonic()
                        
                        # Update the interpolator
                        pose_interp = pose_interp.schedule_waypoint(
                            time=target_time,
                            pose=target_pose,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = target_time
                except Empty:
                    pass  # No commands to process
                
                # Get the current time and interpolated pose
                t_now = time.monotonic()
                target_pose = pose_interp.get_pose(t_now)
                
                # Send the pose command at control frequency
                if t_now - t_start >= iter_idx * self.dt:
                    self._send_trajectory(target_pose, self.lookahead_time)
                    iter_idx += 1
                
                # Update robot state in the ring buffer
                if self.current_joint_state is not None:
                    state = {
                        'ActualTCPPose': np.zeros(6),  # Placeholder until we implement FK
                        'ActualTCPSpeed': np.zeros(6),  # Placeholder
                        'ActualQ': np.array(self.current_joint_state.position),
                        'ActualQd': np.array(self.current_joint_state.velocity),
                        'TargetTCPPose': target_pose,
                        'TargetQ': self._inverse_kinematics(target_pose),
                        'robot_receive_timestamp': time.time(),
                        'robot_timestamp': time.time() - self.receive_latency
                    }
                    self.ring_buffer.put(state)
                
                # Sleep to maintain control frequency
                t_elapsed = time.monotonic() - t_cycle_start
                sleep_time = max(0, self.dt - t_elapsed)
                
                if sleep_time > 0:
                    if self.soft_real_time:
                        precise_wait(t_cycle_start + self.dt)
                    else:
                        time.sleep(sleep_time)
                elif self.verbose and t_elapsed > self.dt * 1.1:
                    print(f"[ROS2InterpolationController] Control loop overran: {t_elapsed:.4f}s > {self.dt:.4f}s")
        
        except Exception as e:
            print(f"[ROS2InterpolationController] Error in control loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up ROS2 resources
            if hasattr(self, 'node') and self.ros_initialized:
                self.node.destroy_node()
