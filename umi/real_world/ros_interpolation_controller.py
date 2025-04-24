import rospy
import actionlib
import threading
import time
import enum
import numpy as np
import tf2_ros
from queue import Queue, Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class ROSInterpolationController:
    def __init__(self, 
                 shm_manager=None,  # Added for compatibility with RTDEInterpolationController
                 robot_ip=None,     # Added for compatibility, not used in ROS controller
                 joint_names=None, 
                 traj_action_name='/z1_joint_traj_controller/follow_joint_trajectory',
                 frequency=125,
                 max_pos_speed=0.25,
                 max_rot_speed=0.16,
                 launch_timeout=3,
                 tcp_offset_pose=None,
                 payload_mass=None,
                 payload_cog=None,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 debug=False,
                 receive_keys=None,
                 get_max_k=None,
                 receive_latency=0.0,
                 group_name="manipulator",
                 eef_link="link06",
                 reference_frame="link00"):
        """
        ROS Interpolation Controller with interface compatible with RTDEInterpolationController
        
        Parameters:
        -----------
        shm_manager: SharedMemoryManager
            Not used in ROS controller, added for API compatibility
        robot_ip: str
            Not used in ROS controller, added for API compatibility
        joint_names: list
            List of joint names to control
        traj_action_name: str
            Name of the ROS action server for trajectory control
        frequency: float
            Control frequency in Hz
        max_pos_speed: float
            Maximum position speed in m/s
        max_rot_speed: float
            Maximum rotation speed in rad/s
        launch_timeout: float
            Timeout for waiting for action server
        tcp_offset_pose: list/array
            Tool center point offset
        payload_mass: float
            Mass of payload (not directly used in ROS controller)
        payload_cog: list/array
            Center of gravity of payload (not directly used in ROS controller)
        joints_init: list/array
            Initial joint positions (if provided, moves to this position on init)
        joints_init_speed: float
            Speed for initial joint movement
        soft_real_time: bool
            Enable soft real-time scheduling (not fully implemented in ROS controller)
        verbose: bool
            Enable verbose logging
        debug: bool
            Enable debug print statements
        receive_keys: list
            Not used in ROS controller, added for API compatibility
        get_max_k: int
            Max number of states to buffer
        receive_latency: float
            Latency compensation for timestamps
        group_name: str
            MoveIt group name for IK/FK calculations
        eef_link: str
            End effector link name for IK/FK calculations
        reference_frame: str
            Reference frame for IK/FK calculations (default: 'link00')
        """
        if joint_names is None:
            # Default joint names if none provided
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            
        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('ros_interpolation_controller', anonymous=True, disable_signals=True)
            
        self.joint_names = joint_names
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.debug = debug
        self.receive_latency = receive_latency
        self.group_name = group_name
        self.eef_link = eef_link
        self.reference_frame = reference_frame
        
        # Initialize the maximum buffer size for state history
        self.max_buffer_size = 500
        if get_max_k is not None:
            self.max_buffer_size = get_max_k
            
        # Set up action client for trajectory control
        self.trajectory_client = actionlib.SimpleActionClient(
            traj_action_name,
            FollowJointTrajectoryAction
        )
        
        # Subscribe to joint state feedback
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.current_joint_positions = [0.0] * len(joint_names)
        self.joint_state_lock = threading.Lock()
        
        # Buffer for storing historical state data (like RTDEInterpolationController's ring buffer)
        self.state_buffer = {
            'ActualTCPPose': [],  # End effector pose [x, y, z, rx, ry, rz]
            'ActualQ': [],        # Joint positions
            'ActualQd': [],       # Joint velocities
            'robot_timestamp': [], # Timestamps
            'robot_receive_timestamp': [] # Raw timestamps when data was received
        }
        self.state_buffer_lock = threading.Lock()
        
        # Command queue and worker thread
        self.command_queue = Queue()
        self.running = False
        self.worker_thread = None
        self.ready_event = threading.Event()
        
        # Current pose tracking
        self.current_pose = None
        self.pose_lock = threading.Lock()

        # Initialize IK/FK services
        self._initialize_kinematics()
        
        # TF listener for potential transformations
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _initialize_kinematics(self):
        """Initialize the IK and FK service connections"""
        # Set up IK service
        ik_service_name = '/compute_ik'
        try:
            rospy.loginfo(f"Waiting for IK service: {ik_service_name}")
            rospy.wait_for_service(ik_service_name, timeout=self.launch_timeout)
            self.compute_ik_service = rospy.ServiceProxy(ik_service_name, GetPositionIK)
            rospy.loginfo(f"Connected to IK service: {ik_service_name}")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"IK service not available: {e}. Using null kinematics.")
            self.compute_ik_service = None
            
        # Set up FK service
        fk_service_name = '/compute_fk'
        try:
            rospy.loginfo(f"Waiting for FK service: {fk_service_name}")
            rospy.wait_for_service(fk_service_name, timeout=self.launch_timeout)
            self.compute_fk_service = rospy.ServiceProxy(fk_service_name, GetPositionFK)
            rospy.loginfo(f"Connected to FK service: {fk_service_name}")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"FK service not available: {e}. Using null kinematics.")
            self.compute_fk_service = None

    def compute_ik(self, target_pose):
        """
        Compute inverse kinematics for a target pose
        
        Parameters:
        -----------
        target_pose: geometry_msgs.msg.Pose
            Target pose for the end effector
            
        Returns:
        --------
        success: bool
            Whether IK calculation was successful
        joint_positions: list
            Joint positions to achieve the target pose
        time_taken: float
            Time taken for the computation
        """
        if self.compute_ik_service is None:
            rospy.logwarn("IK service not available. Cannot compute IK.")
            return False, None, 0.0
            
        tic = time.time()
        
        # Prepare the service request
        ik_request = GetPositionIKRequest()
        ik_request.ik_request.group_name = self.group_name
        ik_request.ik_request.pose_stamped.header.frame_id = self.reference_frame
        ik_request.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        ik_request.ik_request.pose_stamped.pose = target_pose
        
        # Set end effector link if available
        if self.eef_link:
            ik_request.ik_request.ik_link_name = self.eef_link
            
        # Current robot state - we'll use current joint positions
        robot_state = self._get_current_robot_state()
        ik_request.ik_request.robot_state = robot_state
        
        # Set timeout
        ik_request.ik_request.timeout.secs = 0
        ik_request.ik_request.timeout.nsecs = int(0.5 * 1000000000)  # 0.5 seconds
        
        success = False
        joint_positions = None
        try:
            # Call the service
            response = self.compute_ik_service(ik_request)
            
            # Check if the response is valid
            if response.error_code.val == response.error_code.SUCCESS:
                success = True
                joint_positions = response.solution.joint_state.position
                if self.verbose:
                    rospy.loginfo(f"IK computed successfully: {joint_positions}")
            else:
                rospy.logwarn(f"IK computation failed with error code: {response.error_code.val}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            
        computation_time = time.time() - tic
        return success, joint_positions, computation_time

    def compute_fk(self, joint_positions):
        """
        Compute forward kinematics for a set of joint positions
        
        Parameters:
        -----------
        joint_positions: list
            Joint positions to compute the end effector pose for
            
        Returns:
        --------
        success: bool
            Whether FK calculation was successful
        pose: geometry_msgs.msg.Pose
            Pose of the end effector
        time_taken: float
            Time taken for the computation
        """
        if self.compute_fk_service is None:
            rospy.logwarn("FK service not available. Cannot compute FK.")
            return False, None, 0.0
            
        tic = time.time()
        
        # Prepare the service request
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = self.reference_frame
        fk_request.header.stamp = rospy.Time.now()
        
        # Set which links to compute FK for
        if self.eef_link:
            fk_request.fk_link_names = [self.eef_link]
        else:
            raise ValueError("End effector link name is required for FK computation")
            
        # Set the robot state
        from sensor_msgs.msg import JointState
        from moveit_msgs.msg import RobotState
        
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = joint_positions
        
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        fk_request.robot_state = robot_state
        
        success = False
        pose = None
        try:
            # Call the service
            response = self.compute_fk_service(fk_request)
            
            # Check if the response is valid
            if response.error_code.val == response.error_code.SUCCESS:
                success = True
                pose = response.pose_stamped[0].pose
                if self.verbose:
                    rospy.loginfo(f"FK computed successfully: {pose}")
            else:
                rospy.logwarn(f"FK computation failed with error code: {response.error_code.val}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            
        computation_time = time.time() - tic
        return success, pose, computation_time

    def _get_current_robot_state(self):
        """
        Get the current robot state from joint positions
        
        Returns:
        --------
        robot_state: moveit_msgs.msg.RobotState
            Current robot state
        """
        from moveit_msgs.msg import RobotState
        from sensor_msgs.msg import JointState
        
        with self.joint_state_lock:
            # Create a joint state message
            joint_state = JointState()
            joint_state.name = self.joint_names
            joint_state.position = self.current_joint_positions
            
            # Create the robot state
            robot_state = RobotState()
            robot_state.joint_state = joint_state
            
        return robot_state

    def joint_states_callback(self, msg):
        """Callback for joint state messages"""
        with self.joint_state_lock:
            # Map received joint states to self.joint_names order
            name_to_pos = dict(zip(msg.name, msg.position))
            name_to_vel = dict(zip(msg.name, msg.velocity))
            
            # Update current joint positions
            self.current_joint_positions = [name_to_pos.get(j, 0.0) for j in self.joint_names]
            
            # Update current joint velocities (if available)
            current_joint_velocities = [name_to_vel.get(j, 0.0) for j in self.joint_names]
            
            # Store in the state buffer with timestamps
            t_recv = time.time()
            t_stamp = t_recv - self.receive_latency
            
            with self.state_buffer_lock:
                # Add to state buffer
                self.state_buffer['ActualQ'].append(np.array(self.current_joint_positions))
                self.state_buffer['ActualQd'].append(np.array(current_joint_velocities))
                self.state_buffer['robot_timestamp'].append(t_stamp)
                self.state_buffer['robot_receive_timestamp'].append(t_recv)
                # print(f"Joint states updated: {self.current_joint_positions}")
                
                # Compute TCP pose using forward kinematics
                success, pose, _ = self.compute_fk(self.current_joint_positions)
                
                if success and pose is not None:
                    # Convert the pose to Euler angles and save it
                    position = np.array([pose.position.x, pose.position.y, pose.position.z])
                    orientation = np.array([pose.orientation.x, pose.orientation.y, 
                                           pose.orientation.z, pose.orientation.w])
                    
                    # Convert quaternion to axis-angle representation
                    rotation = Rotation.from_quat(orientation)
                    euler_angles = rotation.as_euler('xyz', degrees=False)
                    
                    # Store the pose as [x, y, z, rx, ry, rz]
                    tcp_pose = np.concatenate([position, euler_angles])
                    self.state_buffer['ActualTCPPose'].append(tcp_pose)
                    
                    # Update the current pose reference
                    with self.pose_lock:
                        self.current_pose = tcp_pose
                else:
                    # If FK failed, use the previously stored pose or fallback to zeros
                    if len(self.state_buffer['ActualTCPPose']) > 0:
                        self.state_buffer['ActualTCPPose'].append(self.state_buffer['ActualTCPPose'][-1])
                    else:
                        self.state_buffer['ActualTCPPose'].append(np.zeros(6))
                
                # Limit buffer size
                if len(self.state_buffer['ActualQ']) > self.max_buffer_size:
                    for key in self.state_buffer:
                        self.state_buffer[key] = self.state_buffer[key][-self.max_buffer_size:]

    def schedule_waypoint(self, pose, target_time):
        """Schedule a waypoint to be reached at a specific time"""
        pose = np.array(pose)
        assert pose.shape == (6,), f"Pose must be a 6D array, got {pose.shape}"
        
        # Update current pose reference
        with self.pose_lock:
            self.current_pose = pose
            
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.command_queue.put(message)

    def servoL(self, pose, duration=0.1):
        """
        Move in a linear path to the target pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        # Update current pose reference
        with self.pose_lock:
            self.current_pose = pose
            
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.command_queue.put(message)
        
    def getActualTCPPose(self):
        """
        Get the current TCP pose of the robot
        
        Returns:
        --------
        tcp_pose: numpy.ndarray
            Current TCP pose as [x, y, z, rx, ry, rz]
        """
        while self.current_pose is None:
            with self.pose_lock:
                current_pose = self.current_pose
            time.sleep(0.01)  # Wait for the pose to be updated
        return self.current_pose
            
    def run(self):
        """Main control loop"""
        # Handle initial joint positions if provided
        if self.joints_init is not None:
            rospy.loginfo(f"Moving to initial joint positions: {self.joints_init}")
                
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            point.positions = self.joints_init
            point.time_from_start = rospy.Duration(self.joints_init_speed)
            traj.points.append(point)
            
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = traj
            self.trajectory_client.send_goal(goal)
            self.trajectory_client.wait_for_result()
        
        dt = 1.0 / self.frequency
        t_start = time.monotonic()
        iter_idx = 0
        
        # Initialize pose interpolator
        curr_pose = self.getActualTCPPose()  # Use the actual TCP pose
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose]
        )
        
        # Debug: Print initial state
        if self.debug:
            print(f"DEBUG: Run loop started. dt={dt:.6f}s, frequency={self.frequency}Hz")
            print(f"DEBUG: Initial pose_interp has {len(pose_interp.times)} points")
        
        last_command_time = time.time()
        command_count = 0
        
        keep_running = True
        while keep_running and not rospy.is_shutdown():
            try:
                loop_start_time = time.monotonic()
                cycle_t_now = time.time()  # Wall clock time for debug prints
                
                # Process commands
                try:
                    # Process at most one command per cycle to maintain frequency
                    command = self.command_queue.get_nowait()
                    cmd = command['cmd']
                    command_count += 1
                    if self.debug:
                        print(f"DEBUG: Processing command #{command_count} of type {cmd} at {cycle_t_now:.6f}, {time.monotonic():.6f} (monotonic)")
                    
                    if cmd == Command.STOP.value:
                        if self.debug:
                            print("DEBUG: Received STOP command")
                        keep_running = False
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = time.monotonic() + dt
                        t_insert = curr_time + duration
                        if self.debug:
                            print(f"DEBUG: ServoL - curr_time={curr_time:.6f}, t_insert={t_insert:.6f}, duration={duration:.6f}")
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            rospy.loginfo(f"New pose target:{target_pose} duration:{duration}s")
                            
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        mono_target_time = time.monotonic() - time.time() + target_time
                        curr_time = time.monotonic() + dt
                        
                        # Debug timings
                        if self.debug:
                            print(f"DEBUG: schedule_waypoint - wall target_time={target_time:.6f}, current wall={time.time():.6f}, delta={target_time-time.time():.6f}s")
                            print(f"DEBUG: schedule_waypoint - mono_target_time={mono_target_time:.6f}, curr_time={curr_time:.6f}, delta={mono_target_time-curr_time:.6f}s")
                        
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=mono_target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        if self.debug:
                            print("pose_interp: ", len(pose_interp.poses))
                        last_waypoint_time = mono_target_time
                        last_command_time = time.time()
                        
                except Empty:
                    pass
                
                # Send trajectory commands
                t_now = time.monotonic()
                
                # Set debug flag for the PoseTrajectoryInterpolator class
                PoseTrajectoryInterpolator.debug = self.debug
                
                # Debug: Print timing info
                if self.verbose or iter_idx % 100 == 0:
                    time_since_last_cmd = time.time() - last_command_time
                    if self.debug:
                        print(f"DEBUG: Time since last command: {time_since_last_cmd:.3f}s, Queue size: {self.command_queue.qsize()}")
                
                # Get interpolated pose for current time
                cartesian_pose = pose_interp(t_now)
                if self.verbose:
                    rospy.loginfo(f"Interpolated pose: {cartesian_pose}")
                
                # Convert from [x,y,z,rx,ry,rz] to Pose message
                pose_msg = Pose()
                pose_msg.position.x = cartesian_pose[0]
                pose_msg.position.y = cartesian_pose[1]
                pose_msg.position.z = cartesian_pose[2]
                
                # Convert from Euler angles to quaternion
                rotation = Rotation.from_euler('xyz', cartesian_pose[3:6])
                quat = rotation.as_quat()
                pose_msg.orientation.x = quat[0]
                pose_msg.orientation.y = quat[1]
                pose_msg.orientation.z = quat[2]
                pose_msg.orientation.w = quat[3]
                
                # Compute IK for the pose
                success, joint_positions, computation_time = self.compute_ik(pose_msg)
                if self.verbose:
                    rospy.loginfo(f"IK computation time: {computation_time:.4f}s, success: {success}, joint_positions: {joint_positions}")
                
                if success and joint_positions is not None:
                    # Create and send trajectory for this control cycle
                    traj = JointTrajectory()
                    traj.joint_names = self.joint_names
                    
                    # Use the joint positions from IK
                    point = JointTrajectoryPoint()
                    point.positions = joint_positions
                    point.time_from_start = rospy.Duration(dt)
                    traj.points.append(point)
                    
                    # Send the trajectory
                    goal = FollowJointTrajectoryGoal()
                    goal.trajectory = traj
                    
                    # Debug: Print when sending IK
                    if self.verbose or iter_idx % 50 == 0:
                        if self.debug:
                            print(f"DEBUG: Sending IK solution at {time.time():.6f}: {joint_positions}")
                    
                    send_start = time.monotonic()
                    self.trajectory_client.send_goal_and_wait(goal, rospy.Duration(dt*0.5))
                    send_duration = time.monotonic() - send_start
                    
                    # Debug: Print send duration if it's taking too long
                    if send_duration > dt * 0.5:
                        if self.debug:
                            print(f"WARNING: Goal send_and_wait took {send_duration:.6f}s, which is {(send_duration/dt)*100:.1f}% of dt={dt:.6f}s")
                else:
                    if self.verbose or iter_idx % 10 == 0:
                        if self.debug:
                            print(f"DEBUG: IK failed for pose: {cartesian_pose}")
                
                # Calculate loop duration and sleep time
                loop_duration = time.monotonic() - loop_start_time
                if loop_duration > dt:
                    if self.debug:
                        print(f"WARNING: Loop took {loop_duration:.6f}s, exceeding dt={dt:.6f}s by {loop_duration-dt:.6f}s")
                
                # Regulate frequency 
                t_wait_until = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_until, time_func=time.monotonic)
                total_cycle_time = time.monotonic() - loop_start_time
                
                # First loop successful, ready to receive commands
                if iter_idx == 0:
                    self.ready_event.set()
                    
                iter_idx += 1
                
                if self.verbose and iter_idx % 100 == 0:  # Less frequent logging
                    actual_freq = 1/total_cycle_time if total_cycle_time > 0 else float('inf')
                    if self.debug:
                        print(f"DEBUG: Cycle {iter_idx}: Actual frequency {actual_freq:.2f}Hz (target: {self.frequency:.2f}Hz)")
                    
            except Exception as e:
                import traceback
                print(f"ERROR in control loop: {e}")
                print(traceback.format_exc())
                keep_running = False
                
        # Clean up
        self.ready_event.set()  # Ensure the ready event is set to avoid deadlocks
        
        if self.verbose:
            rospy.loginfo("ROS interpolation controller stopped")

    # State API
    def get_state(self, k=None, out=None):
        """Get the current state or the last k states"""
        with self.state_buffer_lock:
            if k is None:
                # Return the most recent state
                result = {
                    'ActualTCPPose': self.state_buffer['ActualTCPPose'][-1] if self.state_buffer['ActualTCPPose'] else np.zeros(6),
                    'ActualQ': self.state_buffer['ActualQ'][-1] if self.state_buffer['ActualQ'] else np.zeros(len(self.joint_names)),
                    'ActualQd': self.state_buffer['ActualQd'][-1] if self.state_buffer['ActualQd'] else np.zeros(len(self.joint_names)),
                    'robot_timestamp': self.state_buffer['robot_timestamp'][-1] if self.state_buffer['robot_timestamp'] else time.time(),
                    'robot_receive_timestamp': self.state_buffer['robot_receive_timestamp'][-1] if self.state_buffer['robot_receive_timestamp'] else time.time()
                }
            else:
                # Return the last k states
                k = min(k, len(self.state_buffer['ActualQ']))
                result = {
                    'ActualTCPPose': np.array(self.state_buffer['ActualTCPPose'][-k:]),
                    'ActualQ': np.array(self.state_buffer['ActualQ'][-k:]),
                    'ActualQd': np.array(self.state_buffer['ActualQd'][-k:]),
                    'robot_timestamp': np.array(self.state_buffer['robot_timestamp'][-k:]),
                    'robot_receive_timestamp': np.array(self.state_buffer['robot_receive_timestamp'][-k:])
                }
                
        # Handle the output parameter if provided
        if out is not None:
            for key in out:
                if key in result:
                    out[key][:] = result[key]
            return out
        else:
            return result

    def get_all_state(self):
        """Get all the states in the buffer"""
        with self.state_buffer_lock:
            return {
                'ActualTCPPose': np.array(self.state_buffer['ActualTCPPose']),
                'ActualQ': np.array(self.state_buffer['ActualQ']),
                'ActualQd': np.array(self.state_buffer['ActualQd']),
                'robot_timestamp': np.array(self.state_buffer['robot_timestamp']),
                'robot_receive_timestamp': np.array(self.state_buffer['robot_receive_timestamp'])
            }
            
    def get_current_joint_positions(self):
        """Get the current joint positions"""
        with self.joint_state_lock:
            return list(self.current_joint_positions)

    # Process management API
    def start(self, wait=True):
        """Start the controller"""
        if self.verbose:
            rospy.loginfo("Starting ROS interpolation controller")
            
        # Check if action server is available
        rospy.loginfo("Waiting for trajectory action server...")
        server_exists = self.trajectory_client.wait_for_server(timeout=rospy.Duration(self.launch_timeout))
        if not server_exists:
            rospy.logwarn("Trajectory action server not available after waiting. Commands may fail.")
        else:
            rospy.loginfo("Trajectory action server connected!")
            
        self.running = True
        self.worker_thread = threading.Thread(target=self.run)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        if wait:
            self.start_wait()
            
        if self.verbose:
            rospy.loginfo(f"ROS interpolation controller started")
            
    def stop(self, wait=True):
        """Stop the controller"""
        if self.running:
            self.running = False
            self.command_queue.put({'cmd': Command.STOP.value})
            if wait:
                self.stop_wait()
                
    def start_wait(self):
        """Wait for the controller to be ready"""
        self.ready_event.wait(timeout=self.launch_timeout)
        assert self.is_alive(), "Controller failed to start"
        
    def stop_wait(self):
        """Wait for the controller to stop"""
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=self.launch_timeout)
            
    def is_alive(self):
        """Check if the controller is alive"""
        return self.worker_thread is not None and self.worker_thread.is_alive()
        
    @property
    def is_ready(self):
        """Check if the controller is ready to receive commands"""
        return self.ready_event.is_set()
        
    # Context manager support
    def __enter__(self):
        """Enter context manager"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        self.stop()

# Example usage:
if __name__ == '__main__':
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    controller = ROSInterpolationController(
        joint_names=joint_names, 
        group_name="manipulator",
        eef_link="tool0",
        verbose=True
    )
    try:
        # Example: move to [0,0,0.5,0,0,0] in 2 seconds from now
        controller.schedule_waypoint([0,0,0.5,0,0,0], time.time() + 2.0)
        rospy.sleep(3.0)
    finally:
        controller.stop()