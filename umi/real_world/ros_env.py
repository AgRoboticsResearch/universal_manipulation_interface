from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
import rospy
# Add scipy import
import scipy.spatial.transform as st
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multiprocessing.managers import SharedMemoryManager
import threading
# Add imports for visualization
from visualization_msgs.msg import Marker, MarkerArray # Import MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA
from rospy import Duration
from umi.real_world.ros_interpolation_controller import ROSInterpolationController
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import (
    draw_predefined_mask, 
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, 
    optimal_row_cols
)
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.real_world.video_recorder import VideoRecorder


class RosEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip=None,  # Not used in ROS controller, but kept for API compatibility
            gripper_ip=None, # Not used in ROS controller, but kept for API compatibility
            gripper_port=None, # Not used in ROS controller, but kept for API compatibility
            # env params
            frequency=20,
            # ros params
            joint_names=None,
            traj_action_name='/z1_joint_traj_controller/follow_joint_trajectory',
            group_name="manipulator",
            eef_link="link06",
            reference_frame="link00",
            camera_topic="/camera_ee_cam",
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            no_mirror=False,
            fisheye_converter=None,
            mirror_crop=False,
            mirror_swap=False,
            # timing
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            robot_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=None,
            init_joints=False,
            joints_init=None,
            # video recording
            camera_fps=30,
            video_bit_rate=3000*1000,
            # shared memory
            shm_manager=None,
            require_camera=True
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('ros_env', anonymous=True, disable_signals=True)
            
        self.require_camera = require_camera

        # Add target pose publisher for RViz visualization (using MarkerArray and latch=True)
        self.target_pose_pub = rospy.Publisher('/rviz/target_pose', MarkerArray, queue_size=1, latch=True)
        self._marker_id_counter = 0 # To give unique IDs to markers

        if self.require_camera:
            # Setup image subscriber
            self.bridge = CvBridge()
            self.last_camera_data = None
            self.camera_buffer = {
                'color': [],
                'timestamp': []
            }
            self.camera_buffer_lock = threading.Lock()
            self.camera_sub = rospy.Subscriber(
                camera_topic, 
                Image, 
                self.camera_callback,
                queue_size=1
            )
            rospy.loginfo(f"Subscribing to camera topic: {camera_topic}")
        else:
            self.bridge = None
            self.last_camera_data = None
            self.camera_buffer = {'color': [], 'timestamp': []}
            self.camera_buffer_lock = threading.Lock()
            self.camera_sub = None
            self._ready = True
        
        # Setup video recorder
        self.video_recorder = VideoRecorder.create_hevc_nvenc(
            fps=camera_fps,
            input_pix_fmt='rgb24',
            bit_rate=video_bit_rate
        )
        self.video_recording = False
        
        # Setup robot controller
        tcp_offset_pose = None
        if tcp_offset is not None:
            tcp_offset_pose = [0, 0, tcp_offset, 0, 0, 0]
            
        self.robot = ROSInterpolationController(
            shm_manager=shm_manager,
            joint_names=joint_names,
            traj_action_name=traj_action_name,
            frequency=frequency,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            tcp_offset_pose=tcp_offset_pose,
            joints_init=joints_init if init_joints else None,
            joints_init_speed=1.05,
            verbose=False,
            receive_latency=robot_obs_latency,
            group_name=group_name,
            eef_link=eef_link,
            reference_frame=reference_frame,
            debug=True
        )

        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        self.obs_image_resolution = obs_image_resolution
        self.obs_float32 = obs_float32
        self.no_mirror = no_mirror
        self.fisheye_converter = fisheye_converter
        self.shm_manager = shm_manager
        self.camera_fps = camera_fps
        
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.robot_action_latency = robot_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self._ready = False
    
    def camera_callback(self, msg):
        """Callback for camera image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process image
            timestamp = msg.header.stamp.to_sec()
            t_recv = time.time()
            
            # Apply image transformations similar to UmiEnv
            img = cv_image
            if self.fisheye_converter is None:
                crop_img = None
                if self.mirror_crop:
                    slices = get_mirror_crop_slices(img.shape[:2], left=False)
                    crop = img[slices]
                    crop_img = cv2.resize(crop, self.obs_image_resolution)
                    crop_img = crop_img[:,::-1,::-1]  # bgr to rgb
                
                f = get_image_transform(
                    input_res=img.shape[:2],
                    output_res=self.obs_image_resolution, 
                    bgr_to_rgb=True
                )
                img = np.ascontiguousarray(f(img))
                img = draw_predefined_mask(img, color=(0,0,0), 
                    mirror=self.no_mirror, gripper=True, finger=False, use_aa=True)
                if crop_img is not None:
                    img = np.concatenate([img, crop_img], axis=-1)
            else:
                img = self.fisheye_converter.forward(img)
                img = img[...,::-1]  # bgr to rgb
                
            if self.obs_float32:
                img = img.astype(np.float32) / 255
            
            # Create a copy for the video recorder (RGB format, uint8)
            if self.video_recording and self.video_recorder.is_ready():
                # Convert back to uint8 if needed
                video_img = img if not self.obs_float32 else (img * 255).astype(np.uint8)
                self.video_recorder.write_frame(video_img, frame_time=t_recv)
                
            with self.camera_buffer_lock:
                self.camera_buffer['color'].append(img)
                self.camera_buffer['timestamp'].append(timestamp)
                
                # Limit buffer size
                max_buffer = self.max_obs_buffer_size
                if len(self.camera_buffer['color']) > max_buffer:
                    self.camera_buffer['color'] = self.camera_buffer['color'][-max_buffer:]
                    self.camera_buffer['timestamp'] = self.camera_buffer['timestamp'][-max_buffer:]
                
        except Exception as e:
            rospy.logerr(f"Error processing camera image: {e}")
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        if not self.require_camera:
            return self.robot.is_ready
        return self._ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.robot.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        self.robot.stop(wait=False)
        if self.video_recording:
            self.video_recorder.stop_recording()
            self.video_recording = False
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.robot.start_wait()
        if not self.require_camera:
            self._ready = True
            return
        # Wait for camera data to be available
        timeout = 5.0  # 5 seconds timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.camera_buffer_lock:
                if len(self.camera_buffer['color']) > 0:
                    self._ready = True
                    break
            time.sleep(0.1)
        
        if not self._ready:
            rospy.logwarn("Camera data not available after timeout")
    
    def stop_wait(self):
        self.robot.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Get observations from the environment.
        Timestamp alignment policy:
        'current' time is the last timestamp of camera
        All low-dim observations, interpolate with respect to 'current' time
        """
        if self.require_camera:
            assert self.is_ready

        if not self.require_camera:
            # Only return robot state, no camera
            last_robot_data = self.robot.get_all_state()
            dt = 1 / self.frequency
            robot_obs_timestamps = time.time() - (
                np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
            robot_pose_interpolator = PoseInterpolator(
                t=np.array(last_robot_data['robot_timestamp']), 
                x=np.array(last_robot_data['ActualTCPPose']))
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                'robot0_eef_pos': robot_pose[...,:3],
                'robot0_eef_rot_axis_angle': robot_pose[...,3:],
                'timestamp': robot_obs_timestamps
            }
            return robot_obs

        # get data from ROS camera topic
        with self.camera_buffer_lock:
            if len(self.camera_buffer['color']) == 0:
                rospy.logwarn("No camera data available")
                return None
            
            camera_data = {
                'color': np.array(self.camera_buffer['color']),
                'timestamp': np.array(self.camera_buffer['timestamp'])
            }
            
        # get robot data
        last_robot_data = self.robot.get_all_state()
        
        # Get latest timestamp from camera
        last_timestamp = camera_data['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        
        # Find nearest camera frames to the desired timestamps
        this_timestamps = camera_data['timestamp']
        this_idxs = list()
        for t in camera_obs_timestamps:
            if len(this_timestamps) > 0:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            else:
                this_idxs.append(0)
        
        # Get the camera observations at the desired timestamps
        if len(this_idxs) > 0:
            if self.mirror_crop:
                camera_obs['camera0_rgb'] = camera_data['color'][this_idxs][...,:3]
                camera_obs['camera0_rgb_mirror_crop'] = camera_data['color'][this_idxs][...,3:]
            else:
                camera_obs['camera0_rgb'] = camera_data['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        
        # Create pose interpolator
        robot_pose_interpolator = PoseInterpolator(
            t=np.array(last_robot_data['robot_timestamp']), 
            x=np.array(last_robot_data['ActualTCPPose']))
        
        # Interpolate robot pose to align with camera timestamps
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # Accumulate obs for recording if needed
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        if self.require_camera:
            assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i,:6]  # robot pose
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    def publish_target_pose(self, pose_array):
        """Publishes the target pose as an RViz marker within a MarkerArray."""
        marker_array = MarkerArray()
        
        marker = Marker()
        marker.header.frame_id = self.robot.reference_frame # Use the robot's reference frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_pose"
        marker.id = 0 # Use a fixed ID for the single marker in the array
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set the pose of the marker
        marker.pose.position.x = pose_array[0]
        marker.pose.position.y = pose_array[1]
        marker.pose.position.z = pose_array[2]
        
        # Convert axis-angle rotation to quaternion
        rotation = st.Rotation.from_rotvec(pose_array[3:6])
        quat = rotation.as_quat() # Returns (x, y, z, w)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # Set the scale of the marker (arrow dimensions)
        marker.scale.x = 0.1  # Arrow length
        marker.scale.y = 0.02 # Arrow width
        marker.scale.z = 0.02 # Arrow height

        # Set the color of the marker (RGBA)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8 # Alpha (transparency)

        # Set the lifetime of the marker (Duration(0) means infinite if latched)
        marker.lifetime = Duration(0) # Keep marker indefinitely since latch=True

        marker_array.markers.append(marker)
        self.target_pose_pub.publish(marker_array)

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        if self.require_camera:
            assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video recorder if first image exists
        with self.camera_buffer_lock:
            if len(self.camera_buffer['color']) > 0:
                example_image = self.camera_buffer['color'][-1]
                # If image is float32, convert to uint8 for video recording
                if self.obs_float32:
                    example_image = (example_image * 255).astype(np.uint8)
                
                # Start video recorder
                self.video_recorder.start(
                    shm_manager=self.shm_manager, 
                    data_example=example_image
                )
                video_path = str(this_video_dir.joinpath('0.mp4').absolute())
                self.video_recorder.start_recording(video_path=video_path, start_time=start_time)
                self.video_recording = True
                rospy.loginfo(f"Started video recording to {video_path}")
        
        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        if self.require_camera:
            assert self.is_ready
        
        # Stop video recording
        if self.video_recording:
            self.video_recorder.stop_recording()
            self.video_recording = False
            rospy.loginfo("Video recording stopped")
        
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')