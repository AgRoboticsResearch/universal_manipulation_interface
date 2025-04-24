from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:
    # Class variable for debug mode - set to False by default
    debug = False
    
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:,:3]
            rot = st.Rotation.from_rotvec(poses[:,3:])

            self.pos_interp = si.interp1d(times, pos, 
                axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # Debug: Print initial info
        initial_length = len(self.times)
        
        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # Debug: Print when waypoint is skipped due to being in the past
                if self.debug:
                    print(f"DEBUG: Skipping waypoint - target_time {time:.3f} <= curr_time {curr_time:.3f}")
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    # Debug: Print when time <= last_waypoint_time
                    if self.debug:
                        print(f"DEBUG: time {time:.3f} <= last_waypoint_time {last_waypoint_time:.3f}, setting end_time to curr_time {curr_time:.3f}")
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
                    # Debug: Print when updating end_time
                    if self.debug:
                        print(f"DEBUG: Updated end_time to max(last_waypoint_time {last_waypoint_time:.3f}, curr_time {curr_time:.3f}) = {end_time:.3f}")
            else:
                end_time = curr_time
                # Debug: Print when setting end_time to curr_time
                if self.debug:
                    print(f"DEBUG: Setting end_time to curr_time {curr_time:.3f}")

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # Debug: Print final start_time and end_time
        if self.debug:
            print(f"DEBUG: Final start_time={start_time:.3f}, end_time={end_time:.3f}, time={time:.3f}")

        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # Debug: Print how many points were trimmed
        if self.debug:
            print(f"DEBUG: Trimmed interpolator from {initial_length} to {len(trimmed_interp.times)} points")
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration
        # Debug: Print duration and distances
        if self.debug:
            print(f"DEBUG: pos_dist={pos_dist:.6f}, rot_dist={rot_dist:.6f}, pos_min_duration={pos_min_duration:.6f}, rot_min_duration={rot_min_duration:.6f}")
            print(f"DEBUG: Final duration={duration:.6f}, last_waypoint_time={last_waypoint_time:.3f}")

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        # Debug: Print final size
        if self.debug:
            print(f"DEBUG: Final interpolator size: {len(final_interp.times)} points")
        
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:,:3] = self.pos_interp(t)
            pose[:,3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
