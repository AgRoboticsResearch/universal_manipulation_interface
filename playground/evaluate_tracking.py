import numpy as np
import matplotlib.pyplot as plt

# Load data with timestamps as first column
def load_data_with_timestamps(filename):
    data = np.loadtxt(filename, delimiter=",")
    timestamps = data[:, 0]
    values = data[:, 1:]
    return timestamps, values

def align_time_zero(timestamps, values):
    t0 = timestamps[0]
    return timestamps - t0, values

def compute_errors(target_timestamps, target_poses, robot_timestamps, robot_poses):
    # Interpolate robot poses to target timestamps for fair comparison
    interp_robot_poses = np.empty_like(target_poses)
    for i in range(target_poses.shape[1]):
        interp_robot_poses[:, i] = np.interp(target_timestamps, robot_timestamps, robot_poses[:, i])
    errors = interp_robot_poses - target_poses
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mean_error = np.mean(abs_errors, axis=0)
    return errors, mean_error, rmse, interp_robot_poses

def plot_comparison(target_timestamps, target_poses, robot_timestamps, robot_poses, label_prefix="Pose"):
    n_dim = target_poses.shape[1]
    plt.figure(figsize=(15, 2 * n_dim))
    for i in range(n_dim):
        plt.subplot(n_dim, 1, i + 1)
        plt.plot(target_timestamps, target_poses[:, i], label=f"Target {label_prefix} {i}")
        plt.plot(robot_timestamps, robot_poses[:, i], label=f"Robot {label_prefix} {i}", alpha=0.7)
        plt.ylabel(f"Dim {i}")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory(target_poses, robot_poses, title="3D Trajectory Comparison"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(target_poses[:, 0], target_poses[:, 1], target_poses[:, 2], 'b-', label='Target Trajectory', alpha=0.7)
    ax.plot(robot_poses[:, 0], robot_poses[:, 1], robot_poses[:, 2], 'r-', label='Robot Trajectory', alpha=0.7)
    ax.scatter(target_poses[0, 0], target_poses[0, 1], target_poses[0, 2], color='blue', s=60, label='Target Start')
    ax.scatter(robot_poses[0, 0], robot_poses[0, 1], robot_poses[0, 2], color='red', s=60, label='Robot Start')
    ax.scatter(target_poses[-1, 0], target_poses[-1, 1], target_poses[-1, 2], color='blue', s=60, marker='x', label='Target End')
    ax.scatter(robot_poses[-1, 0], robot_poses[-1, 1], robot_poses[-1, 2], color='red', s=60, marker='x', label='Robot End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Load all files
    target_ts, target_poses = load_data_with_timestamps("./temp/temp_target_poses.txt")
    robot_pose_ts, robot_poses = load_data_with_timestamps("./temp/temp_robot_poses.txt")
    robot_joint_ts, robot_joints = load_data_with_timestamps("./temp/temp_robot_states.txt")

    # Align all to the first target pose timestamp
    t0 = target_ts[0]
    target_ts, target_poses = align_time_zero(target_ts, target_poses)
    robot_pose_ts, robot_poses = align_time_zero(robot_pose_ts, robot_poses)
    robot_joint_ts, robot_joints = align_time_zero(robot_joint_ts, robot_joints)

    # Compute and print pose errors
    pose_errors, pose_mean_error, pose_rmse, interp_robot_poses = compute_errors(
        target_ts, target_poses, robot_pose_ts, robot_poses)
    print("End-Effector Pose Error (Target vs Robot):")
    print("Mean Error per dim:", pose_mean_error)
    print("RMSE per dim:", pose_rmse)
    print("Overall Mean Error:", np.mean(pose_mean_error))
    print("Overall RMSE:", np.mean(pose_rmse))

    # Plot pose comparison (time-dim)
    plot_comparison(target_ts, target_poses, robot_pose_ts, robot_poses, label_prefix="Pose")

    # Plot 3D trajectory comparison
    plot_3d_trajectory(target_poses, interp_robot_poses, title="3D Trajectory Comparison (Target vs Robot)")

    # Compute and print joint errors (if target joints available, otherwise just plot)
    # Here, we only plot robot joints over time
    plt.figure(figsize=(15, 2 * robot_joints.shape[1]))
    for i in range(robot_joints.shape[1]):
        plt.subplot(robot_joints.shape[1], 1, i + 1)
        plt.plot(robot_joint_ts, robot_joints[:, i], label=f"Joint {i}")
        plt.ylabel(f"Joint {i}")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
