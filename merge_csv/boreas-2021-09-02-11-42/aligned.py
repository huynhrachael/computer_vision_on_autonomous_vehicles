import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def load_pose_csv(filepath):
    """
    Reads a Boreas pose CSV.
    Expected columns: timestamp, x, y, z, roll, pitch, yaw (or quaternion)
    """
    df = pd.read_csv(filepath)
    timestamps = df['timestamp'].values
    positions = df[['x', 'y', 'z']].values
    
    # Convert Euler angles (roll, pitch, yaw) to quaternions for SLERP
    eulers = df[['roll', 'pitch', 'yaw']].values
    quats = R.from_euler('xyz', eulers, degrees=False).as_quat()
    
    return timestamps, positions, quats

def interpolate_pose(target_time, source_times, positions, quats):
    """
    Interpolates position (Linear) and rotation (SLERP) at target_time.
    """
    # Find bounds for interpolation
    idx = np.searchsorted(source_times, target_time) - 1
    idx = max(0, min(idx, len(source_times) - 2))
    
    t1, t2 = source_times[idx], source_times[idx + 1]
    
    # Handle bounds/extrapolation safely
    if target_time <= source_times[0]:
        return positions[0], quats[0]
    if target_time >= source_times[-1]:
        return positions[-1], quats[-1]
        
    alpha = (target_time - t1) / (t2 - t1)
    
    # Linear interpolation for 3D position
    interp_pos = (1 - alpha) * positions[idx] + alpha * positions[idx + 1]
    
    # SLERP for orientation quaternion
    slerp = Slerp([t1, t2], R.from_quat(quats[idx:idx + 2]))
    interp_quat = slerp(target_time).as_quat()
    
    return interp_pos, interp_quat

def synchronize_boreas_poses(lidar_path, camera_path, radar_path, output_path):
    t_lid, pos_lid, quat_lid = load_pose_csv(lidar_path)
    t_cam, pos_cam, quat_cam = load_pose_csv(camera_path)
    t_rad, pos_rad, quat_rad = load_pose_csv(radar_path)
    
    synchronized_rows = []
    
    for i, t_ref in enumerate(t_lid):
        # Lidar pose (Reference)
        p_l, q_l = pos_lid[i], quat_lid[i]
        
        # Camera pose interpolated at lidar timestamp
        p_c, q_c = interpolate_pose(t_ref, t_cam, pos_cam, quat_cam)
        
        # Radar pose interpolated at lidar timestamp
        p_r, q_r = interpolate_pose(t_ref, t_rad, pos_rad, quat_rad)
        
        # Flatten into one synchronized record
        row = {
            'timestamp': t_ref,
            'lidar_frame_idx': i,
            # Lidar Pose
            'lidar_x': p_l[0], 'lidar_y': p_l[1], 'lidar_z': p_l[2],
            'lidar_qx': q_l[0], 'lidar_qy': q_l[1], 'lidar_qz': q_l[2], 'lidar_qw': q_l[3],
            # Camera Pose (Matched)
            'cam_x': p_c[0], 'cam_y': p_c[1], 'cam_z': p_c[2],
            'cam_qx': q_c[0], 'cam_qy': q_c[1], 'cam_qz': q_c[2], 'cam_qw': q_c[3],
            # Radar Pose (Matched)
            'radar_x': p_r[0], 'radar_y': p_r[1], 'radar_z': p_r[2],
            'radar_qx': q_r[0], 'radar_qy': q_r[1], 'radar_qz': q_r[2], 'radar_qw': q_r[3]
        }
        synchronized_rows.append(row)
        
    df_sync = pd.DataFrame(synchronized_rows)
    df_sync.to_csv(output_path, index=False)
    print(f"Successfully synchronized {len(df_sync)} frames into {output_path}")

# Run Synchronization
synchronize_boreas_poses("lidar_poses.csv", "camera_poses.csv", "radar_poses.csv", "aligned_multimodal_poses.csv")