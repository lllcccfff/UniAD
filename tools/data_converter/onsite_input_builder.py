import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR


LIDAR_TO_EGO = np.array([
    [0.0, 1.0, 0.0, 0.5],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.5],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)
DEFAULT_IMG_NORM_CFG = {
    "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "to_rgb": False,
}


def _turn_signal_to_uniad_command(raw_command):
    # UniAD planning command ids: 0=RIGHT, 1=LEFT, 2=FORWARD.
    if raw_command < 0:
        return 0
    if raw_command > 0:
        return 1
    return 2


def build_uniad_input(frames, vehicle_feedback, device, scene_token):
    vehicle_pose = np.asarray(vehicle_feedback.vehicle_pose, dtype=np.float32).reshape(4, 4)
    ego_rot = SCR.from_matrix(vehicle_pose[:3, :3]).as_euler("XYZ").astype(np.float32)
    ego_pos = vehicle_pose[:3, 3].astype(np.float32)
    vehicle_velocity = np.asarray(vehicle_feedback.bcm_feedback.vehicle_velocity, dtype=np.float32).reshape(3)
    acceleration = np.asarray(vehicle_feedback.bcm_feedback.acceleration, dtype=np.float32).reshape(3)
    angular_velocity = np.asarray(vehicle_feedback.bcm_feedback.angular_velocity, dtype=np.float32).reshape(3)

    ego_to_global = vehicle_pose[:3, :3]
    ego_position = vehicle_pose[:3, 3]
    l2g_r_mat = ego_to_global @ LIDAR_TO_EGO[:3, :3]
    l2g_t = ego_position + ego_to_global @ LIDAR_TO_EGO[:3, 3]

    imgs_shape = []
    imgs = []
    lidar2img = []

    for frame in frames:
        raw = frame["rgb"][..., ::-1].astype(np.float32)
        if "camera_pose_in_ego" in frame:
            camera_pose_in_ego = np.asarray(frame["camera_pose_in_ego"], dtype=np.float32).reshape(4, 4)
        else:
            camera_pose_in_ego = np.asarray(frame["camera_pose"], dtype=np.float32).reshape(4, 4)
        intrinsic3 = np.asarray(frame["intrinsic"], dtype=np.float32).reshape(3, 3)

        imgs_shape.append(list(raw.shape))
        imgs.append(torch.from_numpy((raw - DEFAULT_IMG_NORM_CFG["mean"]) / DEFAULT_IMG_NORM_CFG["std"]).permute(2, 0, 1))

        ego2camera = np.linalg.inv(camera_pose_in_ego)
        intrinsic4 = np.eye(4, dtype=np.float32)
        intrinsic4[:3, :3] = intrinsic3
        lidar2img.append(intrinsic4 @ ego2camera @ LIDAR_TO_EGO)

    timestamp = float(vehicle_feedback.header.sim_ts) / 1e3

    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[:3] = ego_pos
    can_bus[3:7] = SCR.from_euler("XYZ", ego_rot).as_quat()[[3, 0, 1, 2]]
    can_bus[7:10] = acceleration
    can_bus[10:13] = angular_velocity
    can_bus[13:16] = vehicle_velocity
    yaw = float(ego_rot[2])
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180

    raw_command = int(vehicle_feedback.command)
    command = _turn_signal_to_uniad_command(raw_command)

    return {
        "img": [torch.stack(imgs, dim=0)[None, ...].to(device)],
        "img_metas": [[{
            "scene_token": str(scene_token),
            "can_bus": can_bus,
            "img_shape": imgs_shape,
            "lidar2img": lidar2img,
            "l2g_r_mat": l2g_r_mat,
        }]],
        "l2g_t": torch.tensor(l2g_t, dtype=torch.float32, device=device)[None, :],
        "l2g_r_mat": torch.tensor(l2g_r_mat, dtype=torch.float32, device=device)[None, :, :],
        "timestamp": [torch.tensor([timestamp], dtype=torch.float64, device=device)],
        "command": [torch.tensor([command], dtype=torch.long, device=device)],
    }, float(vehicle_feedback.bcm_feedback.vehicle_speed), float(vehicle_feedback.steering_feedback.steering_wheel_angle)
