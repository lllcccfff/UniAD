import math

import numpy as np

from metadrive.utils.ilqr import plan2control


DEFAULT_WHEELBASE = 2.469
UNIAD_PLAN_DT = 0.5
EPS = 1e-5


def _wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _interpolate_plan(plan_traj: np.ndarray, control_dt: float) -> np.ndarray:
    total_horizon_s = len(plan_traj) * UNIAD_PLAN_DT
    target_len = round(total_horizon_s / control_dt)

    src_times = np.arange(0, len(plan_traj) + 1, dtype=np.float64) * UNIAD_PLAN_DT
    dst_times = np.arange(1, target_len + 1, dtype=np.float64) * control_dt
    anchored_plan = np.vstack((np.zeros((1, 2), dtype=np.float64), plan_traj))

    interpolated = np.empty((target_len, 2), dtype=np.float64)
    interpolated[:, 0] = np.interp(dst_times, src_times, anchored_plan[:, 0])
    interpolated[:, 1] = np.interp(dst_times, src_times, anchored_plan[:, 1])
    return interpolated


def _build_reference_trajectory(
    *,
    future_local: np.ndarray,
    current_state: np.ndarray,
    control_dt: float,
    wheelbase: float,
    max_steer_rad: float,
) -> np.ndarray:
    M = len(future_local)
    future_positions = np.zeros((M, 2), dtype=np.float64)
    future_positions[:, 0] = future_local[:, 1]
    future_positions[:, 1] = -future_local[:, 0]

    ego_xy = current_state[:2]
    ego_heading = current_state[2]

    reference_trajectory = np.zeros((M + 1, 5), dtype=np.float64)
    reference_trajectory[0] = current_state
    reference_trajectory[1:, :2] = future_positions

    headings = np.empty((M,), dtype=np.float64)
    last_heading = ego_heading
    last_position = ego_xy
    for idx in range(M):
        displacement = future_positions[idx] - last_position
        if np.linalg.norm(displacement) >= EPS:
            last_heading = math.atan2(displacement[1], displacement[0])
        headings[idx] = last_heading
        last_position = future_positions[idx]
    reference_trajectory[1:, 2] = headings

    segment_points = np.vstack((ego_xy, future_positions))
    delta_pos = np.diff(segment_points, axis=0)
    segment_dist = np.linalg.norm(delta_pos, axis=1)
    segment_speed = segment_dist / control_dt
    reference_trajectory[1:, 3] = segment_speed[:M]

    curvature = np.zeros((M,), dtype=np.float64)
    curvature[0] = _wrap_to_pi(headings[0] - ego_heading) / max(segment_dist[0], EPS)
    for idx in range(1, M):
        ds = max(segment_dist[idx], EPS)
        dtheta = _wrap_to_pi(headings[idx] - headings[idx - 1])
        curvature[idx] = dtheta / ds

    tan_limit = np.tan(max_steer_rad)
    steer_profile = np.arctan(np.clip(curvature * wheelbase, -tan_limit, tan_limit))
    steer_profile = np.clip(steer_profile, -max_steer_rad, max_steer_rad)
    reference_trajectory[1:, 4] = steer_profile

    return reference_trajectory


class OnsiteILQRController:
    def __init__(
        self,
        *,
        control_dt=0.1,
        max_steer=0.6,
        wheelbase=DEFAULT_WHEELBASE,
    ):
        self.control_dt = control_dt
        self.max_steer = max_steer
        self.wheelbase = wheelbase

    def act(self, plan_traj, current_speed, current_steer):
        plan_traj = np.asarray(plan_traj, dtype=np.float64)
        if plan_traj.ndim != 2 or plan_traj.shape[0] == 0:
            return 0.0, 0.0

        plan_traj = _interpolate_plan(plan_traj, control_dt=self.control_dt)
        current_state = np.array([0.0, 0.0, 0.0, current_speed, current_steer], dtype=np.float64)
        reference_trajectory = _build_reference_trajectory(
            future_local=plan_traj,
            current_state=current_state,
            control_dt=self.control_dt,
            wheelbase=self.wheelbase,
            max_steer_rad=self.max_steer,
        )
        throttle_brake, steering = plan2control(reference_trajectory, current_state)
        return steering, throttle_brake
