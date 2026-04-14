#!/usr/bin/env python3
"""OnSite UniAD driver."""

import argparse
import importlib
import logging
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIAD_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, UNIAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model

from metadrive.misc.onsite_middleware import OnSiteSwitch, SIM_STATE, TERMINAL_TYPE
from metadrive.misc.onsite_middleware.onsite_proto.main.proto.enums_pb2 import (
    NT_ABORT_TEST,
    NT_FINISH_TEST,
    NT_START_TEST,
)
from tools.data_converter.onsite_input_builder import build_uniad_input
from tools.data_converter.onsite_ilqr_control import OnsiteILQRController


logger = logging.getLogger("onsite_uniad_driver")

ONSITE_CONTROL_DT = 0.1
ONSITE_MAX_STEER_RAD = math.radians(40.0)



def process_notify(middleware, sim_state, session_id, actor_id):
    notifies = middleware.recv_all_notifies()
    if not notifies:
        return sim_state, session_id, actor_id

    for notify in notifies:
        logger.info("Received Notify: type=%s role_id=%s", notify.type, notify.role_id)
        if notify.type in (NT_ABORT_TEST, NT_FINISH_TEST):
            sim_state = SIM_STATE.IDLE
            session_id = ""
            actor_id = ""
            continue
        if notify.type == NT_START_TEST:
            sim_state = SIM_STATE.STARTED

    return sim_state, session_id, actor_id


def wait_for_started_inputs(middleware, sim_state, session_id, actor_id, none_sleep_s):
    images = None
    vehicle_feedback = None

    while True:
        sim_state, session_id, actor_id = process_notify(middleware, sim_state, session_id, actor_id)
        if sim_state != SIM_STATE.STARTED:
            return sim_state, session_id, actor_id, None, None

        ret = None
        if vehicle_feedback is None:
            vehicle_feedback = middleware.recv_vehicle_feedback()
        if images is None:
            ret, images = middleware.recv_image()

        if ret == 403:
            middleware.send_last_vehicle_control()
            images = None
            vehicle_feedback = None
        if images and vehicle_feedback:
            return sim_state, session_id, actor_id, images, vehicle_feedback



def load_uniad_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        module_dir = os.path.dirname(plugin_dir if plugin_dir else cfg.filename).split("/")
        module_path = module_dir[0]
        for part in module_dir[1:]:
            module_path = module_path + "." + part
        importlib.import_module(module_path)
    cfg.model.motion_head.anchor_info_path = str(UNIAD_ROOT / "data/others/motion_anchor_infos_mode6.pkl")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model = MMDataParallel(model.to(device), device_ids=[torch.cuda.current_device()] if device.type == "cuda" else None)
    model.eval()
    if hasattr(model.module, "planning_head"):
        model.module.planning_head.use_col_optim = False
    return model


def main():
    parser = argparse.ArgumentParser(description="OnSite UniAD trajectory driver")
    parser.add_argument("--onsite_dir", type=str, default="onsite", help="OnSite workspace directory")
    parser.add_argument("--uniad_config", type=str, default="UniAD/projects/configs/stage2_e2e/base_e2e.py")
    parser.add_argument("--checkpoint", type=str, default="UniAD/ckpts/uniad_base_e2e.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--none_sleep_s", type=float, default=0.02)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    resolved_log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    device = torch.device(args.device)
    model = load_uniad_model(args.uniad_config, args.checkpoint, device)
    logging.basicConfig(level=resolved_log_level, force=True)
    logging.getLogger().setLevel(resolved_log_level)
    logging.getLogger("metadrive").setLevel(resolved_log_level)
    logging.getLogger("metadrive.misc.onsite_middleware").setLevel(resolved_log_level)
    logging.getLogger("metadrive.misc.onsite_middleware.onsite_switch").setLevel(resolved_log_level)
    logger.setLevel(resolved_log_level)
    logger.debug("Driver args: %s", vars(args))
    controller = OnsiteILQRController(control_dt=ONSITE_CONTROL_DT, max_steer=ONSITE_MAX_STEER_RAD)
    middleware = OnSiteSwitch(onsite_dir=args.onsite_dir, terminal_type=TERMINAL_TYPE.TESTEE)

    sim_state = SIM_STATE.IDLE
    session_id = ""
    actor_id = ""
    scene_name = ""
    exit_code = 0

    try:
        while True:
            sim_state, session_id, actor_id = process_notify(middleware, sim_state, session_id, actor_id)

            if sim_state == SIM_STATE.IDLE:
                session_id, actor_id, scene_name = "", "", ""
                result = middleware.recv_actor_prepare()
                if result is not None:
                    session_id, actor_id, _, scene_name = result
                    sim_state = SIM_STATE.PREPARED
                time.sleep(0.5)

            elif sim_state == SIM_STATE.PREPARED:
                middleware.send_actor_prepare_result(session_id=session_id, actor_id=actor_id, result=True)
                time.sleep(0.5)

            elif sim_state == SIM_STATE.STARTED:
                sim_state, session_id, actor_id, images, vehicle_feedback = wait_for_started_inputs(
                    middleware,
                    sim_state,
                    session_id,
                    actor_id,
                    args.none_sleep_s,
                )
                if sim_state != SIM_STATE.STARTED:
                    continue

                data, current_speed, current_steer = build_uniad_input(images, vehicle_feedback, device, scene_name)
                logger.info(
                    "UniAD input summary: timestamp=%s command=%s speed=%.4f steer=%.4f",
                    data["timestamp"],
                    data["command"],
                    current_speed,
                    current_steer,
                )
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, **data)
                plan_traj = results[0]["planning"]["result_planning"]["sdc_traj"][0].detach().cpu().numpy()
                steering, throttle_brake = controller.act(plan_traj, current_speed=current_speed, current_steer=current_steer)
                logger.info(
                    "UniAD output summary: traj=%s steering=%.4f throttle_brake=%.4f",
                    plan_traj,
                    steering,
                    throttle_brake,
                )
                middleware.send_vehicle_control(steering, throttle_brake)
            
    except KeyboardInterrupt:
        exit_code = 130
        logger.info("Interrupted by user")
    except BaseException:
        exit_code = 1
        logger.exception("Unhandled exception in OnSite UniAD driver")
    finally:
        middleware.close()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
