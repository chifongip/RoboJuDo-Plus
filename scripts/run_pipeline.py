# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Publish upper-body control samples to robojudo-recorder",
    )
    parser.add_argument(
        "--record-endpoint",
        type=str,
        default=None,
        help="Override the recorder PUSH bind endpoint",
    )
    parser.add_argument(
        "--record-task",
        type=str,
        default=None,
        help="Task text stored with recorded episodes",
    )
    parser.add_argument(
        "--gr00t-task",
        type=str,
        default=None,
        help="Override the language instruction published to GR00T",
    )
    parser.add_argument(
        "--gr00t-command-endpoint",
        type=str,
        default=None,
        help="Override the GR00T command SUB endpoint",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    if args.record or args.record_endpoint is not None or args.record_task is not None:
        cfg.record = type(cfg.record)(
            **{
                **cfg.record.model_dump(),
                "enabled": True,
                **({"endpoint": args.record_endpoint} if args.record_endpoint is not None else {}),
                **({"task": args.record_task} if args.record_task is not None else {}),
            },
        )
    gr00t_task = args.gr00t_task or args.record_task
    if gr00t_task is not None:
        for ctrl_cfg in cfg.ctrl:
            if ctrl_cfg.ctrl_type == "Gr00tZmqCtrl":
                ctrl_cfg.observation_task = gr00t_task
    if args.gr00t_command_endpoint is not None:
        for ctrl_cfg in cfg.ctrl:
            if ctrl_cfg.ctrl_type == "Gr00tZmqCtrl":
                ctrl_cfg.endpoint = args.gr00t_command_endpoint

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()
    elif getattr(pipeline, "_has_default_pose_mode", False):
        pipeline._set_default_pose_mode(True)
        logger.warning("Sim mode — holding default pose, press R to start motion")

    try:
        while not getattr(pipeline, "should_exit", False):
            time_start = time.time()
            pipeline.step()
            time_end = time.time()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    time.sleep(time_diff)
                elif not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        if cfg.env.env_type == "AgiBotCppEnv":
                            logger.critical(
                                "Excessive X2 frame drop; allowing the AimDK hard watchdog to resolve the fault"
                            )
                        else:
                            logger.critical("Exiting due to excessive frame drop")
                            pipeline.env.shutdown()
                            time.sleep(10)
                            break
    finally:
        close_recording = getattr(pipeline, "close_recording", None)
        if close_recording is not None:
            close_recording()
        ctrl_manager = getattr(pipeline, "ctrl_manager", None)
        if ctrl_manager is not None:
            ctrl_manager.close()


if __name__ == "__main__":
    main()
