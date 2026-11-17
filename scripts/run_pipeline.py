# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import select
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def wait_for_pipeline(pipeline, deadline):
    """Service completion on the control thread without advancing its next tick."""
    notifications = getattr(pipeline, "get_idle_notifications", lambda: ())()
    while not getattr(pipeline, "should_exit", False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if not notifications:
            time.sleep(remaining)
            return
        ready, _, _ = select.select(notifications, (), (), remaining)
        if ready:
            pipeline.service_idle_notifications()


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
    parser.add_argument("--upper-body-latency-trace", default=None, help="Export bounded performance trace at shutdown")
    parser.add_argument("--latency-trace-duration-s", type=float, default=30.0)
    parser.add_argument("--latency-trace-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--upper-body-feedback-send-hwm",
        type=int,
        choices=range(1, 9),
        metavar="1..8",
        default=None,
        help="Override the upper-body feedback PUB queue limit",
    )
    parser.add_argument(
        "--upper-body-command-receive-hwm",
        type=int,
        choices=range(1, 101),
        metavar="1..100",
        default=None,
        help="Override the upper-body command SUB queue limit",
    )
    args = parser.parse_args()
    return args


def configure_upper_body_transport(cfg, args):
    """Apply bounded diagnostic/queue overrides before any sockets are created."""
    overrides = {}
    if getattr(args, "upper_body_latency_trace", None) is not None:
        overrides.update(
            latency_trace_path=args.upper_body_latency_trace,
            latency_trace_duration_s=args.latency_trace_duration_s,
            latency_trace_delay_s=args.latency_trace_delay_s,
        )
    if getattr(args, "upper_body_feedback_send_hwm", None) is not None:
        overrides["feedback_send_hwm"] = args.upper_body_feedback_send_hwm
    if getattr(args, "upper_body_command_receive_hwm", None) is not None:
        overrides["command_receive_hwm"] = args.upper_body_command_receive_hwm
    if overrides:
        from robojudo.controller.ctrl_cfgs import UpperBodyZmqCtrlCfg

        controllers = [c for c in cfg.ctrl if isinstance(c, UpperBodyZmqCtrlCfg) and c.feedback_endpoint is not None]
        if len(controllers) != 1:
            raise ValueError("upper-body transport overrides require exactly one feedback-enabled controller")
        controller = controllers[0]
        updated = type(controller).model_validate(
            {
                **controller.model_dump(),
                **overrides,
            }
        )
        cfg.ctrl = [updated if c is controller else c for c in cfg.ctrl]


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    configure_upper_body_transport(cfg, args)
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

    try:
        if not cfg.env.is_sim:
            pipeline.prepare()
        elif getattr(pipeline, "_has_default_pose_mode", False):
            pipeline._set_default_pose_mode(True)
            logger.warning("Sim mode — holding default pose, press R to start motion")

        while not getattr(pipeline, "should_exit", False):
            time_start = time.monotonic()
            pipeline.step()
            time_end = time.monotonic()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    wait_for_pipeline(pipeline, time_start + pipeline.dt)
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
        try:
            # Do not leave X2 native publishers alive while other teardown work
            # runs or raises. This also covers interruption and preparation errors.
            if cfg.env.env_type == "AgiBotCppEnv":
                pipeline.env.shutdown()
        finally:
            try:
                close_recording = getattr(pipeline, "close_recording", None)
                if close_recording is not None:
                    close_recording()
            finally:
                ctrl_manager = getattr(pipeline, "ctrl_manager", None)
                if ctrl_manager is not None:
                    ctrl_manager.close()


if __name__ == "__main__":
    main()
