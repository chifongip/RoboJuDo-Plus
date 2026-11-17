"""Owner-thread construction of one executable frame per desired frame."""

import math
import time
from contextlib import ExitStack
from numbers import Real

import numpy as np

from robojudo.controller.xr_target_filter import filter_targets


class XRTargetExecutor:
    def __init__(self, cfg):
        try:
            from xrobotoolkit_teleop.common.executor_model import CONTRACT, ExecutorModel
            from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
            from xrobotoolkit_teleop.integrations.robot_profiles import load_profile
        except ImportError as exc:
            raise RuntimeError("Install XR executor support with scripts/install_xr_executor.py") from exc
        self.resources = ExitStack()
        try:
            settings = TeleopSettings(
                collision_checking=cfg.executor_collision_checking,
                collision_margin=cfg.executor_collision_margin,
                collision_trigger=cfg.executor_collision_trigger,
            )
            self.profile = self.resources.enter_context(load_profile(cfg.xr_profile, settings))
            self.profile.validate()
            if tuple(cfg.joint_names) != tuple(self.profile.spec.joints):
                raise ValueError("executor arm order does not match model")
            self.model = ExecutorModel(self.profile, settings)
            self.hand_command_mode = None if self.profile.hands is None else cfg.hand_command_mode
            self.hand_velocity_limit = None if self.profile.hands is None else cfg.hand_velocity_limit
            self.contract = {
                "contract": CONTRACT,
                "robot_profile": cfg.xr_profile,
                "model_fingerprint": self.model.fingerprint,
                "collision_checking": settings.collision_checking,
                "collision_margin": settings.collision_margin,
                "collision_trigger": settings.collision_trigger,
            }
            if self.profile.hands is not None:
                self.contract.update(
                    hand_command_mode=self.hand_command_mode, hand_velocity_limit=self.hand_velocity_limit
                )
        except Exception:
            self.resources.close()
            raise
        self.hand_names = [] if self.profile.hands is None else list(self.profile.hands.active)
        self.reset()

    def close(self):
        self.resources.close()

    def reset(self):
        self.prepared = None
        self.consumed = -1
        self.baseline_id = None
        self.issued = None

    def validate(self, feedback, message, positions):
        if message.get("executor") != self.contract:
            raise ValueError("incompatible executor model or collision policy or finger settings")
        active = message.get("active_joint_names")
        if (
            not isinstance(active, list)
            or any(not isinstance(n, str) for n in active)
            or len(active) != len(set(active))
            or not set(active) <= set(feedback.command_names)
        ):
            raise ValueError("invalid active command joint names")
        caps = message.get("velocity_limits")
        if not isinstance(caps, dict) or set(caps) != {"arm", "return"}:
            raise ValueError("missing executor velocity caps")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0 for v in caps.values()):
            raise ValueError("invalid executor velocity caps")
        if message is self.prepared:
            if feedback.command_id != self.baseline_id or positions != self.issued:
                raise ValueError("executable transition baseline or target changed")
            return
        if message["frame_id"] <= self.consumed:
            raise ValueError("desired frame already consumed")
        limits = feedback._states[message["state_id"]][1]["position_limits"]
        model_limits = {**self.profile.spec.limits, **({} if self.profile.hands is None else self.profile.hands.limits)}
        for name, value in positions.items():
            low = max(limits[name][0], model_limits[name][0])
            high = min(limits[name][1], model_limits[name][1])
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("desired target exceeds compatible position limits")

    def prepare(self, feedback, desired, measured):
        feedback.validate(desired, desired["positions"])
        self.consumed = desired["frame_id"]
        baseline = feedback._completed_positions
        if baseline is None:
            raise ValueError("executor has no completed baseline")
        state = feedback._states[desired["state_id"]][1]
        caps = desired["velocity_limits"]
        limits, speeds = {}, {}
        homing = desired["status"] == "homing"
        for name in feedback.command_names:
            arm = name in feedback.cfg.joint_names
            model_limits = self.profile.spec.limits if arm else self.profile.hands.limits
            limits[name] = [
                max(state["position_limits"][name][0], model_limits[name][0]),
                min(state["position_limits"][name][1], model_limits[name][1]),
            ]
            if arm:
                speed = min(feedback.cfg.max_joint_velocity_rad_s, caps["arm"], self.profile.velocities[name])
                if homing:
                    speed = min(speed, feedback.cfg.return_joint_velocity_rad_s, caps["return"])
            else:
                speed = 0.0 if homing else min(self.hand_velocity_limit, self.profile.hands.velocities[name])
            speeds[name] = speed
        active = set(desired["active_joint_names"])
        wanted = {n: value if n in active else baseline[n] for n, value in desired["positions"].items()}
        if homing:
            wanted.update({n: baseline[n] for n in self.hand_names})
        # Arms and filtered fingers share EMA, velocity bounds and float32 rounding.
        direct_fingers = self.hand_command_mode == "direct" and not homing
        filtered_names = [n for n in wanted if not direct_fingers or n in feedback.cfg.joint_names]
        issued = filter_targets(
            {n: wanted[n] for n in filtered_names},
            {n: baseline[n] for n in filtered_names},
            limits,
            speeds,
            state["control_dt"],
            feedback.cfg.ema_alpha,
        )
        if direct_fingers:
            for name in self.hand_names:
                low, high = limits[name]
                lo32, hi32 = np.float32(low), np.float32(high)
                if float(lo32) < low:
                    lo32 = np.nextafter(lo32, np.float32(np.inf))
                if float(hi32) > high:
                    hi32 = np.nextafter(hi32, np.float32(-np.inf))
                if lo32 > hi32:
                    raise ValueError("no float32 finger target within limits")
                issued[name] = float(np.clip(np.float32(wanted[name]), lo32, hi32))
        started = time.monotonic()
        try:
            self.model.validate_transition(measured, baseline, issued)
        except RuntimeError as exc:
            # An unissued candidate rejection holds without revoking the clutch.
            # Runtime/transport failures after admission retain existing invalidation.
            raise ValueError(f"executor transition rejected: {exc}") from exc
        result = {**desired, "positions": issued, "desired_positions": wanted}
        result["_command_timestamps"] = dict(desired.get("_command_timestamps", {}))
        result["_command_timestamps"].update(
            executor_validation_started=started, executor_validation_completed=time.monotonic()
        )
        self.prepared, self.issued = result, dict(issued)
        self.baseline_id = feedback.command_id
        # Collision checking may have consumed the remaining measurement budget.
        feedback.validate(result, issued)
        return result
