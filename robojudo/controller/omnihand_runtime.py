"""Threaded direct control runtime for dual OmniHand Pro 2025 hands."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robojudo.controller.ctrl_cfgs import OmniHandCfg

logger = logging.getLogger(__name__)

OMNIHAND_JOINT_SUFFIXES = (
    "thumb_roll_joint",
    "thumb_abad_joint",
    "thumb_mcp_joint",
    "thumb_pip_joint",
    "index_abad_joint",
    "index_mcp_joint",
    "index_pip_joint",
    "middle_abad_joint",
    "middle_mcp_joint",
    "middle_pip_joint",
    "ring_mcp_joint",
    "pinky_mcp_joint",
)
OMNIHAND_LEFT_JOINT_NAMES = tuple(f"L_{suffix}" for suffix in OMNIHAND_JOINT_SUFFIXES)
OMNIHAND_RIGHT_JOINT_NAMES = tuple(f"R_{suffix}" for suffix in OMNIHAND_JOINT_SUFFIXES)
OMNIHAND_JOINT_NAMES = (*OMNIHAND_LEFT_JOINT_NAMES, *OMNIHAND_RIGHT_JOINT_NAMES)

# Physical API limits from OmniHand Pro 2025 (O12) SDK documentation.
OMNIHAND_LEFT_LIMITS = np.asarray(
    [
        (-0.9424777960769379, 0.0),
        (0.0, 1.387536755335492),
        (-0.8272860654453121, 0.0),
        (-1.2915436464758039, 0.0),
        (-0.2617993877991494, 0.2617993877991494),
        (0.0, 1.3526301702956054),
        (0.0, 1.530653753999027),
        (-0.2617993877991494, 0.2617993877991494),
        (0.0, 1.3578661580515883),
        (0.0, 1.8151424220741028),
        (0.0, 1.53588974175501),
        (0.0, 1.53588974175501),
    ],
    dtype=np.float64,
)
OMNIHAND_RIGHT_LIMITS = OMNIHAND_LEFT_LIMITS.copy()
OMNIHAND_RIGHT_LIMITS[:4] = np.asarray(
    [
        (0.0, 0.9424777960769379),
        (-1.387536755335492, 0.0),
        (-0.8272860654453121, 0.0),
        (-1.2915436464758039, 0.0),
    ],
    dtype=np.float64,
)

HandFactory = Callable[[OmniHandCfg, str], object]


@dataclass(frozen=True)
class OmniHandCommandFrame:
    """One inseparable left/right command originating from the same control frame."""

    left_command: np.ndarray
    right_command: np.ndarray
    source_timestamp_ns: int
    frame_id: int | None


def omnihand_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return OMNIHAND_LEFT_JOINT_NAMES
    if side == "right":
        return OMNIHAND_RIGHT_JOINT_NAMES
    raise ValueError(f"Unsupported OmniHand side {side!r}")


def omnihand_limits(side: str) -> np.ndarray:
    if side == "left":
        return OMNIHAND_LEFT_LIMITS
    if side == "right":
        return OMNIHAND_RIGHT_LIMITS
    raise ValueError(f"Unsupported OmniHand side {side!r}")


class OmniHandRuntime:
    """Apply dual-hand commands through the Omnihand SDK without blocking the robot loop.

    All Omnihand SDK calls are confined to one worker thread. The controller thread only
    sets commands and reads immutable copies of the latest runtime state.
    """

    def __init__(
        self,
        cfg: OmniHandCfg,
        *,
        hand_factory: HandFactory | None = None,
    ):
        self.cfg = cfg
        self._hand_factory = hand_factory or self._create_hand
        # Protect state shared by the controller thread and the SDK worker thread.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._enabled = False
        self._last_command_at: dict[str, float | None] = {"left": None, "right": None}
        self._last_joint_state_at: dict[str, float | None] = {"left": None, "right": None}
        self._applied_commands = {
            "left": np.zeros(12, dtype=np.float32),
            "right": np.zeros(12, dtype=np.float32),
        }
        self._measured_joint_positions = {
            "left": np.zeros(12, dtype=np.float32),
            "right": np.zeros(12, dtype=np.float32),
        }
        self._applied_source_timestamp_ns: int | None = None
        self._applied_frame_id: int | None = None
        # A capacity of one implements latest-command-wins semantics and prevents
        # stale teleoperation commands from accumulating behind slow hardware I/O.
        self._command_queue: queue.Queue[OmniHandCommandFrame] = queue.Queue(maxsize=1)

        self._thread = threading.Thread(target=self._run, name="OmniHandRuntime", daemon=True)
        self._thread.start()
        if not self._ready.wait(cfg.startup_timeout_s):
            self.close()
            raise TimeoutError(f"OmniHand runtime did not start within {cfg.startup_timeout_s:.1f} seconds")
        if self._error is not None:
            self.close()
            raise RuntimeError("failed to start OmniHand runtime") from self._error

    @staticmethod
    def _create_hand(cfg: OmniHandCfg, side: str):
        try:
            from omnihand import HandType, OmniHandPro2025
        except ImportError as exc:
            raise RuntimeError(
                "OmniHand control requires the vendor Python SDK; run "
                "`python submodule_install.py omnihand_sdk` in the active environment"
            ) from exc

        hand_type = HandType.LEFT if side == "left" else HandType.RIGHT
        hand_can_id = cfg.hand_can_id
        if hand_can_id is None:
            hand_can_id = OmniHandPro2025.kDefaultHandDeviceId
        if cfg.transport == "hcan":
            # HCAN exposes each USB adapter as a device index. Its SDK channel is
            # always zero, so users only configure the left/right adapter indices.
            adapter_index = cfg.left_adapter_index if side == "left" else cfg.right_adapter_index
            hand = OmniHandPro2025.create_hand_by_hcan(
                hand_type=hand_type,
                hand_device_id=hand_can_id,
                canfd_device_id=adapter_index,
                canfd_channel_id=0,
            )
        elif cfg.transport == "zlgcan":
            # ZLGCAN may expose multiple CAN-FD channels on one adapter, therefore
            # its channel IDs remain independently configurable.
            adapter_index = cfg.left_adapter_index if side == "left" else cfg.right_adapter_index
            configured_channel = (
                cfg.zlgcan_left_channel_id if side == "left" else cfg.zlgcan_right_channel_id
            )
            channel_id = (0 if side == "left" else 1) if configured_channel is None else configured_channel
            hand = OmniHandPro2025.create_hand_by_zlgcan(
                hand_type=hand_type,
                hand_device_id=hand_can_id,
                canfd_device_id=adapter_index,
                canfd_channel_id=channel_id,
            )
        elif cfg.transport == "socketcan":
            interface = cfg.left_interface if side == "left" else cfg.right_interface
            hand = OmniHandPro2025.create_hand_socketcan(
                hand_type=hand_type,
                hand_device_id=hand_can_id,
                can_interface=interface,
            )
        else:  # pragma: no cover - protected by the config literal
            raise ValueError(f"Unsupported OmniHand transport {cfg.transport!r}")

        if hand is None:
            raise RuntimeError(f"OmniHand SDK could not create the {side} hand")
        if not hand.init():
            raise RuntimeError(f"OmniHand SDK failed to initialize the {side} hand")
        return hand

    def set_takeover_enabled(self, enabled: bool):
        """Open or close the hardware command gate and invalidate old pending commands."""

        enabled = bool(enabled)
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
            self._last_command_at = {"left": None, "right": None}
            self._applied_source_timestamp_ns = None
            self._applied_frame_id = None
        self._discard_pending_commands()
        logger.warning("OmniHand control %s", "enabled" if enabled else "disabled")

    def set_joint_commands(
        self,
        left_joint_positions,
        right_joint_positions,
        source_timestamp_ns: int | None = None,
        frame_id: int | None = None,
    ) -> np.ndarray:
        """Set the newest atomic left/right joint command without blocking the caller."""
        left_command = np.asarray(left_joint_positions, dtype=np.float64)
        right_command = np.asarray(right_joint_positions, dtype=np.float64)
        if left_command.shape != (12,) or right_command.shape != (12,):
            raise ValueError("OmniHand joint commands must each have shape (12,)")
        if not np.isfinite(left_command).all() or not np.isfinite(right_command).all():
            raise ValueError("OmniHand joint commands must be finite")
        # Clip before enqueueing so hardware execution and recorded actions use the
        # same command values.
        left_command = np.clip(left_command, OMNIHAND_LEFT_LIMITS[:, 0], OMNIHAND_LEFT_LIMITS[:, 1])
        right_command = np.clip(right_command, OMNIHAND_RIGHT_LIMITS[:, 0], OMNIHAND_RIGHT_LIMITS[:, 1])
        timestamp_ns = time.time_ns() if source_timestamp_ns is None else int(source_timestamp_ns)
        command_frame = OmniHandCommandFrame(
            left_command=left_command.copy(),
            right_command=right_command.copy(),
            source_timestamp_ns=timestamp_ns,
            frame_id=None if frame_id is None else int(frame_id),
        )
        try:
            self._command_queue.put_nowait(command_frame)
        except queue.Full:
            # Never wait for the worker: replace the unsent command with this newer frame.
            try:
                self._command_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - another producer cannot remove entries
                pass
            self._command_queue.put_nowait(command_frame)
        return np.concatenate((left_command, right_command)).astype(np.float32)

    def _discard_pending_commands(self):
        while True:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _validate_joint_positions(value, side: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (12,) or not np.isfinite(result).all():
            raise RuntimeError(f"OmniHand {side} joint positions must contain 12 finite angles")
        return result

    def _send_joint_command(self, hands: dict[str, object], side: str, joint_command: np.ndarray, now: float):
        limits = omnihand_limits(side)
        clipped = np.clip(joint_command, limits[:, 0], limits[:, 1])
        if not np.array_equal(clipped, joint_command):
            logger.warning("Clipped %s OmniHand command to SDK limits", side)
        # This method is called only by _run; vendor objects never cross threads.
        hands[side].set_all_active_joint_angles(clipped.tolist())
        with self._lock:
            self._applied_commands[side] = clipped.astype(np.float32)
            self._last_command_at[side] = now

    def _update_measured_joint_positions(self, hands: dict[str, object], now: float):
        # Read both hands before publishing either result so returned data never contains
        # a newly sampled left hand paired with an older right hand from this cycle.
        measured_positions = {
            side: self._validate_joint_positions(hand.get_all_active_joint_angles(), side)
            for side, hand in hands.items()
        }
        with self._lock:
            for side, positions in measured_positions.items():
                self._measured_joint_positions[side] = positions
                self._last_joint_state_at[side] = now

    def _run(self):
        hands = {}
        try:
            # Hardware initialization lives in the worker because the SDK owns native
            # resources that must be used consistently from the same thread.
            hands = {side: self._hand_factory(self.cfg, side) for side in ("left", "right")}
            now = time.monotonic()
            self._update_measured_joint_positions(hands, now)
            self._ready.set()
            joint_state_period_s = 1.0 / self.cfg.joint_state_fps
            next_joint_state_at = now + joint_state_period_s

            while not self._stop.is_set():
                try:
                    command_frame = self._command_queue.get(timeout=0.01)
                except queue.Empty:
                    pass
                else:
                    with self._lock:
                        enabled = self._enabled
                    if enabled:
                        now = time.monotonic()
                        # Both commands come from one command frame. Apply them back to
                        # back and publish the source metadata only after both succeed.
                        self._send_joint_command(hands, "left", command_frame.left_command, now)
                        self._send_joint_command(hands, "right", command_frame.right_command, now)
                        with self._lock:
                            self._applied_source_timestamp_ns = command_frame.source_timestamp_ns
                            self._applied_frame_id = command_frame.frame_id

                now = time.monotonic()
                if now >= next_joint_state_at:
                    self._update_measured_joint_positions(hands, now)
                    next_joint_state_at = now + joint_state_period_s
        except Exception as exc:
            self._error = exc
            logger.exception("OmniHand runtime stopped: %s", exc)
            self._ready.set()
        finally:
            hands.clear()

    def get_data(self) -> dict:
        """Return consistent copies of measured positions, commands, and freshness state."""

        if self._error is not None:
            raise RuntimeError("OmniHand runtime is unavailable") from self._error
        now = time.monotonic()
        with self._lock:
            enabled = self._enabled
            command_ages = {
                side: None if timestamp is None else now - timestamp
                for side, timestamp in self._last_command_at.items()
            }
            joint_state_ages = {
                side: None if timestamp is None else now - timestamp
                for side, timestamp in self._last_joint_state_at.items()
            }
            measured_joint_positions = np.concatenate(
                (self._measured_joint_positions["left"], self._measured_joint_positions["right"])
            ).astype(np.float32)
            applied_joint_commands = np.concatenate(
                (self._applied_commands["left"], self._applied_commands["right"])
            ).astype(np.float32)
            applied_source_timestamp_ns = self._applied_source_timestamp_ns
            applied_frame_id = self._applied_frame_id

        # Freshness uses local monotonic time. Source wall-clock timestamps are kept
        # only for correlation because the publisher and robot clocks may differ.
        side_fresh = {
            side: enabled and age is not None and age <= self.cfg.command_timeout_s
            for side, age in command_ages.items()
        }
        joint_state_fresh = all(
            age is not None and age <= self.cfg.joint_state_timeout_s for age in joint_state_ages.values()
        )
        received_ages = [age for age in command_ages.values() if age is not None]
        return {
            "joint_names": list(OMNIHAND_JOINT_NAMES),
            "joint_positions": measured_joint_positions,
            "joint_position_commands": applied_joint_commands,
            "left_fresh": side_fresh["left"],
            "right_fresh": side_fresh["right"],
            "joint_state_fresh": joint_state_fresh,
            "fresh": side_fresh["left"] and side_fresh["right"] and joint_state_fresh,
            "age_s": max(received_ages) if received_ages else None,
            "enabled": enabled,
            "applied_source_timestamp_ns": applied_source_timestamp_ns,
            "applied_frame_id": applied_frame_id,
        }

    def reset(self):
        self.set_takeover_enabled(False)

    def close(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("OmniHand runtime did not stop within 3 seconds")
            self._thread = None
