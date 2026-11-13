"""Dual OmniHand Pro control with one isolated SDK process owning both adapters.

Both dex-teleop and checked XR proposals use this runtime. Connection policy is
configuration-driven; XR explicitly requires serial-verified robot-side HCAN.
"""

import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time

import numpy as np

from robojudo.controller.ctrl_cfgs import OmniHandCfg

logger = logging.getLogger(__name__)


class HandIdentityError(RuntimeError):
    """An opened adapter reported a different physical hand."""


class HandConfigurationError(RuntimeError):
    """The SDK installation cannot support this runtime."""


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


PASSIVE_SUFFIXES = (
    "thumb_dip_joint",
    "index_dip_joint",
    "middle_dip_joint",
    "ring_pip_joint",
    "ring_dip_joint",
    "pinky_pip_joint",
    "pinky_dip_joint",
)
ALL_HAND_NAMES = tuple(
    f"{prefix}_{suffix}" for prefix in ("L", "R") for suffix in (*OMNIHAND_JOINT_SUFFIXES, *PASSIVE_SUFFIXES)
)


def measured_command_baseline(positions):
    """Represent a sampled startup pose as bounded float32 command values."""
    result = {}
    limits = np.concatenate((OMNIHAND_LEFT_LIMITS, OMNIHAND_RIGHT_LIMITS))
    for name, (lower, upper) in zip(OMNIHAND_JOINT_NAMES, limits, strict=True):
        value = positions[name]
        if not np.isfinite(value) or value < lower - 1e-6 or value > upper + 1e-6:
            raise ValueError(f"measured startup finger is outside SDK limits: {name}")
        value = np.float32(np.clip(value, lower, upper))
        if float(value) < lower:
            value = np.nextafter(value, np.float32(np.inf))
        if float(value) > upper:
            value = np.nextafter(value, np.float32(-np.inf))
        result[name] = float(value)
    return result


def connect_verified_hand(cfg, side):
    """Open exactly the requested adapter and verify identity without moving it."""
    from omnihand import HandType, OmniHandPro2025

    serial = getattr(cfg, f"{side}_hcan_serial")
    expected = getattr(cfg, f"{side}_hand_serial")
    if not serial or not expected:
        raise ValueError(f"{side} HCAN adapter serial and expected hand serial are required")
    device_id = cfg.hand_can_id or 1
    context = f"{side}: HCAN adapter {serial!r}"
    try:
        hand = OmniHandPro2025.create_hand_by_hcan(
            hand_type=HandType.LEFT if side == "left" else HandType.RIGHT,
            hand_device_id=device_id,
            hcan_serial_number=serial,
            canfd_channel_id=0,
        )
    except Exception as exc:
        raise RuntimeError(f"{context}: open failed: {exc}") from exc
    if hand is None:
        raise RuntimeError(f"{side}: HCAN adapter {serial!r} could not be opened")
    try:
        hand.set_frame_recv_timeout(cfg.frame_recv_timeout_ms)
        vendor, device = hand.get_vendor_info(), hand.get_device_info()
        actual = vendor.product_seq_num.rstrip("\0")
    except Exception as exc:
        raise RuntimeError(f"{context}: identity read failed (expected {expected!r}): {exc}") from exc
    if not vendor.product_model or not actual:
        raise RuntimeError(f"{context}: identity unreadable: expected {expected!r}, received {actual!r}")
    if device.hand_device_id != device_id or actual != expected:
        raise HandIdentityError(
            f"{context}: hand identity mismatch: expected {expected!r}/CAN ID {device_id}, "
            f"received {actual!r}/CAN ID {device.hand_device_id}"
        )
    return hand


def create_hand(cfg: OmniHandCfg, side: str):
    try:
        from omnihand import HandType, OmniHandPro2025
    except ImportError as exc:
        raise HandConfigurationError(
            "OmniHand control requires the vendor Python SDK; run "
            "`python submodule_install.py omnihand_sdk` in the active environment"
        ) from exc

    if cfg.transport == "hcan" and cfg.left_hcan_serial is not None:
        return connect_verified_hand(cfg, side)
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
        configured_channel = cfg.zlgcan_left_channel_id if side == "left" else cfg.zlgcan_right_channel_id
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
    hand.set_frame_recv_timeout(cfg.frame_recv_timeout_ms)
    return hand


class _StopFlag:
    """A process may die while waiting, so cancellation must not use a Condition."""

    def __init__(self, context):
        self._value = context.Value("b", 0, lock=False)

    def is_set(self):
        return bool(self._value.value)

    def set(self):
        self._value.value = 1

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, 0.005))
        return True


class _CommandMailbox:
    """Single shared command slot; overwriting never depends on a Queue feeder."""

    def __init__(self, context):
        self._buffer = context.Array("B", 16_384)
        self._length = context.Value("i", 0, lock=False)

    def put_nowait(self, command):
        payload = pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self._buffer):
            raise ValueError("OmniHand command exceeds mailbox capacity")
        with self._buffer.get_lock():
            self._buffer[: len(payload)] = payload
            self._length.value = len(payload)

    def get_nowait(self):
        with self._buffer.get_lock():
            if not self._length.value:
                raise queue.Empty
            payload = bytes(self._buffer[: self._length.value])
            self._length.value = 0
        return pickle.loads(payload)


def _publish_state(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def _hand_loop(cfg, side, hand, commands, stop, gate, publish, fail, measurements=None):
    """Serialize calls on one handle, publishing writes before the next read."""
    state = {
        "positions": None,
        "sampled_at": None,
        "applied": None,
        "targets": None,
        "commanded_at": None,
        "generation": None,
        "consecutive_failures": 0,
        "last_read_error": None,
    }
    stage = "read measurements"
    last_read_log = float("-inf")
    adapter = (
        getattr(cfg, f"{side}_interface")
        if cfg.transport == "socketcan"
        else getattr(cfg, f"{side}_hcan_serial") or getattr(cfg, f"{side}_adapter_index")
    )

    def allowed(generation, deadline):
        with gate.get_lock():
            now = time.monotonic()
            fresh = measurements is None or all(
                stamp > 0 and 0 <= now - stamp <= cfg.joint_state_timeout_s for stamp in measurements
            )
            return fresh and not stop.is_set() and gate[0] == generation and gate[1] == 1 and now <= deadline

    try:
        while not stop.is_set():
            cycle_started = time.monotonic()
            try:
                generation, deadline, identity, angles, submitted_at = commands.get_nowait()
            except queue.Empty:
                pass
            else:
                state["applied"], state["generation"] = None, None
                if allowed(generation, deadline):
                    stage = "write"
                    hand.set_all_active_joint_angles(angles)
                    # In-flight writes cannot be undone, even after revocation.
                    state["targets"], state["commanded_at"] = angles, submitted_at
                    if allowed(generation, deadline):
                        state["applied"], state["generation"] = identity, generation
                if state["positions"] is not None:
                    # Acknowledgements must not wait for another blocking read.
                    # Keep the previous measurement's original sample timestamp.
                    publish(side, state.copy())
            if stop.is_set():
                return
            stage = "read measurements"
            sampled_at = time.monotonic()
            try:
                angles = np.asarray(hand.get_all_joint_angles(), dtype=float)
                if angles.shape != (19,) or not np.isfinite(angles).all():
                    raise ValueError(
                        f"expected 19 finite finger angles; received shape={angles.shape}, "
                        f"nonfinite_indices={np.flatnonzero(~np.isfinite(angles)).tolist()}"
                    )
            except Exception as exc:
                state["consecutive_failures"] += 1
                state["last_read_error"] = f"{type(exc).__name__}: {exc}"[:1000]
                if time.monotonic() - last_read_log >= 1.0:
                    logger.warning(
                        "%s: %s adapter %r: measurement rejected: %s",
                        side,
                        cfg.transport,
                        adapter,
                        state["last_read_error"],
                    )
                    last_read_log = time.monotonic()
            else:
                state.update(
                    positions=angles.tolist(), sampled_at=sampled_at, consecutive_failures=0, last_read_error=None
                )
                if measurements is not None:
                    with gate.get_lock():
                        measurements[0 if side == "left" else 1] = sampled_at
            publish(side, state.copy())
            stop.wait(max(0.0, 1.0 / cfg.joint_state_fps - (time.monotonic() - cycle_started)))
    except Exception as exc:
        fail(side, stage, exc)


def _worker(cfg, commands, states, stop, gate, errors, measurements=None):
    """Initialize sequentially, then give each handle its own I/O thread."""
    hands, snapshot, threads = {}, {}, []
    local_commands = {side: queue.Queue(1) for side in ("left", "right")}
    snapshot_lock = threading.Lock()
    failure_lock = threading.Lock()
    failed = False

    def publish(side, state):
        with snapshot_lock:
            snapshot[side] = state
            if len(snapshot) == 2:
                _publish_state(states, snapshot.copy())

    def fail(side, stage, exc):
        nonlocal failed
        with failure_lock:
            if failed:
                return
            failed = True
            with gate.get_lock():
                gate[0] += 1
                gate[1] = 0
            if cfg.transport == "socketcan":
                adapter = getattr(cfg, f"{side}_interface")
            else:
                adapter = getattr(cfg, f"{side}_hcan_serial") or getattr(cfg, f"{side}_adapter_index")
            # Faults have a separate reliable slot, independent of measurements.
            errors.put_nowait(
                {
                    "message": f"{side}: {cfg.transport} adapter {adapter!r}: {stage}: {type(exc).__name__}: {exc}"[
                        :4000
                    ],
                    "fatal": stage != "connect/start worker"
                    or isinstance(exc, HandIdentityError | HandConfigurationError | ValueError | ImportError),
                }
            )
            stop.set()

    side = "left"
    try:
        for side in ("left", "right"):
            if stop.is_set():
                return
            hands[side] = create_hand(cfg, side)
        for side in ("left", "right"):
            thread = threading.Thread(
                target=_hand_loop,
                args=(cfg, side, hands[side], local_commands[side], stop, gate, publish, fail, measurements),
                daemon=True,
                name=f"omnihand-{side}",
            )
            threads.append(thread)
            thread.start()
        while not stop.is_set():
            if measurements is not None:
                with gate.get_lock():
                    if gate[1] and not all(
                        stamp > 0 and 0 <= time.monotonic() - stamp <= cfg.joint_state_timeout_s
                        for stamp in measurements
                    ):
                        gate[0] += 1
                        gate[1] = 0
                        gate[2] += 1
            try:
                generation, deadline, identity, angles, submitted_at = commands.get_nowait()
            except queue.Empty:
                pass
            else:
                for index, side in enumerate(("left", "right")):
                    message = (generation, deadline, identity, angles[index * 12 : (index + 1) * 12], submitted_at)
                    _publish_state(local_commands[side], message)
            stop.wait(0.001)
    except Exception as exc:
        fail(side, "connect/start worker", exc)
    finally:
        stop.set()
        # Never destroy handles while an I/O thread is using them. The parent
        # terminates this process if a native call prevents these joins.
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        snapshot.clear()
        hands.clear()


class OmniHandRuntime:
    """Bounded parent-side coordinator; application IDs require both hand writes."""

    def __init__(self, cfg, *, read_only=False, require_serials=False):
        verified = cfg.transport == "hcan" and all(
            getattr(cfg, f"{side}_{kind}_serial") for side in ("left", "right") for kind in ("hcan", "hand")
        )
        if require_serials and not verified:
            raise ValueError("XR hardware requires HCAN adapter and exact hand serials for both sides")
        self.cfg, self.read_only = cfg, read_only
        self._can_reconnect = cfg.auto_reconnect and verified
        self._ctx = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._started = threading.Event()
        self._process = None
        self._error = None
        self._states = {}
        self._pending = None
        self._sequence = 0
        self._closed = False
        self._requested = False
        self._healthy = False
        self._status = "starting"
        self._connection_generation = 0
        self._stream_generation = 0
        self._last_error = None
        self._initial_targets = None
        self._attempt_ready = False
        self._seen_outages = 0
        self._supervisor = threading.Thread(target=self._supervise, name="omnihand-supervisor", daemon=True)
        self._supervisor.start()
        try:
            if not self._started.wait(cfg.startup_timeout_s):
                raise TimeoutError("OmniHand startup verification timed out")
            self._poll()
        except Exception:
            self.close()
            raise

    def _start_worker(self):
        # A killed native process may leave IPC locks held. Each attempt gets
        # completely new IPC; old snapshots retain their actual timestamps.
        with self._lock:
            self._stop = _StopFlag(self._ctx)
            # Generation, effective authorization, worker-observed outages.
            self._gate = self._ctx.Array("q", [0, 0, 0])
            self._seen_outages = 0
            self._attempt_ready = False
            self._measurements = self._ctx.Array("d", [0.0, 0.0], lock=False)
            self._commands = _CommandMailbox(self._ctx)
            self._state_queue = self._ctx.Queue(1)
            self._errors = _CommandMailbox(self._ctx)
            self._process = self._ctx.Process(
                target=_worker,
                args=(
                    self.cfg,
                    self._commands,
                    self._state_queue,
                    self._stop,
                    self._gate,
                    self._errors,
                    self._measurements,
                ),
                daemon=True,
            )
            self._process.start()
            self._connection_generation += 1
            self._attempt_started = time.monotonic()

    def _reap_worker(self):
        process = self._process
        if process is None:
            return
        self._stop.set()
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            if process.is_alive():
                raise RuntimeError("OmniHand worker could not be stopped; refusing a second SDK owner")
        self._state_queue.cancel_join_thread()
        self._state_queue.close()
        self._process = None

    def _set_health(self, healthy):
        if healthy != self._healthy:
            self._stream_generation += 1
            self._pending = None
            self._healthy = healthy
            with self._gate.get_lock():
                self._gate[0] += 1
                self._gate[1] = 0
            if not self._shutdown.is_set():
                log = logger.info if healthy else logger.warning
                log("OmniHand hardware feedback %s", "recovered" if healthy else "stale; holding commands")
        if healthy:
            with self._gate.get_lock():
                self._gate[1] = int(self._requested and not self.read_only and not self._closed)

    def _merge_snapshot(self, snapshot):
        """Keep historical targets and measurements when a new worker has none."""
        for side, state in snapshot.items():
            previous = self._states.get(side, {})
            merged = state.copy()
            if state["positions"] is None and previous.get("positions") is not None:
                merged.update(positions=previous["positions"], sampled_at=previous["sampled_at"])
            if state["targets"] is None and previous.get("targets") is not None:
                merged.update(targets=previous["targets"], commanded_at=previous["commanded_at"])
            # Application identity always comes from the current worker.
            self._states[side] = merged

    def _refresh_health(self):
        """Reconcile freshness and worker holds while the parent lock is held."""
        with self._gate.get_lock():
            stamps = list(self._measurements)
            outages = self._gate[2]
        if outages != self._seen_outages:
            self._seen_outages = outages
            # Even an outage that recovered between supervisor ticks must
            # invalidate parent proposals and pending application identities.
            self._set_health(False)
        now = time.monotonic()
        fresh = self._attempt_ready and all(
            stamp > 0 and 0 <= now - stamp <= self.cfg.joint_state_timeout_s for stamp in stamps
        )
        fresh = (
            fresh
            and len(self._states) == 2
            and all(
                state["sampled_at"] is not None and 0 <= now - state["sampled_at"] <= self.cfg.joint_state_timeout_s
                for state in self._states.values()
            )
        )
        self._set_health(fresh and not self._closed and not self._stop.is_set() and self._error is None)
        if self._attempt_ready and self._status not in ("fault", "reconnecting", "closed"):
            self._status = "streaming" if self._healthy else "stale"
        return stamps

    def _supervise(self):
        delay = self.cfg.reconnect_interval_s
        ever_ready = False
        try:
            while not self._shutdown.is_set():
                self._start_worker()
                reason = None
                fatal = False
                while not self._shutdown.wait(0.005):
                    with self._lock:
                        try:
                            error = self._errors.get_nowait()
                        except queue.Empty:
                            error = None
                        if error is not None:
                            reason, fatal = error["message"], error["fatal"]
                            break
                        if self._process.exitcode is not None:
                            # A final write/identity fault must win over an exit
                            # observed just after the first mailbox check.
                            try:
                                error = self._errors.get_nowait()
                            except queue.Empty:
                                reason = f"OmniHand worker exited ({self._process.exitcode})"
                            else:
                                reason, fatal = error["message"], error["fatal"]
                            break
                        try:
                            snapshot = self._state_queue.get_nowait()
                        except queue.Empty:
                            snapshot = None
                        if snapshot is not None:
                            self._merge_snapshot(snapshot)
                        # A snapshot must belong to this process, with both
                        # measurements present, before publishing recovery.
                        now = time.monotonic()
                        complete = snapshot is not None and all(
                            snapshot[side]["sampled_at"] is not None
                            and 0 <= now - snapshot[side]["sampled_at"] <= self.cfg.joint_state_timeout_s
                            for side in ("left", "right")
                        )
                        if not self._attempt_ready and complete:
                            positions = {
                                name: value
                                for index, side in enumerate(("left", "right"))
                                for name, value in zip(
                                    ALL_HAND_NAMES[index * 19 : (index + 1) * 19],
                                    snapshot[side]["positions"],
                                    strict=True,
                                )
                            }
                            self._initial_targets = measured_command_baseline(positions)
                            self._attempt_ready = True
                        stamps = self._refresh_health()
                        now = time.monotonic()
                        if self._healthy:
                            ever_ready = True
                            self._status = "streaming"
                            self._started.set()
                            delay = self.cfg.reconnect_interval_s
                        else:
                            self._status = (
                                "stale" if self._attempt_ready else "reconnecting" if ever_ready else "starting"
                            )
                        reference = min(stamps) if all(stamp > 0 for stamp in stamps) else self._attempt_started
                        limit = self.cfg.reconnect_timeout_s if self._attempt_ready else self.cfg.startup_timeout_s
                        if now - reference >= limit:
                            reason = "hand measurements did not recover before the communication deadline"
                            break
                with self._lock:
                    self._set_health(False)
                    self._attempt_ready = False
                    self._pending = None
                    self._status = "reconnecting"
                    self._last_error = reason
                    if hasattr(self, "_gate"):
                        with self._gate.get_lock():
                            self._gate[0] += 1
                            self._gate[1] = 0
                self._reap_worker()
                if self._shutdown.is_set():
                    break
                if fatal or not self._can_reconnect:
                    with self._lock:
                        self._error = reason
                        self._status = "fault"
                    logger.error("OmniHand hardware fault: %s", reason)
                    self._started.set()
                    break
                logger.warning("OmniHand hardware reconnect in %.1fs: %s", delay, reason)
                if self._shutdown.wait(delay):
                    break
                delay = min(delay * 2, 5.0)
        except Exception as exc:
            with self._lock:
                if hasattr(self, "_gate"):
                    self._set_health(False)
                    self._stop.set()
                self._error = f"OmniHand supervisor failed: {exc}"
                self._status = "fault"
            self._started.set()
        finally:
            try:
                self._reap_worker()
            except Exception:
                logger.exception("OmniHand worker cleanup failed")
            if self._shutdown.is_set():
                with self._lock:
                    self._status = "closed"

    def _poll(self):
        """Raise latched faults; the supervisor exclusively owns worker I/O."""
        if self._error is not None:
            raise RuntimeError(self._error)

    def set_takeover_enabled(self, enabled):
        with self._lock:
            self._requested = bool(enabled)
            if not hasattr(self, "_gate"):
                return
            with self._gate.get_lock():
                self._gate[0] += 1
                self._gate[1] = int(
                    enabled
                    and not self.read_only
                    and not self._closed
                    and not self._stop.is_set()
                    and self._healthy
                    and self._error is None
                )
            self._pending = None

    def submit(self, positions, identity, deadline, *, replace_pending=False):
        with self._lock:
            return self._submit(positions, identity, deadline, replace_pending=replace_pending)

    def _submit(self, positions, identity, deadline, *, replace_pending=False):
        self._poll()
        self._refresh_health()
        if not self._healthy:
            # The pipeline treats stale data as recoverable and preserves XR
            # engagement; RuntimeError is reserved for latched/authorization faults.
            raise ValueError("finger measurements are stale")
        angles = np.asarray([positions[n] for n in OMNIHAND_JOINT_NAMES], dtype=float)
        limits = np.concatenate((OMNIHAND_LEFT_LIMITS, OMNIHAND_RIGHT_LIMITS))
        if not np.isfinite(angles).all() or np.any(angles < limits[:, 0]) or np.any(angles > limits[:, 1]):
            raise ValueError("checked finger commands exceed SDK limits")
        with self._gate.get_lock():
            generation, enabled = self._gate[:2]
            if not enabled or not np.isfinite(deadline) or time.monotonic() > deadline:
                raise RuntimeError("OmniHand command is unauthorized or expired")
            if self._pending is not None and not replace_pending:
                raise RuntimeError("OmniHand command is already pending")
            if replace_pending:
                self._gate[0] += 1
                generation = self._gate[0]
        self._pending = identity
        submitted_at = time.monotonic()
        try:
            self._commands.put_nowait((generation, deadline, identity, angles.tolist(), submitted_at))
        except Exception:
            self.set_takeover_enabled(False)
            raise

    def set_joint_commands(self, left_joint_positions, right_joint_positions, source_timestamp_ns=None, frame_id=None):
        with self._lock:
            self._poll()
            self._refresh_health()
            return self._set_joint_commands(left_joint_positions, right_joint_positions, source_timestamp_ns, frame_id)

    def _set_joint_commands(self, left_joint_positions, right_joint_positions, source_timestamp_ns=None, frame_id=None):
        """Accept a latest-only dex-teleop frame; XR uses strict ``submit`` instead."""
        arrays = [np.asarray(values, dtype=float) for values in (left_joint_positions, right_joint_positions)]
        if any(values.shape != (12,) or not np.isfinite(values).all() for values in arrays):
            raise ValueError("OmniHand commands must each contain 12 finite joint angles")
        positions = np.concatenate(
            [
                np.clip(values, limits[:, 0], limits[:, 1])
                for values, limits in zip(arrays, (OMNIHAND_LEFT_LIMITS, OMNIHAND_RIGHT_LIMITS), strict=True)
            ]
        )
        with self._gate.get_lock():
            enabled = self._gate[1]
        if enabled:
            self._sequence += 1
            timestamp_ns = time.time_ns() if source_timestamp_ns is None else int(source_timestamp_ns)
            identity = (self._sequence, None if frame_id is None else int(frame_id), timestamp_ns)
            self.submit(
                dict(zip(OMNIHAND_JOINT_NAMES, positions.tolist(), strict=True)),
                identity,
                time.monotonic() + self.cfg.command_timeout_s,
                replace_pending=True,
            )
        return positions.astype(np.float32)

    def get_data(self):
        with self._lock:
            return self._get_data()

    def _get_data(self):
        self._poll()
        measured, stamps, commands = {}, {}, {}
        for index, side in enumerate(("left", "right")):
            if side not in self._states or self._states[side]["positions"] is None:
                continue
            state = self._states[side]
            names = ALL_HAND_NAMES[index * 19 : (index + 1) * 19]
            measured.update(zip(names, state["positions"], strict=True))
            stamps.update(dict.fromkeys(names, state["sampled_at"]))
            active = OMNIHAND_JOINT_NAMES[index * 12 : (index + 1) * 12]
            commands.update(zip(active, state["targets"] or state["positions"][:12], strict=True))
        if self._initial_targets is not None:
            for index, side in enumerate(("left", "right")):
                if side in self._states and self._states[side]["targets"] is None:
                    for name in OMNIHAND_JOINT_NAMES[index * 12 : (index + 1) * 12]:
                        commands[name] = self._initial_targets[name]
        self._refresh_health()
        fresh = self._healthy
        with self._gate.get_lock():
            generation, enabled = self._gate[:2]
            enabled = bool(enabled)
        applied = None
        if len(self._states) == 2:
            left, right = (self._states[side]["applied"] for side in ("left", "right"))
            if (
                left is not None
                and left == right
                and all(state["generation"] == generation for state in self._states.values())
            ):
                applied = left
        now = time.monotonic()
        command_ages = {
            side: None if state["commanded_at"] is None else now - state["commanded_at"]
            for side, state in self._states.items()
        }
        side_fresh = {
            side: enabled
            and age is not None
            and 0 <= age <= self.cfg.command_timeout_s
            and self._states[side]["generation"] == generation
            for side, age in command_ages.items()
        }
        source_identity = applied if applied is not None and len(applied) == 3 else (None, None, None)
        return {
            "connection_status": self._status,
            "connection_generation": self._connection_generation,
            "stream_generation": self._stream_generation,
            "last_error": self._last_error,
            "hand_health": {
                side: {
                    "measurement_age_s": None if state["sampled_at"] is None else now - state["sampled_at"],
                    "consecutive_failures": state.get("consecutive_failures", 0),
                    "last_read_error": state.get("last_read_error"),
                }
                for side, state in self._states.items()
            },
            "enabled": enabled,
            "left_fresh": side_fresh.get("left", False),
            "right_fresh": side_fresh.get("right", False),
            "age_s": max((age for age in command_ages.values() if age is not None), default=None),
            "applied_frame_id": source_identity[1],
            "applied_source_timestamp_ns": source_identity[2],
            "positions": measured,
            "sample_times": stamps,
            "targets": commands,
            "applied": applied,
            "joint_names": list(OMNIHAND_JOINT_NAMES),
            "joint_positions": np.array([measured.get(n, 0) for n in OMNIHAND_JOINT_NAMES], dtype=np.float32),
            "joint_position_commands": np.array([commands.get(n, 0) for n in OMNIHAND_JOINT_NAMES], dtype=np.float32),
            "fresh": fresh and applied is not None and all(side_fresh.get(side, False) for side in ("left", "right")),
            "joint_state_fresh": fresh,
        }

    def acknowledge(self):
        with self._lock:
            self._pending = None

    @property
    def takeover_requested(self):
        return self._requested

    def invalidate_commands(self):
        """Revoke work without changing the operator's requested takeover."""
        with self._lock:
            self.set_takeover_enabled(self._requested)

    def reset(self):
        self.set_takeover_enabled(False)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.set_takeover_enabled(False)
            self._shutdown.set()
            self._status = "closed"
        self._supervisor.join(timeout=3)
        if self._supervisor.is_alive():
            raise RuntimeError("OmniHand supervisor did not stop")
