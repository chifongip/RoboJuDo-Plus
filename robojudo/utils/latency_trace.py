"""Bounded, opt-in performance telemetry; never used for command authorization.

Keep the RoboJuDo and XR package copies identical. Export happens explicitly or
at shutdown, outside the timed control path. PUB acceptance does not mean delivery.
"""

import json
import math
import os
import time
from pathlib import Path

STAMP_NAMES = (
    "received",
    "hand_submitted",
    "arm_submission_started",
    "arm_submission_completed",
    "acknowledged",
    "worker_completion_ready",
    "parent_snapshot_received",
    "feedback_built",
    "first_feedback_built",
    "left_submitted",
    "left_write_started",
    "left_write_completed",
    "left_read_started",
    "left_read_completed",
    "right_submitted",
    "right_write_started",
    "right_write_completed",
    "right_read_started",
    "right_read_completed",
)


def command_stamps(values):
    if not isinstance(values, dict):
        return {}
    result = {}
    for name in STAMP_NAMES:
        value = values.get(name)
        if type(value) not in (int, float):
            continue
        try:
            value = float(value)
        except OverflowError:
            continue
        if math.isfinite(value):
            result[name] = value
    return result


def identity(message, *, applied=False):
    result = {k: message.get(k) for k in ("session_id", "generation", "state_id", "command_id")}
    for key in ("bridge_id", "frame_id"):
        result[key] = message.get(f"applied_{key}" if applied else key)
    # Trace only bounded scalar identities, never raw command values.
    return {
        k: v if (type(v) is int and 0 <= v < 2**64) or (isinstance(v, str) and len(v) <= 64) else None
        for k, v in result.items()
    }


def clock_domain():
    try:
        return {
            "clock": "monotonic",
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "time_namespace": os.readlink("/proc/self/ns/time"),
        }
    except OSError:
        return None


class LatencyTrace:
    MAX_RECORDS = 8192

    def __init__(self, path=None, duration_s=30.0, delay_s=0.0):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("trace duration must be finite and positive")
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("trace delay must be finite and nonnegative")
        self.delay_s = delay_s
        self.path = path
        self.duration_s = duration_s
        self.domain = clock_domain() if path is not None else None
        self.reset()

    @property
    def enabled(self):
        return self.path is not None

    def reset(self):
        self.started_at = time.monotonic() + self.delay_s if self.enabled else None
        self.records = []
        self.overflow = False

    def record(self, event, *, at=None, **fields):
        if not self.enabled:
            return
        at = time.monotonic() if at is None else at
        if not self.started_at <= at <= self.started_at + self.duration_s:
            return
        if len(self.records) >= self.MAX_RECORDS:
            self.overflow = True
            return
        self.records.append({"event": event, "at": at, **fields})

    def snapshot(self):
        now = time.monotonic()
        return {
            "version": 1,
            "clock_domain": self.domain,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "window_complete": self.enabled and now >= self.started_at + self.duration_s,
            "overflow": self.overflow,
            "records": list(self.records),
        }

    def export(self):
        if not self.enabled:
            return
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.snapshot(), allow_nan=False))
        temporary.replace(path)
