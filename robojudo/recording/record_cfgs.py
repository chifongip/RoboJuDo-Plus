from pydantic import model_validator

from robojudo.config import Config


class RecordCfg(Config):
    """Publish synchronized control samples to an external recorder service."""

    enabled: bool = False
    endpoint: str = "tcp://*:8560"
    task: str = "upper body teleoperation"
    send_hwm: int = 256
    lifecycle_timeout_ms: int = 0

    @model_validator(mode="after")
    def validate_recording(self):
        if not self.endpoint.startswith("tcp://"):
            raise ValueError("record endpoint must use tcp://")
        if not self.task.strip():
            raise ValueError("record task must not be empty")
        if self.send_hwm <= 0:
            raise ValueError("record send_hwm must be positive")
        if self.lifecycle_timeout_ms < 0:
            raise ValueError("record lifecycle_timeout_ms must be non-negative")
        return self
