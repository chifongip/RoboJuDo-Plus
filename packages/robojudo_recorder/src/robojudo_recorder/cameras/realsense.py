import logging

from robojudo_recorder.config import CameraConfig

from . import register_camera
from .threaded import ThreadedCameraSource

logger = logging.getLogger(__name__)


@register_camera("realsense")
class RealSenseCameraSource(ThreadedCameraSource):
    def __init__(self, cfg: CameraConfig):
        self.serial_number = str(cfg.options.get("serial_number", ""))
        self.width = int(cfg.options.get("width", 640))
        self.height = int(cfg.options.get("height", 480))
        self.fps = int(cfg.options.get("fps", 30))
        self._pipeline = None
        self._consecutive_timeouts = 0
        super().__init__((self.height, self.width, 3))

    def _open(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense camera support is not installed. Run `python scripts/install_realsense.py` "
                "from the RoboJuDo repository (required on Jetson), or install robojudo-recorder[realsense]."
            ) from exc
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self._pipeline.start(config)

    def _capture(self):
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as exc:
            if "Frame didn't arrive within" not in str(exc):
                raise
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= 5:
                serial = self.serial_number or "auto-selected"
                raise RuntimeError(
                    f"RealSense {serial} timed out {self._consecutive_timeouts} consecutive times"
                ) from exc
            if self._consecutive_timeouts == 1:
                logger.warning(
                    "RealSense frame timeout; retrying camera serial=%s",
                    self.serial_number or "auto-selected",
                )
            return None
        recovered = self._consecutive_timeouts
        self._consecutive_timeouts = 0
        if recovered:
            logger.info(
                "RealSense stream recovered after %d timeout(s): serial=%s",
                recovered,
                self.serial_number or "auto-selected",
            )
        color = frames.get_color_frame()
        if not color:
            return None
        import numpy as np

        return np.asanyarray(color.get_data())

    def _close(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
