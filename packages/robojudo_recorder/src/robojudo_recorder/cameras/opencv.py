from robojudo_recorder.config import CameraConfig

from . import register_camera
from .threaded import ThreadedCameraSource


@register_camera("opencv")
class OpenCVCameraSource(ThreadedCameraSource):
    def __init__(self, cfg: CameraConfig):
        self.device = cfg.options.get("device", 0)
        self.width = int(cfg.options.get("width", 640))
        self.height = int(cfg.options.get("height", 480))
        self.fps = int(cfg.options.get("fps", 30))
        self._capture_device = None
        super().__init__((self.height, self.width, 3))

    def _open(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV camera requires robojudo-recorder[opencv]") from exc
        self._cv2 = cv2
        self._capture_device = cv2.VideoCapture(self.device)
        self._capture_device.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture_device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture_device.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture_device.isOpened():
            raise ConnectionError(f"failed to open camera {self.device!r}")

    def _capture(self):
        ok, image = self._capture_device.read()
        if not ok:
            raise RuntimeError("OpenCV camera read failed")
        return self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)

    def _close(self):
        if self._capture_device is not None:
            self._capture_device.release()
            self._capture_device = None
