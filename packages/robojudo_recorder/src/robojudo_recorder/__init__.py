from .config import RecorderConfig, load_config

__all__ = ["RecorderConfig", "RecorderService", "load_config"]


def __getattr__(name: str):
    if name == "RecorderService":
        from .service import RecorderService

        return RecorderService
    raise AttributeError(name)
