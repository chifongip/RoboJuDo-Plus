import argparse
import logging
import signal

from .config import load_config
from .service import RecorderService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture timestamped RoboJuDo control and camera data into raw episodes"
    )
    parser.add_argument("--config", required=True, help="Recorder YAML configuration")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = RecorderService(load_config(args.config))

    def stop(_signum, _frame):
        service.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    service.run()


if __name__ == "__main__":
    main()
