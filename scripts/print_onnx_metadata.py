#!/usr/bin/env python3
"""Print standard and custom metadata embedded in an ONNX model."""

import argparse
import json
from pathlib import Path

import onnxruntime as ort

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPOSITORY_ROOT / "assets/models/x2/beyondmimic/Solo_dance_default_pose.onnx"


def _model_io(value: ort.NodeArg) -> dict[str, object]:
    return {
        "name": value.name,
        "shape": value.shape,
        "type": value.type,
    }


def read_metadata(model_path: Path) -> dict[str, object]:
    """Load metadata and the inference contract without running the model."""
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    model_meta = session.get_modelmeta()
    return {
        "model": model_path.as_posix(),
        "producer_name": model_meta.producer_name,
        "graph_name": model_meta.graph_name,
        "domain": model_meta.domain,
        "description": model_meta.description,
        "version": model_meta.version,
        "custom_metadata": dict(sorted(model_meta.custom_metadata_map.items())),
        "inputs": [_model_io(value) for value in session.get_inputs()],
        "outputs": [_model_io(value) for value in session.get_outputs()],
    }


def _print_text(metadata: dict[str, object]) -> None:
    print(f"Model: {metadata['model']}")
    print(f"Producer: {metadata['producer_name']}")
    print(f"Graph: {metadata['graph_name']}")
    print(f"Domain: {metadata['domain']}")
    print(f"Version: {metadata['version']}")
    print(f"Description: {metadata['description']!r}")

    print("\nCustom metadata:")
    custom_metadata = metadata["custom_metadata"]
    assert isinstance(custom_metadata, dict)
    for key, value in custom_metadata.items():
        print(f"  {key}: {value}")

    print("\nInputs:")
    for value in metadata["inputs"]:
        assert isinstance(value, dict)
        print(f"  {value['name']}: {value['shape']} ({value['type']})")

    print("\nOutputs:")
    for value in metadata["outputs"]:
        assert isinstance(value, dict)
        print(f"  {value['name']}: {value['shape']} ({value['type']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX model to inspect (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    metadata = read_metadata(model_path)
    if args.json:
        print(json.dumps(metadata, indent=2))
    else:
        _print_text(metadata)


if __name__ == "__main__":
    main()
