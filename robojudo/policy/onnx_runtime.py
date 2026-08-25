"""Shared ONNX Runtime session setup for deployment policies."""

from collections.abc import Sequence

import onnxruntime as ort

from robojudo.policy.policy_cfgs import PolicyCfg


def build_onnx_session_options(cfg_policy: PolicyCfg) -> ort.SessionOptions:
    """Build thread-limited ONNX Runtime options for a policy session."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg_policy.onnx_intra_op_num_threads
    options.inter_op_num_threads = cfg_policy.onnx_inter_op_num_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    allow_spinning = "1" if cfg_policy.onnx_allow_spinning else "0"
    options.add_session_config_entry("session.intra_op.allow_spinning", allow_spinning)
    options.add_session_config_entry("session.inter_op.allow_spinning", allow_spinning)
    return options


def create_onnx_session(
    policy_file: str,
    cfg_policy: PolicyCfg,
    providers: Sequence[str] | None = None,
) -> ort.InferenceSession:
    """Create an ONNX session with the policy's CPU thread settings."""
    session_options = build_onnx_session_options(cfg_policy)
    if providers is None:
        return ort.InferenceSession(policy_file, sess_options=session_options)
    return ort.InferenceSession(policy_file, sess_options=session_options, providers=providers)
