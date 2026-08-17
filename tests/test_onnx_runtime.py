import unittest
from types import SimpleNamespace
from unittest.mock import patch

import onnxruntime as ort

from robojudo.policy.onnx_runtime import build_onnx_session_options, create_onnx_session


class TestOnnxRuntime(unittest.TestCase):
    @staticmethod
    def _cfg(**overrides):
        defaults = {
            "onnx_intra_op_num_threads": 1,
            "onnx_inter_op_num_threads": 1,
            "onnx_allow_spinning": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_build_session_options_limits_cpu_threads_by_default(self):
        options = build_onnx_session_options(self._cfg())

        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)

    def test_create_session_applies_configured_threads_and_providers(self):
        cfg = self._cfg(onnx_intra_op_num_threads=2, onnx_inter_op_num_threads=3, onnx_allow_spinning=True)
        with patch("robojudo.policy.onnx_runtime.ort.InferenceSession", return_value="session") as session:
            result = create_onnx_session("policy.onnx", cfg, providers=["CPUExecutionProvider"])

        self.assertEqual(result, "session")
        self.assertEqual(session.call_args.args[0], "policy.onnx")
        self.assertEqual(session.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        options = session.call_args.kwargs["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 3)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)

    def test_create_session_keeps_default_provider_selection(self):
        with patch("robojudo.policy.onnx_runtime.ort.InferenceSession", return_value="session") as session:
            create_onnx_session("policy.onnx", self._cfg())

        self.assertEqual(session.call_args.args[0], "policy.onnx")
        self.assertNotIn("providers", session.call_args.kwargs)
