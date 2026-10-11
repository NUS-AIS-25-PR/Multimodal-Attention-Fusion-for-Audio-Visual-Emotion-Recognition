"""CPU-only harness contracts; no dataset, downloaded model or GPU required."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_revision_efficiency.py"
spec = importlib.util.spec_from_file_location("efficiency", SCRIPT)
efficiency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efficiency)


class EfficiencyTests(unittest.TestCase):
    def test_input_shapes_and_modality_order(self):
        wave = torch.empty(1, 48000)
        video = torch.empty(1, 8, 3, 112, 112)
        self.assertEqual(efficiency.WAVEFORM_SHAPE, tuple(wave.shape))
        self.assertEqual(efficiency.VIDEO_SHAPE, tuple(video.shape))
        self.assertIs(efficiency.select_inputs("audio", wave, video)[0], wave)
        self.assertIs(efficiency.select_inputs("video", wave, video)[0], video)
        for method in ("gated", "xattn", "chumachenko_ia"):
            selected = efficiency.select_inputs(method, wave, video)
            self.assertIs(selected[0], video)
            self.assertIs(selected[1], wave)
        with self.assertRaises(ValueError):
            efficiency.select_inputs("concat", wave, video)

    def test_config_restoration_requires_all_keys_and_strict_state(self):
        seen = {}

        def builder(num_classes, fusion, use_wavlm, xattn_head, xattn_d_model,
                    smoke, pretrained_video, checkpoint_init):
            seen.update(locals())
            return nn.Linear(2, num_classes)

        config = {"num_classes": 8, "fusion": "xattn", "use_wavlm": True,
                  "xattn_head": "gated", "xattn_d_model": 96, "smoke": False}
        source = nn.Linear(2, 8)
        restored = efficiency.restore_model(config, {"config": config, "model": source.state_dict()}, builder)
        self.assertTrue(torch.equal(restored.weight, source.weight))
        self.assertEqual(seen["xattn_head"], "gated")
        self.assertEqual(seen["xattn_d_model"], 96)
        self.assertTrue(seen["checkpoint_init"])
        self.assertFalse(seen["pretrained_video"])
        with self.assertRaises(RuntimeError):
            efficiency.restore_model(config, {"config": config, "model": {"weight": source.weight}}, builder)
        with self.assertRaises(KeyError):
            efficiency.model_kwargs({k: v for k, v in config.items() if k != "xattn_d_model"}, builder)
        with self.assertRaises(ValueError):
            efficiency.restore_model(config, {"config": {**config, "smoke": True}, "model": source.state_dict()}, builder)

    def test_parameter_count_includes_frozen_unused_heads_without_alias_duplication(self):
        model = nn.Module()
        model.encoder = nn.Linear(3, 4)  # 12 + 4
        model.encoder.requires_grad_(False)
        model.unused_head = nn.Linear(4, 8)  # 32 + 8
        model.alias = model.encoder
        self.assertEqual(efficiency.total_parameters(model), 56)

    def test_statistics_use_sample_std(self):
        result = efficiency.latency_statistics([1., 2., 3.])
        self.assertEqual(result["mean_ms"], 2.)
        self.assertEqual(result["median_ms"], 2.)
        self.assertEqual(result["sample_stdev_ms"], 1.)

    def test_insufficient_timing_is_rejected_before_gpu_use(self):
        for settings in ((19, 100, 3), (20, 99, 3), (20, 100, 2)):
            with self.assertRaises(ValueError):
                efficiency.measure_latency(nn.Identity(), (), *settings)

    def test_cuda_samples_and_unadjusted_wall_clock_stay_separate(self):
        model = nn.Identity()
        event = mock.Mock()
        event.elapsed_time.return_value = 2.0
        with mock.patch("torch.cuda.Event", return_value=event), \
                mock.patch("torch.cuda.synchronize"), \
                mock.patch.object(efficiency.time, "clock_gettime_ns", side_effect=[0, 250000000] * 3), \
                mock.patch.object(efficiency.time, "perf_counter", side_effect=[0., .2] * 3), \
                mock.patch.object(efficiency.subprocess, "check_output", return_value="GPU status"), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            result = efficiency.measure_latency(model, (torch.zeros(1),), 20, 100, 3)
        self.assertFalse(model.training)
        self.assertEqual(result["combined"]["n"], 300)
        self.assertEqual(result["combined"]["mean_ms"], 2.)
        self.assertEqual(result["wall_repeat_average_statistics"]["mean_ms"], 2.5)
        self.assertEqual(result["repeats"][0]["adjusted_perf_counter_loop_ms_diagnostic_only"], 200.)

    def test_profiler_failure_is_na_with_exact_error_class(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("torchinfo.summary", side_effect=ValueError("unsupported op")):
            result = efficiency.operation_profile(nn.Identity(), (), Path(tmp), "audio")
            self.assertIsNone(result["total_mult_adds"])
            self.assertIsNone(result["estimated_gmacs"])
            self.assertEqual(result["status"], "N/A")
            self.assertEqual(result["error_classes"], ["builtins.ValueError"])
            self.assertIn("unsupported op", (Path(tmp) / "audio_profiler_error.txt").read_text())

    def test_environment_mismatch_stops_without_installing(self):
        with mock.patch.object(efficiency, "packages", return_value={"mediapipe": "0.10.32"}):
            with self.assertRaisesRegex(RuntimeError, "STOP: package snapshot mismatch"):
                efficiency.require_environment({"mediapipe": "0.10.21"})

    def test_integrity_snapshot_detects_byte_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics.json").write_text("original")
            before = efficiency.artifact_snapshot(root)
            (root / "metrics.json").write_text("modified")
            after = efficiency.artifact_snapshot(root)
            self.assertNotEqual(before["metrics.json"]["sha256"], after["metrics.json"]["sha256"])


if __name__ == "__main__":
    unittest.main()
