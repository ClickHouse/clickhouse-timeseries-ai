import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from engine import Engine
from model import TimesFMModel, TimestralModel, load_engine


class ModelTests(unittest.TestCase):
    def test_invalid_pin_fails_before_loading(self):
        for revision in ("", "main", "a" * 39):
            with self.subTest(revision=revision), patch.dict(os.environ, {"TIMESFM_REVISION": revision}):
                with self.assertRaises(ValueError):
                    load_engine()

    def test_cpu_default_and_pinned_checkpoint(self):
        with patch.dict(os.environ, {"TIMESFM_REVISION": "a" * 40}, clear=True), patch(
            "timesfm3.TimesFM3Evaluator"
        ) as evaluator:
            engine = load_engine()
        config = evaluator.call_args.args[0]
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.revision, "a" * 40)
        self.assertEqual(engine.revision, "a" * 40)

    def test_timestral_invalid_pin_fails_before_loading(self):
        for revision in ("", "main", "a" * 39, "A" * 40):
            with self.subTest(revision=revision), patch.dict(
                os.environ, {"TIMESTRAL_REVISION": revision}, clear=True
            ), patch("timestral.TimestralForecaster.from_pretrained") as loader:
                with self.assertRaisesRegex(ValueError, "TIMESTRAL_REVISION"):
                    load_engine("timestral-209m")
                loader.assert_not_called()

    def test_timestral_cpu_default_and_pinned_checkpoint(self):
        with patch.dict(os.environ, {"TIMESTRAL_REVISION": "b" * 40}, clear=True), patch(
            "timestral.TimestralForecaster.from_pretrained"
        ) as loader, patch("timesfm3.TimesFM3Evaluator") as timesfm:
            engine = load_engine("timestral-209m")
        loader.assert_called_once_with("mistralai/Timestral-209M", revision="b" * 40, device="cpu")
        timesfm.assert_not_called()
        self.assertIsInstance(engine.model, TimestralModel)
        self.assertEqual(engine.revision, "b" * 40)
        self.assertEqual(engine.model_name, "timestral-209m")

    def test_selected_device_is_forwarded(self):
        with patch.dict(os.environ, {"TIMESTRAL_REVISION": "b" * 40,
                                     "TIMESTRAL_DEVICE": "cuda"}, clear=True), patch(
            "timestral.TimestralForecaster.from_pretrained"
        ) as loader:
            load_engine("timestral-209m")
        self.assertEqual(loader.call_args.kwargs["device"], "cuda")

    def test_unknown_model_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            load_engine("other")

    def test_quantile_mapping_and_covariates(self):
        for channels in (1, 2):
            quantiles = np.arange(channels * 3 * 9).reshape(channels, 3, 9)
            forecaster = Mock()
            forecaster.predict_batch.return_value = [SimpleNamespace(
                quantiles=quantiles[0] if channels == 1 else quantiles)]
            item = {"targets": [[-1.0] * 8] * channels,
                    "past_covariates": [[2.0] * 8], "future_covariates": [[3.0] * 11]}
            result = TimesFMModel(forecaster).predict(item, 3)
            for key, index in (("expected", 4), ("lower_bound", 0), ("upper_bound", 8)):
                self.assertEqual(result[key], quantiles[:, :, index].tolist())
            self.assertEqual(result["quantile_levels"], [i / 10 for i in range(1, 10)])
            self.assertEqual(result["quantiles"], quantiles.tolist())
            kwargs = forecaster.predict_batch.call_args.kwargs
            np.testing.assert_array_equal(kwargs["contexts"][0], item["targets"])
            np.testing.assert_array_equal(kwargs["past_only_covariates"][0], item["past_covariates"])
            np.testing.assert_array_equal(kwargs["past_future_covariates"][0], item["future_covariates"])
            self.assertFalse(kwargs["make_positive"])
            self.assertFalse(kwargs["use_symmetric_averaging"])

    def test_timestral_quantile_mapping_and_covariates(self):
        import torch

        levels = (0.01, 0.05, *[i / 20 for i in range(2, 19)], 0.95, 0.99)
        for channels in (1, 2):
            with self.subTest(channels=channels):
                quantiles = torch.arange(channels * 3 * 21).reshape(1, channels, 3, 21).float()
                forecaster = Mock()
                forecaster.forecast.return_value = SimpleNamespace(
                    quantiles=quantiles, quantile_levels=levels)
                item = {"targets": [[-1.0] * 8] * channels,
                        "past_covariates": [[2.0] * 8], "future_covariates": [[3.0] * 11]}
                result = TimestralModel(forecaster).predict(item, 3)
                for key, level in (("expected", 0.5), ("lower_bound", 0.1), ("upper_bound", 0.9)):
                    self.assertEqual(result[key], quantiles[0, :, :, levels.index(level)].tolist())
                self.assertEqual(result["quantile_levels"], list(levels))
                self.assertEqual(result["quantiles"], quantiles[0].tolist())
                kwargs = forecaster.forecast.call_args.kwargs
                for key, source in (("context", "targets"), ("past_covariates", "past_covariates"),
                                    ("known_covariates", "future_covariates")):
                    np.testing.assert_array_equal(kwargs[key], [item[source]])
                    self.assertEqual(kwargs[key].dtype, np.float32)
                self.assertEqual(kwargs["prediction_length"], 3)
                self.assertFalse(kwargs["clip_nonnegative"])
                self.assertFalse(kwargs["symmetric_averaging"])

    def test_timestral_without_covariates_and_causal_anomalies(self):
        import torch

        def forecast(**kwargs):
            channels = kwargs["context"].shape[1]
            horizon = kwargs["prediction_length"]
            return SimpleNamespace(
                quantiles=torch.tensor([9.0, 10.0, 11.0]).expand(1, channels, horizon, 3),
                quantile_levels=(0.1, 0.5, 0.9))

        forecaster = Mock()
        forecaster.forecast.side_effect = forecast
        engine = Engine(TimestralModel(forecaster), "b" * 40, "timestral-209m")
        body = {"model": "timestral-209m", "horizon": 3,
                "inputs": [{"targets": [[10.0] * 36]}]}
        result = engine.run(body, "forecast")
        self.assertEqual(result["model"], "timestral-209m")
        self.assertIsNone(forecaster.forecast.call_args.kwargs["past_covariates"])
        self.assertIsNone(forecaster.forecast.call_args.kwargs["known_covariates"])
        forecaster.reset_mock()
        item = body["inputs"][0]
        item["targets"][0][34] = 100.0
        item["past_covariates"] = [list(range(36))]
        item["future_covariates"] = [list(range(36))]
        result = engine.run(body, "anomalies")
        calls = forecaster.forecast.call_args_list
        self.assertEqual([c.kwargs["context"].shape[-1] for c in calls], [32, 35])
        self.assertEqual([c.kwargs["past_covariates"].shape[-1] for c in calls], [32, 35])
        self.assertEqual([c.kwargs["known_covariates"].shape[-1] for c in calls], [35, 36])
        self.assertNotIn(100, calls[0].kwargs["context"])
        self.assertEqual(result["predictions"][0]["anomaly_score"][0][34], 90.0)
        self.assertEqual(result["interval_quantiles"], [0.1, 0.9])
        body["model"] = "timesfm-3.0"
        with self.assertRaisesRegex(ValueError, "model must be timestral-209m"):
            engine.run(body, "forecast")

    def test_real_timestral_inference_with_synthetic_weights(self):
        """Exercise the pinned library, not mocks, without pretrained downloads."""
        import torch
        from timestral import TimestralConfig, TimestralForecaster
        from timestral import TimestralModel as Network

        levels = (0.01, 0.05, *[i / 20 for i in range(2, 19)], 0.95, 0.99)
        config = TimestralConfig(
            patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
            block_schedule=("time", "time_variate"), quantile_knots=levels,
            block_size=4, context_length=256,
        )
        with torch.random.fork_rng():
            torch.manual_seed(0)
            network = Network(config)
            for parameter in network.parameters():
                torch.nn.init.normal_(parameter, std=0.05)
        engine = Engine(TimestralModel(TimestralForecaster(network.eval())), "b" * 40, "timestral-209m")
        for channels in (1, 2):
            item = {"targets": [[float(i % 7 - 3) for i in range(40)]] * channels,
                    "past_covariates": [[float(i % 3) for i in range(40)]],
                    "future_covariates": [[float(i % 2) for i in range(44)]]}
            body = {"model": "timestral-209m", "horizon": 4, "inputs": [item]}
            with self.subTest(channels=channels):
                result = engine.run(body, "forecast")["predictions"][0]
                self.assertEqual(result["quantile_levels"], list(levels))
                self.assertEqual(np.asarray(result["quantiles"]).shape, (channels, 4, 21))
                item["future_covariates"][0] = item["future_covariates"][0][:40]
                result = engine.run(body, "anomalies")["predictions"][0]
                self.assertEqual(len(result["expected"]), channels)
                self.assertEqual(result["expected"][0][:32], [None] * 32)
                self.assertTrue(all(np.isfinite(x) for x in result["anomaly_score"][0][32:]))
