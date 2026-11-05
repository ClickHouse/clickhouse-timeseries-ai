import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from model import TimesFMModel, load_engine


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
