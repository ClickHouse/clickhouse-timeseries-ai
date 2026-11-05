"""Forecast distribution contract and unchanged anomaly-score semantics."""

import copy
import unittest
from unittest.mock import patch

from engine import Engine, check_prediction
from test_engine import RecordingModel


class QuantileTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.engine = Engine(self.model, "a" * 40)
        self.body = {"model": "timesfm-3.0", "horizon": 4,
                     "inputs": [{"targets": [[10.0] * 40]}]}

    def test_forecast_exposes_all_levels_without_extra_model_calls(self):
        result = self.engine.run(self.body, "forecast")
        point = result["predictions"][0]
        self.assertEqual(point["quantile_levels"], [i / 10 for i in range(1, 10)])
        self.assertEqual(len(point["quantiles"]), 1)
        self.assertEqual(len(point["quantiles"][0]), 4)
        self.assertEqual(len(point["quantiles"][0][0]), 9)
        self.assertEqual(result["usage"]["model_calls"], 1)
        self.assertEqual(result["usage"]["output_values"], 4)
        self.assertEqual(len(self.model.calls), 1)

    def test_invalid_forecast_quantiles_fail_closed(self):
        original = self.model.predict(self.body["inputs"][0], 4)
        mutations = [
            lambda p: p.pop("quantiles"),
            lambda p: p.update(quantile_levels=[]),
            lambda p: p["quantile_levels"].__setitem__(0, True),
            lambda p: p["quantile_levels"].__setitem__(0, float("nan")),
            lambda p: p["quantile_levels"].__setitem__(0, 0),
            lambda p: p["quantile_levels"].__setitem__(1, 0.1),
            lambda p: p["quantile_levels"].__setitem__(0, 0.11),
            lambda p: p["quantiles"].append(copy.deepcopy(p["quantiles"][0])),
            lambda p: p["quantiles"][0].pop(),
            lambda p: p["quantiles"][0][0].pop(),
            lambda p: p["quantiles"][0][0].__setitem__(1, float("inf")),
            lambda p: p["quantiles"][0][0].__setitem__(1, 100),
            lambda p: p["quantiles"][0][0].__setitem__(4, 10.01),
        ]
        for mutation in mutations:
            bad = copy.deepcopy(original)
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                check_prediction(bad, 1, 4, require_quantiles=True)

    def test_anomaly_scoring_uses_only_fixed_reference_bounds(self):
        self.body["inputs"][0]["targets"][0][35] = 100
        baseline = self.engine.run(self.body, "anomalies")
        predict = self.model.predict

        def different_middle_quantiles(item, horizon):
            result = predict(item, horizon)
            for channel in result["quantiles"]:
                for values in channel:
                    values[1:4] = [values[0]] * 3
                    values[5:8] = [values[8]] * 3
            return result

        with patch.object(self.model, "predict", side_effect=different_middle_quantiles):
            result = self.engine.run(self.body, "anomalies")
        self.assertEqual(result, baseline)
        self.assertEqual(result["interval_quantiles"], [0.1, 0.9])
        self.assertNotIn("quantiles", result["predictions"][0])
        self.assertEqual(result["predictions"][0]["anomaly_score"][0][35], 90)


if __name__ == "__main__":
    unittest.main()
