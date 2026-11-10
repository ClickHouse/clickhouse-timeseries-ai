import copy
import json
import unittest
from unittest.mock import patch

from engine import Engine, MODEL
from udf import evaluate


class RecordingModel:
    def __init__(self):
        self.calls = []

    def predict(self, item, horizon):
        self.calls.append(copy.deepcopy(item))
        centers = [c[-1] for c in item["targets"]]
        return {
            "expected": [[x] * horizon for x in centers],
            "lower_bound": [[x - 1] * horizon for x in centers],
            "upper_bound": [[x + 1] * horizon for x in centers],
            "quantile_levels": [i / 10 for i in range(1, 10)],
            "quantiles": [[[x + (i - 5) / 4 for i in range(1, 10)] for _ in range(horizon)] for x in centers],
        }


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.engine = Engine(self.model, "a" * 40)

    def body(self, size=40):
        return {"model": MODEL, "horizon": 4, "inputs": [{"targets": [[10.0] * size]}]}

    def test_forecast_multivariate_and_negative_values(self):
        body = self.body()
        body["inputs"][0] = {"targets": [[-10.0] * 40, [20.0] * 40], "future_covariates": [[1] * 44]}
        result = self.engine.run(body, "forecast")
        self.assertEqual(result["predictions"][0]["expected"], [[-10.0] * 4, [20.0] * 4])
        self.assertEqual(result["usage"]["model_calls"], 1)

    def test_historical_windows_are_causal(self):
        body = self.body()
        body["inputs"][0]["targets"][0][35] = 100.0
        body["inputs"][0]["past_covariates"] = [list(range(40))]
        body["inputs"][0]["future_covariates"] = [list(range(40))]
        result = self.engine.run(body, "anomalies")["predictions"][0]
        self.assertEqual([len(c["targets"][0]) for c in self.model.calls], [32, 36])
        self.assertNotIn(100, self.model.calls[0]["targets"][0])
        self.assertEqual(len(self.model.calls[0]["past_covariates"][0]), 32)
        self.assertEqual(len(self.model.calls[0]["future_covariates"][0]), 36)
        self.assertTrue(result["is_anomaly"][0][35])
        self.assertEqual(result["anomaly_score"][0][35], 90)
        self.assertEqual(result["expected"][0][:32], [None] * 32)

    def test_trailing_window_and_partial_final_window(self):
        body = self.body()
        body["only_last_n"] = 5
        result = self.engine.run(body, "anomalies")["predictions"][0]
        self.assertEqual([len(c["targets"][0]) for c in self.model.calls], [35, 39])
        self.assertEqual(result["expected"][0][:35], [None] * 35)
        self.assertEqual(result["expected"][0][35:], [10] * 5)

    def test_invalid_inputs_do_not_reach_model(self):
        mutations = [
            lambda b: b.update(horizon=0),
            lambda b: b.update(horizon=True),
            lambda b: b.update(only_last_n=39),
            lambda b: b.update(threshold=float("nan")),
            lambda b: b.update(threshold=0),
            lambda b: b.update(model="other"),
            lambda b: b.update(unknown=1),
            lambda b: b["inputs"][0]["targets"][0].__setitem__(0, None),
            lambda b: b["inputs"][0]["targets"][0].__setitem__(0, float("inf")),
            lambda b: b["inputs"][0]["targets"].append([1.0]),
            lambda b: b["inputs"][0].update(past_covariates=[[1.0]]),
        ]
        for mutation in mutations:
            body = self.body()
            mutation(body)
            with self.subTest(body=str(body)[:100]), self.assertRaises(ValueError):
                self.engine.run(body, "anomalies")
        self.assertFalse(self.model.calls)

    def test_work_limit_before_dispatch(self):
        body = self.body(8192)
        body["horizon"] = 1
        with self.assertRaises(ValueError):
            self.engine.run(body, "anomalies")
        self.assertFalse(self.model.calls)

    def test_zero_width_interval_has_finite_score(self):
        def predict(item, horizon):
            return {name: [[0.0] * horizon] for name in ("expected", "lower_bound", "upper_bound")}
        self.model.predict = predict
        result = self.engine.run(self.body(), "anomalies")
        json.dumps(result, allow_nan=False)

    def test_udf_emits_objects_for_clickhouse_named_tuples(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, limit):
                return json.dumps({"predictions": [{
                    "expected": [[1.0, 2.0]],
                    "lower_bound": [[0.0, 1.0]],
                    "upper_bound": [[2.0, 3.0]],
                    "quantile_levels": [0.1, 0.5, 0.9],
                    "quantiles": [[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]],
                }]}).encode()

        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = Response()
            result = evaluate({"values": [1.0] * 8, "horizon": 2}, "forecast", "https://gateway.example", "test-key")
        self.assertEqual(result, [
            {"expected": 1.0, "lower_bound": 0.0, "upper_bound": 2.0,
             "quantiles": [{"level": 0.1, "value": 0.0}, {"level": 0.5, "value": 1.0}, {"level": 0.9, "value": 2.0}]},
            {"expected": 2.0, "lower_bound": 1.0, "upper_bound": 3.0,
             "quantiles": [{"level": 0.1, "value": 1.0}, {"level": 0.5, "value": 2.0}, {"level": 0.9, "value": 3.0}]},
        ])

    def test_multivariate_historical_targets_are_all_held_out(self):
        body = self.body()
        body["inputs"][0]["targets"].append([20.0] * 32 + [999.0] * 8)
        self.engine.run(body, "anomalies")
        self.assertNotIn(999, self.model.calls[0]["targets"][1])

    def test_udf_model_selection_from_environment(self):
        response = {"predictions": [self.model.predict({"targets": [[10.0] * 8]}, 2)]}
        with patch.dict("os.environ", {"TIMESERIES_MODEL": "timestral-209m"}), patch(
            "urllib.request.build_opener"
        ) as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            evaluate({"values": [10.0] * 8, "horizon": 2}, "forecast", "http://localhost:8091", "key")
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "timestral-209m")


if __name__ == "__main__":
    unittest.main()
