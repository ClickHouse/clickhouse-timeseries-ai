"""Executable UDF contracts for calendar-aware historical scoring."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from engine import Engine
from test_engine import RecordingModel
from udf import evaluate


class AnomalyUDFTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.engine = Engine(self.model, "a" * 40)
        self.row = {
            "values": [10.0] * 35 + [100.0] + [10.0] * 4,
            "options": json.dumps({
                "score_last_n": 8, "horizon": 4, "threshold": 3,
                "past_covariates": [list(range(40))],
                "future_covariates": [[int(i == 35) for i in range(40)]],
            }),
        }

    def run_udf(self, row):
        with patch("urllib.request.build_opener") as opener:
            response = opener.return_value.open.return_value.__enter__.return_value

            def dispatch(request, **kwargs):
                self.assertEqual(request.full_url, "https://gateway.example/v1/timeseries/anomalies")
                body = json.loads(request.data)
                self.assertEqual(body["inputs"][0]["targets"], [row["values"]])
                options = json.loads(row["options"])
                for name in ("past_covariates", "future_covariates"):
                    self.assertNotIn(name, body)
                    if name in options:
                        self.assertEqual(body["inputs"][0][name], options[name])
                for name in ("horizon", "warmup", "score_last_n", "threshold"):
                    if name in options:
                        self.assertEqual(body[name], options[name])
                response.read.return_value = json.dumps(self.engine.run(body, "anomalies")).encode()
                return opener.return_value.open.return_value

            opener.return_value.open.side_effect = dispatch
            return evaluate(row, "anomalies", "https://gateway.example", "test-key")

    def test_output_is_flat_and_aligned_and_calendar_is_causal(self):
        result = self.run_udf(self.row)
        self.assertEqual(len(result), 40)
        self.assertIsNone(result[31]["anomaly_score"])
        self.assertEqual(result[35]["anomaly_score"], 90)
        self.assertTrue(result[35]["is_anomaly"])
        self.assertNotIn("quantiles", result[35])
        self.assertEqual(len(self.model.calls[0]["targets"][0]), 32)
        self.assertNotIn(100, self.model.calls[0]["targets"][0])
        self.assertEqual(len(self.model.calls[0]["past_covariates"][0]), 32)
        self.assertEqual(self.model.calls[0]["future_covariates"][0], json.loads(self.row["options"])["future_covariates"][0][:36])

    def test_empty_covariates_match_omitted_covariates(self):
        options = {"score_last_n": 8, "horizon": 4, "threshold": 3}
        baseline = self.run_udf({**self.row, "options": json.dumps(options)})
        options.update(past_covariates=[], future_covariates=[])
        self.assertEqual(self.run_udf({**self.row, "options": json.dumps(options)}), baseline)

    def test_invalid_options_are_rejected_before_http(self):
        for options in ('[]', '{"inputs":[]}', '{"targets":[]}', '{"model":"other"}'):
            with self.subTest(options=options), patch("urllib.request.build_opener") as opener:
                with self.assertRaises(ValueError):
                    evaluate({**self.row, "options": options}, "anomalies", "https://gateway.example", "test-key")
                opener.assert_not_called()

    def test_future_calendar_length_is_validated(self):
        options = json.loads(self.row["options"])
        options["future_covariates"][0].append(0)
        with self.assertRaisesRegex(ValueError, "incorrect length"):
            self.run_udf({**self.row, "options": json.dumps(options)})
        self.assertFalse(self.model.calls)

    def test_invalid_covariates_fail_before_model_dispatch(self):
        for name in ("past_covariates", "future_covariates"):
            for value in (None, [1], [[None] * 40], [[1.0] * 39]):
                options = json.loads(self.row["options"])
                options[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.run_udf({**self.row, "options": json.dumps(options)})
        self.assertFalse(self.model.calls)

    def test_xml_keeps_the_existing_scalar_anomaly_signature(self):
        root = ET.parse(Path(__file__).resolve().parents[1] / "clickhouse" / "functions.xml").getroot()
        self.assertEqual({f.findtext("name") for f in root}, {"aiForecast", "aiForecastMultivariate", "aiDetectAnomalies"})
        function = next(f for f in root if f.findtext("name") == "aiDetectAnomalies")
        self.assertEqual(function.findtext("command"), "timeseries_udf anomalies")
        self.assertEqual([(a.findtext("name"), a.findtext("type")) for a in function.findall("argument")], [
            ("values", "Array(Float64)"), ("options", "String")])
        self.assertTrue(function.findtext("return_type").startswith("Array(Tuple("))


if __name__ == "__main__":
    unittest.main()
