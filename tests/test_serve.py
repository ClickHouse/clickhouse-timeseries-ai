import os
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from engine import Engine
from serve import MAX_BODY, create_app, main
from test_engine import RecordingModel


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        with patch.dict(os.environ, {"TIMESERIES_API_KEY": "test-key"}):
            self.app = create_app(Engine(self.model, "a" * 40))
        self.client = TestClient(self.app)
        self.headers = {"Authorization": "Bearer test-key"}
        self.body = {"model": "timesfm-3.0", "horizon": 4,
                     "inputs": [{"targets": [[10.0] * 40]}]}

    def test_missing_key_fails_startup(self):
        with patch.dict(os.environ, {"TIMESERIES_API_KEY": ""}), self.assertRaises(ValueError):
            create_app(None)

    def test_health_and_authentication(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            self.assertEqual(self.client.post("/v1/timeseries/forecast", json=self.body,
                                             headers=headers).status_code, 401)
        self.assertFalse(self.model.calls)

    def test_forecast_and_anomalies(self):
        for operation in ("forecast", "anomalies"):
            response = self.client.post("/v1/timeseries/" + operation,
                                        json=self.body, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["operation"], operation)

    def test_invalid_requests(self):
        self.assertEqual(self.client.post("/v1/timeseries/unknown", json=self.body,
                                         headers=self.headers).status_code, 404)
        self.assertEqual(self.client.post("/v1/timeseries/forecast", content=b"{",
                                         headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post("/v1/timeseries/forecast", json={},
                                         headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post("/v1/timeseries/forecast", content=b" " * (MAX_BODY + 1),
                                         headers=self.headers).status_code, 413)

    def test_model_errors_do_not_expose_input_or_traces(self):
        with patch.object(self.model, "predict", side_effect=RuntimeError("private details")):
            response = self.client.post("/v1/timeseries/forecast", json=self.body, headers=self.headers)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "model inference failed"})

    def test_oversized_output_is_rejected(self):
        with patch("engine.Engine.run", return_value={"padding": "x" * MAX_BODY}):
            response = self.client.post("/v1/timeseries/forecast", json=self.body, headers=self.headers)
        self.assertEqual(response.status_code, 413)
        self.assertIn("reduce the inputs or horizon", response.json()["error"])

    def test_busy_returns_429(self):
        entered, release = threading.Event(), threading.Event()
        original = self.model.predict

        def predict(item, horizon):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test timed out")
            return original(item, horizon)

        responses = []
        with patch.object(self.model, "predict", side_effect=predict):
            thread = threading.Thread(target=lambda: responses.append(self.client.post(
                "/v1/timeseries/forecast", json=self.body, headers=self.headers)))
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=5))
                response = self.client.post("/v1/timeseries/forecast", json=self.body, headers=self.headers)
                self.assertEqual(response.status_code, 429)
            finally:
                release.set()
                thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(responses[0].status_code, 200)

    def test_timestral_http_uses_selected_model_name(self):
        with patch.dict(os.environ, {"TIMESERIES_API_KEY": "test-key"}):
            client = TestClient(create_app(Engine(self.model, "b" * 40, "timestral-209m")))
        self.assertEqual(client.post("/v1/timeseries/forecast", json=self.body,
                                     headers=self.headers).status_code, 400)
        self.body["model"] = "timestral-209m"
        for operation in ("forecast", "anomalies"):
            response = client.post("/v1/timeseries/" + operation, json=self.body, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model"], "timestral-209m")
            self.assertEqual(response.json()["model_revision"], "b" * 40)


class CommandLineTests(unittest.TestCase):
    def test_startup_selection(self):
        for args, selected in (([], "timesfm-3.0"), (["--model", "timesfm-3.0"], "timesfm-3.0"),
                               (["--model", "timestral-209m"], "timestral-209m")):
            with self.subTest(args=args), patch("model.load_engine") as loader, patch(
                "serve.create_app", return_value="app"
            ) as factory, patch("uvicorn.run") as run, patch.dict(os.environ, {}, clear=True):
                main(args)
                loader.assert_called_once_with(selected)
                factory.assert_called_once_with(loader.return_value)
                run.assert_called_once_with("app", host="127.0.0.1", port=8091)

    def test_invalid_model_does_not_load_weights(self):
        import contextlib
        import io

        with patch("model.load_engine") as loader, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                main(["--model", "other"])
        self.assertEqual(exc.exception.code, 2)
        loader.assert_not_called()
