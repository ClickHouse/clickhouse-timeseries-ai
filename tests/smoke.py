"""Functional smoke requests against a running model service (not a benchmark)."""

import json
import math
import os
import urllib.request

from udf import evaluate


endpoint = os.environ.get("TIMESERIES_URL", "http://127.0.0.1:8091")
key = os.environ.get("TIMESERIES_API_KEY", "local-evaluation-only")
model = os.environ.get("TIMESERIES_MODEL", "timesfm-3.0")
values = [10 + math.sin(i / 5) for i in range(64)]


def request(operation, **fields):
    body = {"model": model, **fields}
    req = urllib.request.Request(endpoint + "/v1/timeseries/" + operation,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=190) as response:
        result = json.load(response)
    assert result["model"] == model
    assert len(result["model_revision"]) == 40
    assert result["usage"]["model_calls"] > 0
    print(operation, "OK", result["usage"])
    return result


forecast = request("forecast", horizon=4, inputs=[{"targets": [values]}])
assert len(forecast["predictions"][0]["expected"][0]) == 4
levels = forecast["predictions"][0]["quantile_levels"]
assert all(q in levels for q in (0.1, 0.5, 0.9))
if model == "timesfm-3.0":
    assert levels == [i / 10 for i in range(1, 10)]
elif model == "timestral-209m":
    assert levels == [0.01, 0.05, *[i / 20 for i in range(2, 19)], 0.95, 0.99]
assert len(forecast["predictions"][0]["quantiles"][0][0]) == len(levels)
multi = request("forecast", horizon=4, inputs=[{
    "targets": [values, [2 * x for x in values]],
    "past_covariates": [[float(i % 7) for i in range(64)]],
    "future_covariates": [[float(i % 7 < 2) for i in range(68)]],
}])
assert len(multi["predictions"][0]["expected"]) == 2
values[60] = 100
anomalies = request("anomalies", horizon=4, score_last_n=4, inputs=[{"targets": [values]}])
prediction = anomalies["predictions"][0]
assert prediction["expected"][0][:60] == [None] * 60
assert prediction["is_anomaly"][0][60]
assert len(prediction["expected"][0]) == 64
historical = request("anomalies", horizon=16, inputs=[{"targets": [values]}])
assert historical["usage"]["model_calls"] == 2
assert historical["predictions"][0]["expected"][0][:32] == [None] * 32
udf_result = evaluate({"values": values, "horizon": 4}, "forecast", endpoint, key)
assert len(udf_result) == 4 and len(udf_result[0]["quantiles"]) == len(levels)
print("executable UDF client OK")
